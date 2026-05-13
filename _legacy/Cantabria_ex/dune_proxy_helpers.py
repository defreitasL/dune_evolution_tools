from __future__ import annotations

import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, GeometryCollection

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


NP_FLOAT64_RE = re.compile(r"np\.float64\(([-+0-9.eE]+)\)")

COMMON_CLASS_COLUMNS = (
    "Clase",
    "clase",
    "CLASS",
    "class",
    "tipo",
    "Tipo",
    "categoria",
    "Categoría",
    "category",
    "label",
)


@dataclass
class DuneProxyConfig:
    d_col: str = "d"
    geometry_col: str = "geometry"
    x_col: str = "X"
    y_col: str = "Y"
    output_cross_col: str = "is_dune"
    output_proxy_col: str = "Y_df_AI_proxy"
    output_idx_col: str = "idx_df_AI_proxy"
    output_x_col: str = "x_df_AI_proxy"
    output_y_col: str = "y_df_AI_proxy"
    output_method_col: str = "df_AI_proxy_method"

    # Exact polygon/profile intersection diagnostics.
    # These are measured along the profile distance axis, with d increasing seaward.
    # Therefore d_dune_seaward is the seaward edge of the dune polygon and
    # d_dune_landward is its landward edge. The width is their difference.
    output_width_col: str = "dune_polygon_width_m"
    output_d_landward_col: str = "d_dune_landward_polygon"
    output_d_seaward_col: str = "d_dune_seaward_polygon"
    output_x_landward_col: str = "x_dune_landward_polygon"
    output_y_landward_col: str = "y_dune_landward_polygon"
    output_x_seaward_col: str = "x_dune_seaward_polygon"
    output_y_seaward_col: str = "y_dune_seaward_polygon"
    output_interval_count_col: str = "dune_polygon_n_intervals"
    output_interval_rank_col: str = "dune_polygon_interval_rank"




def _progress_iter(iterable, *, total: int | None = None, desc: str = "", unit: str = "it", show_progress: bool = False):
    if not show_progress or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


def normalize_geodataframe_columns(gdf: gpd.GeoDataFrame | pd.DataFrame) -> gpd.GeoDataFrame | pd.DataFrame:
    """Return *gdf* with a plain object ``Index`` for column labels.

    Some profile pickles written with newer pandas/pyarrow combinations can carry
    an Arrow-backed column index. Recent GeoPandas versions call
    ``(gdf.columns == "geometry").sum()`` during ``GeoDataFrame.copy()``; with an
    Arrow-backed comparison this raises ``AttributeError: 'ArrowExtensionArray'
    object has no attribute 'sum'``. Normalising the labels to a regular pandas
    object Index avoids that bug without touching the data columns themselves.
    """
    try:
        cols = pd.Index([str(c) for c in list(gdf.columns)], dtype=object)
        gdf.columns = cols
    except Exception:
        pass

    if isinstance(gdf, gpd.GeoDataFrame):
        geom_name = getattr(gdf, "_geometry_column_name", None)
        if geom_name is not None:
            geom_name = str(geom_name)
            if geom_name in gdf.columns:
                try:
                    gdf = gdf.set_geometry(geom_name, crs=gdf.crs)
                except Exception:
                    pass
    return gdf


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _profile_exchange_candidates(path: str) -> list[str]:
    """Return stable exchange-format alternatives for a profile pickle.

    Pickle is convenient inside one Python environment, but it is not a robust
    exchange format across pandas/geopandas versions. This matters for the
    optional pybeach environment, which intentionally uses pandas<2.0. If a
    profile table was pickled with a newer stack, the old environment may fail
    before the workflow even reaches the detector.
    """
    base = os.path.splitext(path)[0]
    candidates = [
        base + ".parquet",
        base + ".gpkg",
        base + ".geojson",
        base + ".json",
        base + ".feather",
    ]
    return [candidate for candidate in candidates if os.path.exists(candidate)]


def _read_profiles_exchange(path: str) -> gpd.GeoDataFrame:
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    if suffix in {".gpkg", ".geojson", ".json", ".shp"}:
        return gpd.read_file(path)
    if suffix == ".feather":
        return gpd.read_feather(path)
    raise ValueError(f"Unsupported profile exchange format: {path!r}")


def _read_pickle_with_stringdtype_compat(path: str):
    """Read a pickle while tolerating a narrow pandas StringDtype mismatch."""
    original_init = pd.StringDtype.__init__

    def _compat_init(self, storage=None, na_value=pd.NA):
        # pandas<2 does not accept the extra na_value argument used by some
        # newer pickles. Keep the storage argument and ignore na_value.
        return original_init(self, storage=storage)

    pd.StringDtype.__init__ = _compat_init
    try:
        return pd.read_pickle(path)
    finally:
        pd.StringDtype.__init__ = original_init


def load_profiles_pickle(path: str) -> gpd.GeoDataFrame:
    """Read a profiles GeoDataFrame.

    The historical name is kept for compatibility with the workflow, but this
    function now accepts both pickle and stable exchange formats such as
    GeoParquet/GeoPackage. That is intentional: pickle is not guaranteed to be
    readable when moving from the main modelling environment to the optional
    pybeach environment, because pybeach requires an older pandas stack.
    """
    suffix = os.path.splitext(path)[1].lower()

    if suffix not in {".pkl", ".pickle"}:
        gdf = _read_profiles_exchange(path)
    else:
        try:
            gdf = pd.read_pickle(path)
        except TypeError as exc:
            # Some pandas-version mismatches can be handled by patching
            # StringDtype. If that still fails, fall through to the clearer
            # exchange-format error below.
            try:
                gdf = _read_pickle_with_stringdtype_compat(path)
            except Exception as compat_exc:
                alternatives = _profile_exchange_candidates(path)
                if alternatives:
                    warnings.warn(
                        "Could not read the profile pickle in this environment; "
                        f"using exchange-format fallback instead: {alternatives[0]}",
                        RuntimeWarning,
                    )
                    gdf = _read_profiles_exchange(alternatives[0])
                else:
                    raise TypeError(
                        "Could not read the profiles pickle in this Python environment. "
                        "This usually happens when the file was written with a newer "
                        "pandas/geopandas stack and is being read in the pybeach "
                        "environment with pandas<2.0. Convert the profiles once from "
                        "the environment that can read the pickle, preferably to "
                        "GeoParquet, and then set PROFILES_PATH to that file. Example:\n\n"
                        "  conda activate yates_opt\n"
                        "  python - <<'PY'\n"
                        "  import pandas as pd\n"
                        "  gdf = pd.read_pickle('Cantabria_profiles_35m_checked.pkl')\n"
                        "  gdf.to_parquet('Cantabria_profiles_35m_checked.parquet', index=False)\n"
                        "  PY\n\n"
                        "Then in dune_workflow_config.py use:\n"
                        "  PROFILES_PATH = DATA_DIR / 'Cantabria_profiles_35m_checked.parquet'"
                    ) from compat_exc
        except ImportError as exc:
            msg = str(exc)
            if "PyArrow backed StringArray" in msg or "pyarrow" in msg.lower():
                raise ImportError(
                    "No se pudo abrir el pickle porque fue guardado con strings tipo PyArrow y "
                    "en este entorno falta pyarrow o no es compatible. "
                    "Solución recomendada: instalar pyarrow o convertir el GeoDataFrame de perfiles "
                    "a un formato más robusto para intercambio, como parquet."
                ) from exc
            raise

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"El archivo {path!r} no contiene un GeoDataFrame.")
    return normalize_geodataframe_columns(gdf)


def load_polygon_layer(path: str, restore_shx: bool = True) -> gpd.GeoDataFrame:
    """
    Read polygon layer.

    For shapefiles, it is enough to pass the `.shp` path as long as the
    sidecar files (`.dbf`, `.shx`, `.prj`, optionally `.cpg`) sit next to it.
    """
    if restore_shx:
        os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
    return normalize_geodataframe_columns(gpd.read_file(path))


# ---------------------------------------------------------------------
# String / metadata helpers
# ---------------------------------------------------------------------
def normalize_label(value) -> str:
    """Lowercase + trim + strip accents for robust class matching."""
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def detect_class_column(
    structures: gpd.GeoDataFrame,
    dune_values: Iterable[str] = ("duna",),
    preferred: Optional[str] = None,
) -> Optional[str]:
    """
    Detect the most likely attribute column that stores structure classes.

    Strategy
    --------
    1) If `preferred` exists, use it.
    2) Check common column names.
    3) Check object/string columns whose values overlap with `dune_values`.
    """
    if preferred is not None:
        return preferred if preferred in structures.columns else None

    dune_values_norm = {normalize_label(v) for v in dune_values}

    for col in COMMON_CLASS_COLUMNS:
        if col in structures.columns:
            return col

    for col in structures.columns:
        if col == structures.geometry.name:
            continue
        series = structures[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            values_norm = set(series.dropna().astype(str).map(normalize_label).unique().tolist())
            if values_norm & dune_values_norm:
                return col

    return None


def summarize_structure_classes(
    structures: gpd.GeoDataFrame,
    class_col: Optional[str] = None,
    top_n: int = 20,
) -> pd.Series:
    """Return a value-count summary of the class field, auto-detecting it if needed."""
    col = detect_class_column(structures, preferred=class_col)
    if col is None:
        raise ValueError("No se pudo identificar automáticamente la columna de clases.")
    return structures[col].value_counts(dropna=False).head(top_n)


# ---------------------------------------------------------------------
# CRS / filtering helpers
# ---------------------------------------------------------------------
def looks_like_lonlat_bounds(bounds: Sequence[float]) -> bool:
    minx, miny, maxx, maxy = bounds
    return (-180 <= minx <= 180) and (-180 <= maxx <= 180) and (-90 <= miny <= 90) and (-90 <= maxy <= 90)


def ensure_crs(
    gdf: gpd.GeoDataFrame,
    fallback_crs: Optional[str] = None,
    auto_lonlat_to_epsg4326: bool = True,
) -> gpd.GeoDataFrame:
    """Ensure the GeoDataFrame has a CRS."""
    if gdf.crs is not None:
        return gdf

    if fallback_crs is not None:
        return gdf.set_crs(fallback_crs, allow_override=True)

    if auto_lonlat_to_epsg4326 and looks_like_lonlat_bounds(gdf.total_bounds):
        warnings.warn(
            "La capa no tiene CRS. Se asumirá EPSG:4326 porque sus bounds parecen lon/lat.",
            stacklevel=2,
        )
        return gdf.set_crs("EPSG:4326", allow_override=True)

    raise ValueError("La capa no tiene CRS y no se pudo inferir automáticamente.")


def prepare_dune_polygons(
    structures: gpd.GeoDataFrame,
    profiles_crs,
    class_col: Optional[str] = None,
    dune_values: Iterable[str] = ("duna",),
    assume_all_if_missing: bool = False,
    fallback_crs: Optional[str] = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Keep dune polygons only and reproject them to the profiles CRS.

    Parameters
    ----------
    class_col : str or None
        Name of the attribute that identifies the structure type. If None,
        the helper tries to detect it automatically.
    dune_values : iterable of str
        Values that will be interpreted as dunes. Matching is case/accent insensitive.
    assume_all_if_missing : bool
        If True and no valid class column is found, all polygons are treated as dunes.
    """
    if structures.empty:
        raise ValueError("La capa de estructuras está vacía.")

    structures = ensure_crs(structures, fallback_crs=fallback_crs)
    structures = structures[structures.geometry.notna()].copy()
    structures = structures[structures.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    class_col_found = detect_class_column(structures, dune_values=dune_values, preferred=class_col)
    dune_values_norm = {normalize_label(v) for v in dune_values}

    if class_col_found is not None:
        cls = structures[class_col_found].astype(str).map(normalize_label)
        dunes = structures.loc[cls.isin(dune_values_norm)].copy()
    else:
        if not assume_all_if_missing:
            raise ValueError(
                "No se encontró una columna de clase utilizable. "
                "Indica class_col explícitamente o activa assume_all_if_missing=True."
            )
        warnings.warn(
            "No se pudo filtrar por tipo de estructura; se usarán todos los polígonos como si fueran 'duna'.",
            stacklevel=2,
        )
        dunes = structures.copy()

    if dunes.empty:
        if class_col_found is not None:
            raise ValueError(
                f"No quedaron polígonos tras filtrar {class_col_found!r} con valores {tuple(dune_values)!r}."
            )
        raise ValueError("No quedaron polígonos de duna tras el filtrado.")

    if profiles_crs is None:
        raise ValueError("El GeoDataFrame de perfiles no tiene CRS.")

    return dunes.to_crs(profiles_crs)


# ---------------------------------------------------------------------
# Parsing / sampling helpers
# ---------------------------------------------------------------------
def parse_npfloat_list_string(value) -> Optional[np.ndarray]:
    """
    Parse strings like:
        '[np.float64(1.0), np.float64(2.0)]'
    or common Python list-like strings.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    if not isinstance(value, str):
        return None

    matches = NP_FLOAT64_RE.findall(value)
    if matches:
        return np.asarray([float(m) for m in matches], dtype=float)

    stripped = value.strip().strip("[]")
    if not stripped:
        return np.asarray([], dtype=float)
    try:
        return np.asarray([float(x) for x in stripped.split(",")], dtype=float)
    except Exception:
        return None


def get_profile_distance_array(row: pd.Series, d_col: str = "d") -> np.ndarray:
    d = row[d_col]
    if isinstance(d, np.ndarray):
        return d.astype(float)
    return np.asarray(d, dtype=float)


def get_profile_sampling_xy(
    row: pd.Series,
    d_col: str = "d",
    geometry_col: str = "geometry",
    x_col: str = "X",
    y_col: str = "Y",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return X, Y, d arrays.

    Priority:
    1) Use explicit X/Y arrays if their length matches d.
    2) Otherwise interpolate along the LineString geometry using d.
    """
    d = get_profile_distance_array(row, d_col=d_col)

    x = parse_npfloat_list_string(row.get(x_col))
    y = parse_npfloat_list_string(row.get(y_col))
    if x is not None and y is not None and len(x) == len(d) and len(y) == len(d):
        return x, y, d

    line = row[geometry_col]
    if line is None or line.is_empty:
        raise ValueError("Perfil sin geometría válida.")

    d_clip = np.clip(d, 0.0, float(line.length))
    pts = [line.interpolate(float(di)) for di in d_clip]
    x = np.asarray([pt.x for pt in pts], dtype=float)
    y = np.asarray([pt.y for pt in pts], dtype=float)
    return x, y, d


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def _union_geometries(geometries: gpd.GeoSeries):
    """Return a union that works across old and new GeoPandas versions."""
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return geometries.unary_union


def _candidate_union(dunes: gpd.GeoDataFrame, geom) -> Optional[object]:
    sidx = dunes.sindex
    if sidx is None:
        return _union_geometries(dunes.geometry)
    idx = list(sidx.intersection(geom.bounds))
    if not idx:
        return None
    cand = dunes.iloc[idx]
    cand = cand[cand.intersects(geom)]
    if cand.empty:
        return None
    return _union_geometries(cand.geometry)


def _projected_distances_from_intersection(line: LineString, inter) -> list[float]:
    if inter is None or inter.is_empty:
        return []

    distances: list[float] = []

    if isinstance(inter, Point):
        distances.append(float(line.project(inter)))
    elif isinstance(inter, MultiPoint):
        distances.extend(float(line.project(pt)) for pt in inter.geoms)
    elif isinstance(inter, LineString):
        coords = list(inter.coords)
        if coords:
            distances.append(float(line.project(Point(coords[0]))))
            distances.append(float(line.project(Point(coords[-1]))))
    elif isinstance(inter, MultiLineString):
        for g in inter.geoms:
            distances.extend(_projected_distances_from_intersection(line, g))
    elif isinstance(inter, GeometryCollection):
        for g in inter.geoms:
            distances.extend(_projected_distances_from_intersection(line, g))
    else:
        try:
            for g in inter.geoms:
                distances.extend(_projected_distances_from_intersection(line, g))
        except Exception:
            pass

    return distances


def _projected_intervals_from_intersection(line: LineString, inter, *, min_width: float = 1e-6) -> list[tuple[float, float]]:
    """Return profile-distance intervals where ``line`` lies inside/intersects a polygon.

    The output is a list of ``(d_min, d_max)`` intervals measured along ``line``.
    Points/tangential contacts are ignored because they have zero width and do not
    represent a dune body crossed by the profile.
    """
    if inter is None or inter.is_empty:
        return []

    intervals: list[tuple[float, float]] = []

    if isinstance(inter, LineString):
        coords = list(inter.coords)
        if len(coords) >= 2:
            d0 = float(line.project(Point(coords[0])))
            d1 = float(line.project(Point(coords[-1])))
            lo, hi = sorted((d0, d1))
            if hi - lo > min_width:
                intervals.append((lo, hi))
    elif isinstance(inter, MultiLineString):
        for g in inter.geoms:
            intervals.extend(_projected_intervals_from_intersection(line, g, min_width=min_width))
    elif isinstance(inter, GeometryCollection):
        for g in inter.geoms:
            intervals.extend(_projected_intervals_from_intersection(line, g, min_width=min_width))
    else:
        try:
            for g in inter.geoms:
                intervals.extend(_projected_intervals_from_intersection(line, g, min_width=min_width))
        except Exception:
            pass

    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda t: t[0])
    merged: list[list[float]] = [[intervals[0][0], intervals[0][1]]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + min_width:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    return [(float(lo), float(hi)) for lo, hi in merged if hi - lo > min_width]


def _select_seaward_dune_interval(intervals: list[tuple[float, float]]) -> tuple[float, float, int] | None:
    """Pick the seaward-most dune interval for profiles where d increases seaward."""
    if not intervals:
        return None
    ranked = sorted(enumerate(intervals), key=lambda item: item[1][1], reverse=True)
    rank0, (lo, hi) = ranked[0]
    return float(lo), float(hi), int(rank0)


def _empty_dune_proxy_record(config: DuneProxyConfig, *, crosses: bool, method: str) -> dict:
    """Consistent empty output for no geometry/no intersection/error cases."""
    return {
        config.output_cross_col: bool(crosses),
        config.output_proxy_col: np.nan,
        config.output_idx_col: np.nan,
        config.output_x_col: np.nan,
        config.output_y_col: np.nan,
        config.output_method_col: method,
        config.output_width_col: np.nan,
        config.output_d_landward_col: np.nan,
        config.output_d_seaward_col: np.nan,
        config.output_x_landward_col: np.nan,
        config.output_y_landward_col: np.nan,
        config.output_x_seaward_col: np.nan,
        config.output_y_seaward_col: np.nan,
        config.output_interval_count_col: 0,
        config.output_interval_rank_col: np.nan,
    }


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------
def compute_dune_proxy_for_profile(
    row: pd.Series,
    dunes: gpd.GeoDataFrame,
    config: DuneProxyConfig = DuneProxyConfig(),
    exact_fallback: bool = True,
) -> dict:
    """
    For one profile, compute the dune proxy and the full polygon width.

    Returned distance fields follow the profile convention used in the dune
    workflow: ``d`` increases seaward. The selected dune interval is the
    seaward-most polygon/profile intersection, so:

    - ``Y_df_AI_proxy`` / ``d_dune_seaward_polygon``: seaward polygon edge.
    - ``d_dune_landward_polygon``: landward polygon edge.
    - ``dune_polygon_width_m``: distance between both edges along the profile.

    The previous sampled-point proxy is preserved when possible, but the exact
    polygon interval is always used to compute the width and edge coordinates.
    """
    line = row[config.geometry_col]
    if line is None or line.is_empty:
        return _empty_dune_proxy_record(config, crosses=False, method="no_geometry")

    dune_union = _candidate_union(dunes, line)
    if dune_union is None:
        return _empty_dune_proxy_record(config, crosses=False, method="no_intersection")

    crosses = bool(dune_union.intersects(line))
    if not crosses:
        return _empty_dune_proxy_record(config, crosses=False, method="no_intersection")

    inter = line.intersection(dune_union)
    intervals = _projected_intervals_from_intersection(line, inter)
    selected = _select_seaward_dune_interval(intervals)

    if selected is not None:
        d_landward, d_seaward, interval_rank = selected
        p_landward = line.interpolate(d_landward)
        p_seaward = line.interpolate(d_seaward)
        width_m = max(0.0, float(d_seaward - d_landward))
    else:
        d_landward = d_seaward = np.nan
        interval_rank = np.nan
        p_landward = p_seaward = None
        width_m = np.nan

    x, y, d = get_profile_sampling_xy(
        row,
        d_col=config.d_col,
        geometry_col=config.geometry_col,
        x_col=config.x_col,
        y_col=config.y_col,
    )

    proxy_d = np.nan
    proxy_idx = np.nan
    proxy_x = np.nan
    proxy_y = np.nan
    method = "crosses_but_no_sample_hit"

    order = np.argsort(d)[::-1]
    for j in order:
        if selected is not None:
            if not (d_landward - 1e-6 <= float(d[j]) <= d_seaward + 1e-6):
                continue
        pt = Point(float(x[j]), float(y[j]))
        if dune_union.intersects(pt):
            proxy_d = float(d[j])
            proxy_idx = int(j)
            proxy_x = float(x[j])
            proxy_y = float(y[j])
            method = "sampled_point"
            break

    if (not np.isfinite(proxy_d)) and exact_fallback and selected is not None:
        proxy_d = float(d_seaward)
        proxy_idx = np.nan
        proxy_x = float(p_seaward.x)
        proxy_y = float(p_seaward.y)
        method = "exact_interval_edge"
    elif (not np.isfinite(proxy_d)) and exact_fallback:
        distances = _projected_distances_from_intersection(line, inter)
        if distances:
            proxy_d = float(np.max(distances))
            pt = line.interpolate(proxy_d)
            proxy_idx = np.nan
            proxy_x = float(pt.x)
            proxy_y = float(pt.y)
            method = "exact_fallback_zero_width"

    return {
        config.output_cross_col: crosses,
        config.output_proxy_col: proxy_d,
        config.output_idx_col: proxy_idx,
        config.output_x_col: proxy_x,
        config.output_y_col: proxy_y,
        config.output_method_col: method,
        config.output_width_col: width_m,
        config.output_d_landward_col: d_landward,
        config.output_d_seaward_col: d_seaward,
        config.output_x_landward_col: np.nan if p_landward is None else float(p_landward.x),
        config.output_y_landward_col: np.nan if p_landward is None else float(p_landward.y),
        config.output_x_seaward_col: np.nan if p_seaward is None else float(p_seaward.x),
        config.output_y_seaward_col: np.nan if p_seaward is None else float(p_seaward.y),
        config.output_interval_count_col: int(len(intervals)),
        config.output_interval_rank_col: interval_rank,
    }


def annotate_profiles_with_dune_proxy(
    profiles: gpd.GeoDataFrame,
    dunes: gpd.GeoDataFrame,
    config: DuneProxyConfig = DuneProxyConfig(),
    exact_fallback: bool = True,
    show_progress: bool = False,
    progress_desc: str = "Dune polygon intersections",
) -> gpd.GeoDataFrame:
    """Annotate all profiles with dune-crossing, proxy dune-foot and polygon width."""
    profiles = normalize_geodataframe_columns(profiles)
    dunes = normalize_geodataframe_columns(dunes)
    out = profiles.copy()
    iterator = _progress_iter(
        out.iterrows(),
        total=len(out),
        desc=progress_desc,
        unit="profile",
        show_progress=show_progress,
    )
    records = [
        compute_dune_proxy_for_profile(row, dunes=dunes, config=config, exact_fallback=exact_fallback)
        for _, row in iterator
    ]
    out = pd.concat([out, pd.DataFrame(records, index=out.index)], axis=1)
    return gpd.GeoDataFrame(out, geometry=profiles.geometry.name, crs=profiles.crs)


def transfer_dune_polygon_width_to_results(
    results: pd.DataFrame,
    profiles_with_dunes: gpd.GeoDataFrame | pd.DataFrame,
    *,
    result_profile_col: str = "profile_idx",
    profile_id_col: str | None = None,
    width_col: str = "dune_polygon_width_m",
    d_landward_col: str = "d_dune_landward_polygon",
    d_seaward_col: str = "d_dune_seaward_polygon",
    crest_col: str = "d_crest",
    toe_col: str = "d_toe_final",
    out_width_col: str = "dune_polygon_width_m",
    out_landward_col: str = "d_dune_landward_polygon",
    out_seaward_col: str = "d_dune_seaward_polygon",
    out_landward_from_crest_col: str = "dune_width_landward_of_crest_m",
    out_frontface_width_col: str = "dune_frontface_width_m",
    out_crest_width_col: str = "crest_width_from_polygon_m",
    crest_width_fraction: float = 0.35,
    min_crest_width_m: float = 0.0,
    max_crest_width_m: float | None = 20.0,
) -> pd.DataFrame:
    """Join polygon-derived dune width columns into a detection-results table.

    This is the bridge between the plan-view polygon database and the profile
    geometry table used by ``build_translation_geometry``.

    ``crest_width_from_polygon_m`` is only an initial estimate of the flat crest
    platform. It is computed as a fraction of the polygon width landward of the
    detected crest, because the polygon gives total dune footprint width, not the
    morphological crest-platform width directly.
    """
    out = results.copy()
    prof = profiles_with_dunes.copy()

    if profile_id_col is None:
        if result_profile_col in prof.columns:
            profile_id_col = result_profile_col
        else:
            prof = prof.copy()
            profile_id_col = "__profile_index__"
            prof[profile_id_col] = prof.index

    missing = [c for c in [profile_id_col, width_col, d_landward_col, d_seaward_col] if c not in prof.columns]
    if missing:
        raise KeyError(f"profiles_with_dunes is missing required columns: {missing}")
    if result_profile_col not in out.columns:
        raise KeyError(f"results is missing {result_profile_col!r}")

    join_cols = [profile_id_col, width_col, d_landward_col, d_seaward_col]
    prof_join = prof[join_cols].copy()
    prof_join = prof_join.rename(
        columns={
            profile_id_col: result_profile_col,
            width_col: out_width_col,
            d_landward_col: out_landward_col,
            d_seaward_col: out_seaward_col,
        }
    )

    out = out.drop(columns=[c for c in prof_join.columns if c != result_profile_col and c in out.columns], errors="ignore")
    out = out.merge(prof_join, on=result_profile_col, how="left")

    d_land = pd.to_numeric(out[out_landward_col], errors="coerce")
    d_sea = pd.to_numeric(out[out_seaward_col], errors="coerce")
    d_crest = pd.to_numeric(out[crest_col], errors="coerce") if crest_col in out.columns else np.nan
    d_toe = pd.to_numeric(out[toe_col], errors="coerce") if toe_col in out.columns else d_sea

    out[out_frontface_width_col] = np.maximum(0.0, d_toe - d_crest)
    out[out_landward_from_crest_col] = np.maximum(0.0, d_crest - d_land)

    crest_width = crest_width_fraction * out[out_landward_from_crest_col].astype(float)
    crest_width = np.maximum(float(min_crest_width_m), crest_width)
    if max_crest_width_m is not None:
        crest_width = np.minimum(float(max_crest_width_m), crest_width)

    valid = np.isfinite(d_land) & np.isfinite(d_sea) & (d_sea > d_land)
    out[out_crest_width_col] = np.where(valid, crest_width, np.nan)
    return out


# ---------------------------------------------------------------------
# Optional QA plot helper
# ---------------------------------------------------------------------
def plot_profile_and_dunes(
    profile_row: pd.Series,
    dunes: gpd.GeoDataFrame,
    proxy_x_col: str = "x_df_AI_proxy",
    proxy_y_col: str = "y_df_AI_proxy",
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    dunes.plot(ax=ax, alpha=0.35, edgecolor="k")
    gpd.GeoSeries([profile_row.geometry], crs=dunes.crs).plot(ax=ax, color="red", linewidth=2)

    px = profile_row.get(proxy_x_col)
    py = profile_row.get(proxy_y_col)
    if pd.notna(px) and pd.notna(py):
        gpd.GeoSeries([Point(px, py)], crs=dunes.crs).plot(ax=ax, markersize=60)

    ax.set_aspect("equal")
    ax.set_title(f"Perfil {profile_row.get('id', profile_row.name)}")
    return fig, ax



def plot_all_profiles_and_dunes(
    profiles: gpd.GeoDataFrame,
    dunes: gpd.GeoDataFrame,
    proxy_x_col: str = "x_df_AI_proxy",
    proxy_y_col: str = "y_df_AI_proxy",
    proxy_dist_col: str = "Y_df_AI_proxy",
    class_col: str = "Clase",
    dune_value: str = "Duna",
    extent_buffer_ratio: float = 0.03,
    basemap: bool = True,
    basemap_source=None,
    ax=None,
):
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as ctx
    from shapely.geometry import Point

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    profiles_plot = profiles.copy()
    dunes_plot = dunes.copy()

    # --- Filtrar dunas ---
    if class_col in dunes_plot.columns:
        dunes_plot = dunes_plot[
            dunes_plot[class_col].astype(str).str.strip().str.lower() == dune_value.lower()
        ].copy()

    # --- CRS checks ---
    if profiles_plot.crs is None:
        raise ValueError("`profiles` no tiene CRS definido.")
    if dunes_plot.crs is None:
        raise ValueError("`dunes` no tiene CRS definido.")

    if dunes_plot.crs != profiles_plot.crs:
        dunes_plot = dunes_plot.to_crs(profiles_plot.crs)

    # =========================================================
    # Extensión del mapa en el CRS nativo de los perfiles
    # =========================================================
    minx, miny, maxx, maxy = profiles_plot.total_bounds
    dx = maxx - minx
    dy = maxy - miny

    bx = dx * extent_buffer_ratio if dx > 0 else 10.0
    by = dy * extent_buffer_ratio if dy > 0 else 10.0

    # =========================================================
    # Plot dunas (solo borde)
    # =========================================================
    if not dunes_plot.empty:
        dunes_plot.boundary.plot(
            ax=ax,
            color="goldenrod",
            linewidth=1.5,
            label="Dunas",
            zorder=3,
        )

    # =========================================================
    # Plot perfiles
    # =========================================================
    profiles_plot.plot(
        ax=ax,
        color="red",
        linewidth=1.2,
        alpha=0.9,
        label="Perfiles",
        zorder=4,
    )

    # =========================================================
    # Construcción robusta de puntos proxy
    # =========================================================
    proxy_points = None

    # Opción 1: usar directamente columnas x/y proxy
    if proxy_x_col in profiles.columns and proxy_y_col in profiles.columns:
        valid_xy = profiles[[proxy_x_col, proxy_y_col]].copy()
        valid_xy[proxy_x_col] = pd.to_numeric(valid_xy[proxy_x_col], errors="coerce")
        valid_xy[proxy_y_col] = pd.to_numeric(valid_xy[proxy_y_col], errors="coerce")
        valid_mask = valid_xy[proxy_x_col].notna() & valid_xy[proxy_y_col].notna()

        if valid_mask.any():
            proxy_points = gpd.GeoDataFrame(
                profiles.loc[valid_mask].drop(columns="geometry", errors="ignore").copy(),
                geometry=gpd.points_from_xy(
                    valid_xy.loc[valid_mask, proxy_x_col],
                    valid_xy.loc[valid_mask, proxy_y_col],
                ),
                crs=profiles.crs,
            )

    # Opción 2: reconstruir desde Y_df_AI_proxy + arrays X/Y/d
    if proxy_points is None and proxy_dist_col in profiles.columns:
        if all(c in profiles.columns for c in ["X", "Y", "d"]):
            pts = []

            for _, row in profiles.iterrows():
                dist_target = pd.to_numeric(row.get(proxy_dist_col), errors="coerce")
                if pd.isna(dist_target):
                    continue

                try:
                    X = np.asarray(row["X"], dtype=float)
                    Y = np.asarray(row["Y"], dtype=float)
                    d = np.asarray(row["d"], dtype=float)
                except Exception:
                    continue

                if len(X) == 0 or len(Y) == 0 or len(d) == 0:
                    continue
                if not (len(X) == len(Y) == len(d)):
                    continue

                idx = np.nanargmin(np.abs(d - dist_target))
                if np.isfinite(X[idx]) and np.isfinite(Y[idx]):
                    pts.append(Point(X[idx], Y[idx]))

            if len(pts) > 0:
                proxy_points = gpd.GeoDataFrame(
                    geometry=pts,
                    crs=profiles.crs,
                )

    # Plot proxies si existen
    if proxy_points is not None and not proxy_points.empty:
        proxy_points.plot(
            ax=ax,
            markersize=28,
            marker="o",
            color="cyan",
            edgecolor="black",
            linewidth=0.5,
            label="Proxy pie de duna",
            zorder=5,
        )

    # =========================================================
    # Limitar vista antes del basemap
    # =========================================================
    ax.set_xlim(minx - bx, maxx + bx)
    ax.set_ylim(miny - by, maxy + by)

    # =========================================================
    # Basemap
    # =========================================================
    if basemap:
        if basemap_source is None:
            basemap_source = ctx.providers.Esri.WorldImagery

        ctx.add_basemap(
            ax,
            source=basemap_source,
            crs=profiles_plot.crs,
            reset_extent=False,
        )

    ax.set_title("Perfiles, dunas y proxy de pie de duna")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend()
    return fig, ax
