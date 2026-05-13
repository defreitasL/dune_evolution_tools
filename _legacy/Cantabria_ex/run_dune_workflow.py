from __future__ import annotations

"""Clean one-script workflow for dune proxy, geometry and erosion modelling.

Run::

    python run_dune_workflow.py

All user-editable settings live in ``dune_workflow_config.py``.

Main outputs
-------------
1. 01_profiles_with_dune_proxies.pkl
   Original profiles + dune-foot proxy + polygon width attributes.
2. 02_profiles_with_dune_geometry.pkl
   Product 01 + toe/crest/heel/berm geometry detected from each profile.
3. 03_profiles_with_eroded_dunes.pkl
   Product 02 + final eroded profile arrays ``d_dune_eroded``,
   ``z_dune_eroded``, ``x_dune_eroded`` and ``y_dune_eroded``.
"""

from pathlib import Path
import re
import traceback
from typing import Any, Mapping

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

try:
    import contextily as cx
except Exception:  # pragma: no cover - optional dependency
    cx = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

import dune_workflow_config as cfg

from dune_proxy_helpers import (
    DuneProxyConfig,
    annotate_profiles_with_dune_proxy,
    load_polygon_layer,
    load_profiles_pickle,
    prepare_dune_polygons,
    transfer_dune_polygon_width_to_results,
    normalize_geodataframe_columns,
)
from dune_geometry_ensemble import apply_to_dataset, plot_profile_diagnostics, smooth_longitudinal_feature
from dune_real_profile_bridge import (
    build_translation_geometry,
    coerce_numeric_1d,
    get_profile_arrays,
    merge_modeled_profile_into_real,
    plot_translation_summary,
    simulate_profile_event,
    summarize_simulation,
    translate_modeled_features_to_real,
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def ensure_dirs() -> None:
    for path in [
        cfg.OUTPUT_DIR,
        cfg.PLOTS_DIR,
        cfg.DETECTION_PLOTS_DIR,
        cfg.TRANSLATION_PLOTS_DIR,
        getattr(cfg, "PLANVIEW_BY_PLAYA_DIR", cfg.PLOTS_DIR),
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def progress_iter(iterable, *, total: int | None = None, desc: str = "", unit: str = "it"):
    """Return a tqdm iterator when available and enabled in the config.

    The workflow should run identically on minimal environments where tqdm is not
    installed. In that case this helper simply returns the original iterable,
    keeping progress reporting optional and non-invasive.
    """
    if not getattr(cfg, "PROGRESS_BARS", True) or tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=getattr(cfg, "PROGRESS_BAR_LEAVE", True),
    )


def clean_plot_outputs() -> None:
    """Delete stale diagnostics before a new run.

    Failed profiles do not produce translation figures. Without this cleanup,
    old PNGs from an earlier configuration can remain in the folder and make a
    failed profile look as if it still had a current model result.
    """
    if not getattr(cfg, "CLEAN_PLOTS_ON_RUN", True):
        return

    for folder in [cfg.DETECTION_PLOTS_DIR, cfg.TRANSLATION_PLOTS_DIR]:
        folder = Path(folder)
        if not folder.exists():
            continue
        for path in folder.glob("*.png"):
            path.unlink(missing_ok=True)

    planview = Path(cfg.PLANVIEW_PLOT_PATH)
    if planview.exists():
        planview.unlink(missing_ok=True)


def is_true(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "si", "sí", "s", "dune", "duna"}


def as_array(values: Any, dtype=float) -> np.ndarray:
    return np.asarray(values, dtype=dtype).copy()


def dune_crossing_mask(gdf: gpd.GeoDataFrame | pd.DataFrame) -> pd.Series:
    """Profiles that actually cross a dune polygon with finite along-profile width.

    The workflow uses the polygon intersection as the first gate. Detection and
    storm simulation are only meaningful once the profile has a real dune
    footprint, so this mask is deliberately stricter than checking whether a
    profile belongs to a beach classified as dune-dominated.
    """
    if cfg.IS_DUNE_COL not in gdf.columns:
        return pd.Series(False, index=gdf.index, dtype=bool)

    mask = gdf[cfg.IS_DUNE_COL].map(is_true).fillna(False).astype(bool)

    land_col = getattr(cfg, "POLYGON_LANDWARD_COL", getattr(cfg, "DUNE_LANDWARD_COL", "d_dune_landward_polygon"))
    sea_col = getattr(cfg, "POLYGON_SEAWARD_COL", getattr(cfg, "DUNE_SEAWARD_COL", "d_dune_seaward_polygon"))
    if land_col in gdf.columns and sea_col in gdf.columns:
        d_land = pd.to_numeric(gdf[land_col], errors="coerce")
        d_sea = pd.to_numeric(gdf[sea_col], errors="coerce")
        mask &= np.isfinite(d_land) & np.isfinite(d_sea) & (d_sea > d_land)

    return mask




def simulate_dune_decision(gdf: gpd.GeoDataFrame | pd.DataFrame) -> pd.DataFrame:
    """Return the simulation gate and diagnostic reason for each profile.

    ``is_dune`` is intentionally broad: it means that the profile intersects a
    dune polygon and should be processed by the detector.  ``simulate_dune`` is
    narrower and decides whether the erosion model may run.

    The current Cantabria rule uses the non-erodible line distance stored in the
    original profile table.  Since ``d`` increases seaward, the selected dune is
    considered modelable only when its seaward polygon edge is at, or seaward of,
    ``dist_lnero``.  If the whole dune polygon interval lies behind that line,
    the profile remains detected/diagnosed but is copied unchanged at the model
    stage.
    """
    index = gdf.index
    crossing = dune_crossing_mask(gdf)
    simulate = pd.Series(False, index=index, dtype=bool)
    reason = pd.Series("no_dune_polygon", index=index, dtype=object)

    ref_col = getattr(cfg, "SIMULATE_DUNE_REFERENCE_COL", getattr(cfg, "POLYGON_SEAWARD_COL", "d_dune_seaward_polygon"))
    lnero_col = getattr(cfg, "NON_ERODIBLE_DISTANCE_COL", "dist_lnero")
    tol = float(getattr(cfg, "SIMULATE_DUNE_TOLERANCE_M", 0.0))
    simulate_if_missing = bool(getattr(cfg, "SIMULATE_DUNE_IF_NON_ERODIBLE_MISSING", True))

    ref = pd.Series(np.nan, index=index, dtype=float)
    lnero = pd.Series(np.nan, index=index, dtype=float)

    if ref_col in gdf.columns:
        ref = pd.to_numeric(gdf[ref_col], errors="coerce")
    if lnero_col in gdf.columns:
        lnero = pd.to_numeric(gdf[lnero_col], errors="coerce")

    missing_ref = ~np.isfinite(ref)
    missing_lnero = ~np.isfinite(lnero)

    valid = crossing & ~missing_ref & ~missing_lnero
    allowed = valid & (ref >= (lnero - tol))
    protected = valid & ~allowed

    simulate.loc[allowed] = True
    reason.loc[allowed] = "simulate_dune"
    reason.loc[protected] = "dune_behind_non_erodible_line"

    no_ref = crossing & missing_ref
    reason.loc[no_ref] = f"missing_{ref_col}"

    no_lnero = crossing & ~missing_ref & missing_lnero
    if simulate_if_missing:
        simulate.loc[no_lnero] = True
        reason.loc[no_lnero] = f"simulate_missing_{lnero_col}"
    else:
        reason.loc[no_lnero] = f"missing_{lnero_col}"

    return pd.DataFrame(
        {
            getattr(cfg, "SIMULATE_DUNE_COL", "simulate_dune"): simulate,
            getattr(cfg, "SIMULATE_DUNE_REASON_COL", "simulate_dune_reason"): reason,
            "simulate_dune_reference_d": ref.astype(float),
            "simulate_dune_lnero_d": lnero.astype(float),
        },
        index=index,
    )


def add_simulate_dune_columns(gdf: gpd.GeoDataFrame | pd.DataFrame) -> gpd.GeoDataFrame | pd.DataFrame:
    """Attach/update the simulation gate columns on a profile table."""
    out = safe_copy_gdf(gdf)
    decision = simulate_dune_decision(out)
    for col in decision.columns:
        out[col] = decision[col]
    return out


def simulate_dune_mask(gdf: gpd.GeoDataFrame | pd.DataFrame) -> pd.Series:
    """Profiles allowed to enter the erosion model."""
    col = getattr(cfg, "SIMULATE_DUNE_COL", "simulate_dune")
    if col in gdf.columns:
        mask = gdf[col].map(is_true).fillna(False).astype(bool)
        return mask & dune_crossing_mask(gdf)
    return simulate_dune_decision(gdf)[col]


def profile_stem(profile_idx: Any) -> str:
    try:
        return f"profile_{int(profile_idx):04d}"
    except Exception:
        return f"profile_{str(profile_idx)}"


def safe_copy_gdf(gdf: gpd.GeoDataFrame | pd.DataFrame) -> gpd.GeoDataFrame | pd.DataFrame:
    """Copy after normalising Arrow-backed column labels.

    This avoids a GeoPandas/pandas/pyarrow compatibility error triggered by
    ``GeoDataFrame.copy()`` in some environments.
    """
    gdf = normalize_geodataframe_columns(gdf)
    return gdf.copy()


def ensure_profile_id(gdf):
    out = safe_copy_gdf(gdf)

    if cfg.PROFILE_ID_COL == "id":
        if "id" not in out.columns:
            raise KeyError("PROFILE_ID_COL='id', but the profile table has no 'id' column.")

        ids = pd.to_numeric(out["id"], errors="coerce")

        if ids.isna().any():
            bad = int(ids.isna().sum())
            raise ValueError(f"Found {bad} profiles with non-numeric or missing id values.")

        ids = ids.astype(int)

        if ids.duplicated().any():
            duplicated = ids[ids.duplicated()].unique()[:10]
            raise ValueError(f"Profile id values must be unique. Examples duplicated: {duplicated}")

        out["id"] = ids

        # Critical point:
        # detection routines use gdf.index as profile_idx, so make the index
        # equal to the external id used in QGIS.
        out = out.set_index("id", drop=False)

        # Avoid pandas ambiguity between index name and column name.
        out.index.name = None

        return out

    if cfg.PROFILE_ID_COL not in out.columns:
        out[cfg.PROFILE_ID_COL] = out.index

    return out


def ensure_detection_z_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Expose the configured elevation column under the detector's z name.

    The constrained detector works with ``row['z']`` for a compact interface,
    while the bridge/model keep using ``cfg.Z_COL`` as the authoritative vertical
    profile.
    """
    out = safe_copy_gdf(gdf)
    if cfg.Z_COL not in out.columns:
        if cfg.DETECTION_Z_COL not in out.columns:
            raise KeyError(f"Profile table must contain {cfg.Z_COL!r} or {cfg.DETECTION_Z_COL!r}.")
        return out
    out[cfg.DETECTION_Z_COL] = out[cfg.Z_COL]
    return out


def numeric(row: Mapping[str, Any], col: str, default=np.nan) -> float:
    try:
        return float(pd.to_numeric(row.get(col, default), errors="coerce"))
    except Exception:
        return float(default)


def numeric_array(value: Any) -> np.ndarray:
    """Parse profile arrays stored as arrays, lists or stringified np.float64 lists."""
    return coerce_numeric_1d(value)


def row_numeric_array(row: Mapping[str, Any], col: str) -> np.ndarray:
    if col not in row:
        return np.asarray([], dtype=float)
    return numeric_array(row[col])


def profile_xy_control_points(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distance/x/y control points for a profile.

    Some Berria profile tables store X/Y as only the two transect end points,
    while d/z are full profile arrays. In that common case, the correct way to
    locate toe/crest in plan view is to interpolate along the line between the
    two endpoints using the full d-domain, not to truncate d to the first two
    samples.
    """
    if cfg.X_COL not in row or cfg.Y_COL not in row or cfg.D_COL not in row:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    d0 = row_numeric_array(row, cfg.D_COL)
    x0 = row_numeric_array(row, cfg.X_COL)
    y0 = row_numeric_array(row, cfg.Y_COL)

    d0 = d0[np.isfinite(d0)]
    x0 = x0[np.isfinite(x0)]
    y0 = y0[np.isfinite(y0)]

    if len(d0) < 2 or len(x0) < 2 or len(y0) < 2:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    # Full X/Y arrays already aligned with d.
    if len(x0) == len(d0) and len(y0) == len(d0):
        d_ctrl, x_ctrl, y_ctrl = d0, x0, y0
    # Only two endpoint coordinates are available. Map them to the first and
    # last distance of the profile.
    elif len(x0) == 2 and len(y0) == 2:
        d_ctrl = np.asarray([float(np.nanmin(d0)), float(np.nanmax(d0))], dtype=float)
        x_ctrl = np.asarray([x0[0], x0[-1]], dtype=float)
        y_ctrl = np.asarray([y0[0], y0[-1]], dtype=float)
    else:
        n = min(len(d0), len(x0), len(y0))
        d_ctrl, x_ctrl, y_ctrl = d0[:n], x0[:n], y0[:n]

    mask = np.isfinite(d_ctrl) & np.isfinite(x_ctrl) & np.isfinite(y_ctrl)
    d_ctrl, x_ctrl, y_ctrl = d_ctrl[mask], x_ctrl[mask], y_ctrl[mask]
    if len(d_ctrl) < 2:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    order = np.argsort(d_ctrl)
    d_ctrl, x_ctrl, y_ctrl = d_ctrl[order], x_ctrl[order], y_ctrl[order]
    d_unique, idx = np.unique(d_ctrl, return_index=True)
    return d_unique, x_ctrl[idx], y_ctrl[idx]


def interpolate_profile_xy(row: Mapping[str, Any], d_target: float) -> tuple[float, float]:
    d_ctrl, x_ctrl, y_ctrl = profile_xy_control_points(row)
    if len(d_ctrl) < 2 or not np.isfinite(d_target):
        return np.nan, np.nan
    if d_target < d_ctrl.min() or d_target > d_ctrl.max():
        return np.nan, np.nan
    return (
        float(np.interp(d_target, d_ctrl, x_ctrl)),
        float(np.interp(d_target, d_ctrl, y_ctrl)),
    )


def get_aligned_xy_for_profile(row: Mapping[str, Any], d_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y arrays aligned with ``d_ref``.

    Handles both full X/Y arrays and the common two-endpoint representation.
    """
    d_ctrl, x_ctrl, y_ctrl = profile_xy_control_points(row)
    if len(d_ctrl) < 2:
        return np.full_like(d_ref, np.nan, dtype=float), np.full_like(d_ref, np.nan, dtype=float)

    x = np.full_like(d_ref, np.nan, dtype=float)
    y = np.full_like(d_ref, np.nan, dtype=float)
    valid = (d_ref >= d_ctrl.min()) & (d_ref <= d_ctrl.max())
    x[valid] = np.interp(d_ref[valid], d_ctrl, x_ctrl)
    y[valid] = np.interp(d_ref[valid], d_ctrl, y_ctrl)
    return x, y


def interpolate_profile_z(row: Mapping[str, Any], d_target: float) -> float:
    """Interpolate the observed profile elevation at a profile distance."""
    if not np.isfinite(d_target):
        return np.nan
    try:
        d, z, _ = get_profile_arrays(row, d_col=cfg.D_COL, z_col=cfg.Z_COL)
    except Exception:
        return np.nan
    if d.size < 2 or d_target < np.nanmin(d) or d_target > np.nanmax(d):
        return np.nan
    return float(np.interp(float(d_target), d, z))


def prepare_row_for_model_geometry(row: Mapping[str, Any], toe_col: str, crest_col: str, landward_col: str) -> tuple[pd.Series, dict[str, Any]]:
    """Prepare the row used by the model without changing the stored diagnostics.

    The model uses smoothed control points when available.  Polygons are kept as
    gates/search priors, while the calculation mesh is built from the topographic
    heel, crest and frontal toe selected for this profile.
    """
    model_row = row.copy() if hasattr(row, "copy") else pd.Series(dict(row))

    toe_d = numeric(model_row, toe_col)
    toe_z = interpolate_profile_z(model_row, toe_d)
    if not np.isfinite(toe_z):
        toe_z = numeric(model_row, cfg.Z_TOE_COL)

    crest_d = numeric(model_row, crest_col)
    crest_z = interpolate_profile_z(model_row, crest_d)
    if not np.isfinite(crest_z):
        crest_z = numeric(model_row, cfg.Z_CREST_COL)

    landward_d = numeric(model_row, landward_col)
    landward_z = interpolate_profile_z(model_row, landward_d)

    model_row[toe_col] = toe_d
    model_row[cfg.Z_TOE_COL] = toe_z
    model_row[crest_col] = crest_d
    model_row[cfg.Z_CREST_COL] = crest_z
    model_row["d_model_landward_boundary"] = landward_d

    toe_source = "smooth_toe" if toe_col == cfg.SMOOTHED_TOE_COL else "detected_toe"
    crest_source = "smooth_crest" if crest_col == getattr(cfg, "SMOOTHED_CREST_COL", "") else "detected_crest"
    land_source = "smooth_heel" if landward_col == getattr(cfg, "SMOOTHED_HEEL_COL", "") else landward_col
    return model_row, {
        "model_toe_source": toe_source,
        "model_crest_source": crest_source,
        "model_landward_source": land_source,
        "d_toe_input_model": float(toe_d) if np.isfinite(toe_d) else np.nan,
        "z_toe_input_model": float(toe_z) if np.isfinite(toe_z) else np.nan,
        "d_crest_input_model": float(crest_d) if np.isfinite(crest_d) else np.nan,
        "z_crest_input_model": float(crest_z) if np.isfinite(crest_z) else np.nan,
        "d_landward_input_model": float(landward_d) if np.isfinite(landward_d) else np.nan,
        "z_landward_input_model": float(landward_z) if np.isfinite(landward_z) else np.nan,
    }

def original_eroded_profile_record(row: Mapping[str, Any], status: str, message: str = "") -> dict[str, Any]:
    """Record original profile as the eroded profile for skipped/failed cases."""
    rec = {
        "dune_model_status": status,
        "dune_model_message": message,
        "dune_model_ran": False,
        "model_toe_source": "",
        "model_crest_source": "",
        "beach_slope_source": "",
        "d_toe_input_model": np.nan,
        "z_toe_input_model": np.nan,
        "d_crest_input_model": np.nan,
        "z_crest_input_model": np.nan,
        "d_toe_model": np.nan,
        "z_toe_model": np.nan,
        "d_crest_model": np.nan,
        "z_crest_model": np.nan,
        "x0_final": np.nan,
        "z0_final": np.nan,
        "Ds_final": np.nan,
        "tan_beta_f_model": np.nan,
        "tan_beta_f_observed": np.nan,
        "tan_beta_D_model": np.nan,
        "tan_beta_eff_est_model": np.nan,
        "alpha_rep_deg_model": np.nan,
        "landward_crest_width_m_model": np.nan,
        "landward_back_slope_m_model": np.nan,
        "tan_beta_back_model": np.nan,
        "real_volume_above_0_m2": np.nan,
        "calc_volume_above_0_m2": np.nan,
        "calc_volume_residual_m2": np.nan,
        "calc_volume_match_status": "",
        "calc_initial_volume_above_0_m2": np.nan,
        "calc_final_volume_above_0_m2": np.nan,
        "calc_target_volume_change_m2": np.nan,
        "merged_volume_change_m2": np.nan,
        "volume_scale_factor": np.nan,
        "dune_merge_landward_limit_d": np.nan,
        "dune_merge_initial_landward_limit_d": np.nan,
        "dune_merge_final_landward_limit_d": np.nan,
        "dune_merge_seaward_limit_d": np.nan,
        "dune_merge_datum_offset_m": np.nan,
        "d_toe_shift_m": np.nan,
        "d_crest_shift_m": np.nan,
        "z_crest_change_m": np.nan,
        "d_landward_polygon_model": np.nan,
        "z_landward_polygon_model": np.nan,
        "dune_merge_strategy": "",
        "volume_eroded_front_m2": np.nan,
        "volume_deposited_landward_m2": np.nan,
        "volume_balance_error_m2": np.nan,
        "overwash_volume_available_m2": np.nan,
        "d_dune_eroded": np.nan,
        "z_dune_eroded": np.nan,
        "x_dune_eroded": np.nan,
        "y_dune_eroded": np.nan,
    }

    if not cfg.COPY_ORIGINAL_PROFILE_WHEN_SKIPPED:
        return rec

    try:
        d_real, z_real, _ = get_profile_arrays(row, d_col=cfg.D_COL, z_col=cfg.Z_COL)
        x_real, y_real = get_aligned_xy_for_profile(row, d_real)
        rec.update(
            {
                "d_dune_eroded": as_array(d_real),
                "z_dune_eroded": as_array(z_real),
                "x_dune_eroded": as_array(x_real),
                "y_dune_eroded": as_array(y_real),
            }
        )
    except Exception as exc:
        rec["dune_model_message"] = f"{message} | Could not copy original profile arrays: {exc}"
    return rec


# -----------------------------------------------------------------------------
# Step 1: proxies and polygon widths
# -----------------------------------------------------------------------------
def build_dune_proxies() -> gpd.GeoDataFrame:
    print("\n[1/3] Building dune proxies and polygon widths")
    profiles = normalize_geodataframe_columns(load_profiles_pickle(str(cfg.PROFILES_PATH)))
    profiles = ensure_profile_id(profiles)

    structures = normalize_geodataframe_columns(load_polygon_layer(str(cfg.STRUCTURES_PATH)))
    class_col = cfg.STRUCTURE_CLASS_COL if cfg.STRUCTURE_CLASS_COL in structures.columns else None
    dunes = prepare_dune_polygons(
        structures,
        profiles_crs=profiles.crs,
        class_col=class_col,
        dune_values=cfg.DUNE_VALUES,
        assume_all_if_missing=cfg.ASSUME_ALL_POLYGONS_ARE_DUNES_IF_NO_CLASS,
        fallback_crs=cfg.STRUCTURES_FALLBACK_CRS,
    )

    proxy_cfg = DuneProxyConfig(
        d_col=cfg.D_COL,
        geometry_col=profiles.geometry.name,
        x_col=cfg.X_COL,
        y_col=cfg.Y_COL,
        output_cross_col=cfg.IS_DUNE_COL,
    )

    out = annotate_profiles_with_dune_proxy(
        profiles,
        dunes,
        config=proxy_cfg,
        exact_fallback=cfg.EXACT_PROXY_FALLBACK,
        show_progress=getattr(cfg, "PROGRESS_BARS", True),
        progress_desc="Dune polygon intersections",
    )
    out = ensure_profile_id(out)
    out = add_simulate_dune_columns(out)
    out.to_pickle(cfg.PROXIES_PKL)

    n_dune = int(dune_crossing_mask(out).sum())
    n_sim = int(simulate_dune_mask(out).sum())
    print(f"Saved: {cfg.PROXIES_PKL}")
    print(f"Profiles: {len(out)} | profiles crossing dune polygons: {n_dune} | profiles marked for simulation: {n_sim}")
    if getattr(cfg, "SIMULATE_DUNE_REASON_COL", "simulate_dune_reason") in out.columns:
        print(out[getattr(cfg, "SIMULATE_DUNE_REASON_COL", "simulate_dune_reason")].value_counts(dropna=False))
    return out


# -----------------------------------------------------------------------------
# Step 2: profile-based dune geometry
# -----------------------------------------------------------------------------
def save_detection_plot(gdf_for_detection: gpd.GeoDataFrame, profile_idx: Any) -> None:
    if not cfg.SAVE_DETECTION_PLOTS:
        return
    fig = None
    try:
        ret = plot_profile_diagnostics(
            gdf_for_detection,
            profile_idx,
            **cfg.DETECTION_KWARGS,
            **cfg.DETECTION_PLOT_KWARGS,
        )
        # Current helper returns (fig, axes, summary); older versions returned
        # (fig, axes). Accept both.
        fig = ret[0] if isinstance(ret, tuple) else ret
        fig.savefig(cfg.DETECTION_PLOTS_DIR / f"{profile_stem(profile_idx)}_detection.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        if fig is not None:
            plt.close(fig)
        print(f"  [profile {profile_idx}] detection plot failed: {type(exc).__name__}: {exc}")


def build_dune_geometry(proxies: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    print("\n[2/3] Detecting dune toe/crest/heel geometry")
    if proxies is None:
        proxies = normalize_geodataframe_columns(pd.read_pickle(cfg.PROXIES_PKL))
    proxies = ensure_profile_id(proxies)
    proxies = add_simulate_dune_columns(proxies)
    gdf_det = ensure_detection_z_column(proxies)

    dune_mask = dune_crossing_mask(gdf_det)
    gdf_det_dunes = gdf_det.loc[dune_mask].copy()
    n_candidates = int(dune_mask.sum())
    print(f"Detection candidates crossing dune polygons: {n_candidates}/{len(gdf_det)}")

    if gdf_det_dunes.empty:
        results = pd.DataFrame(columns=["profile_idx"])
    else:
        results = apply_to_dataset(
            gdf_det_dunes,
            show_progress=getattr(cfg, "PROGRESS_BARS", True),
            progress_desc="Detecting dune geometry",
            **cfg.DETECTION_KWARGS,
        )

        # Add polygon-derived widths and estimated crest platform width into the
        # geometry table. This is only done for true dune-polygon crossings; all
        # other profiles stay as plain skipped records downstream.
        results = transfer_dune_polygon_width_to_results(
            results,
            proxies.loc[dune_mask],
            result_profile_col="profile_idx",
            profile_id_col=cfg.PROFILE_ID_COL,
            crest_col=cfg.CREST_COL,
            toe_col=cfg.TOE_COL,
            crest_width_fraction=cfg.CREST_WIDTH_FRACTION,
            min_crest_width_m=cfg.MIN_CREST_WIDTH_M,
            max_crest_width_m=cfg.MAX_CREST_WIDTH_M,
        )

        if getattr(cfg, "SMOOTH_TOE_LONGITUDINALLY", True):
            results = smooth_longitudinal_feature(
                results,
                gdf=gdf_det_dunes,
                feature_col=cfg.TOE_COL,
                profile_col="profile_idx",
                out_col=cfg.SMOOTHED_TOE_COL,
                show_progress=getattr(cfg, "PROGRESS_BARS", True),
                progress_desc="Smoothing toe by beach",
                **getattr(cfg, "TOE_SMOOTHING_KWARGS", {}),
            )

        if getattr(cfg, "SMOOTH_CREST_LONGITUDINALLY", True):
            results = smooth_longitudinal_feature(
                results,
                gdf=gdf_det_dunes,
                feature_col=cfg.CREST_COL,
                profile_col="profile_idx",
                out_col=getattr(cfg, "SMOOTHED_CREST_COL", "d_crest_smooth"),
                show_progress=getattr(cfg, "PROGRESS_BARS", True),
                progress_desc="Smoothing crest by beach",
                **getattr(cfg, "CREST_SMOOTHING_KWARGS", getattr(cfg, "TOE_SMOOTHING_KWARGS", {})),
            )

        if getattr(cfg, "SMOOTH_HEEL_LONGITUDINALLY", True):
            results = smooth_longitudinal_feature(
                results,
                gdf=gdf_det_dunes,
                feature_col=getattr(cfg, "HEEL_COL", "d_heel_final"),
                profile_col="profile_idx",
                out_col=getattr(cfg, "SMOOTHED_HEEL_COL", "d_heel_final_smooth"),
                show_progress=getattr(cfg, "PROGRESS_BARS", True),
                progress_desc="Smoothing heel by beach",
                **getattr(cfg, "HEEL_SMOOTHING_KWARGS", getattr(cfg, "TOE_SMOOTHING_KWARGS", {})),
            )

    if results.empty:
        results_indexed = pd.DataFrame()
    else:
        # Smoothing is applied sequentially to toe, crest and heel. Depending on
        # the pandas version, repeated metadata merges can leave duplicated
        # column labels in the result table. Remove them before mapping back;
        # otherwise ``results_indexed[col]`` may be a DataFrame, and
        # ``Series.map`` fails with cryptic errors such as ``ValueError: 2``.
        if results.columns.duplicated().any():
            results = results.loc[:, ~results.columns.duplicated()]
        results_indexed = results.drop_duplicates("profile_idx").set_index("profile_idx")

    out = safe_copy_gdf(proxies)
    profile_ids = out[cfg.PROFILE_ID_COL]
    for col in results_indexed.columns:
        if col == getattr(out.geometry, "name", None):
            continue

        mapper = results_indexed[col]
        if isinstance(mapper, pd.DataFrame):
            # Last-resort guard for duplicated labels that may survive upstream.
            mapper = mapper.iloc[:, 0]

        # Mapping through a plain dict is more tolerant of object-valued
        # diagnostics than passing a Series directly to pandas.Series.map.
        out[col] = profile_ids.map(mapper.to_dict())


    out.to_pickle(cfg.GEOMETRY_PKL)
    print(f"Saved: {cfg.GEOMETRY_PKL}")
    if results.empty:
        n_errors = n_with_toe = n_with_crest = 0
    else:
        n_errors = int(results.get("error", pd.Series(dtype=object)).notna().sum())
        n_with_toe = int(pd.to_numeric(results.get(cfg.TOE_COL, pd.Series(dtype=float)), errors="coerce").notna().sum())
        n_with_crest = int(pd.to_numeric(results.get(cfg.CREST_COL, pd.Series(dtype=float)), errors="coerce").notna().sum())

    print(f"Geometry detection errors on dune candidates: {n_errors}/{n_candidates}")
    print(f"Profiles skipped before detection: {len(out) - n_candidates}/{len(out)}")
    print(f"Dune candidates with finite toe/crest: {n_with_toe}/{n_candidates} toe | {n_with_crest}/{n_candidates} crest")

    if cfg.SAVE_DETECTION_PLOTS and not gdf_det_dunes.empty:
        print(f"Saving detection plots in: {cfg.DETECTION_PLOTS_DIR}")
        iterator = progress_iter(
            gdf_det_dunes.iterrows(),
            total=len(gdf_det_dunes),
            desc="Saving detection plots",
            unit="profile",
        )
        for profile_index, _ in iterator:
            save_detection_plot(gdf_det_dunes, profile_index)

    return out


# -----------------------------------------------------------------------------
# Step 3: model and profile translation
# -----------------------------------------------------------------------------
def toe_column_for_row(row: Mapping[str, Any]) -> str:
    """Use the smoothed toe when available; otherwise fall back to the raw toe."""
    if np.isfinite(numeric(row, cfg.SMOOTHED_TOE_COL)):
        return cfg.SMOOTHED_TOE_COL
    return cfg.TOE_COL


def crest_column_for_row(row: Mapping[str, Any]) -> str:
    """Use the alongshore-smoothed crest when available."""
    smooth_col = getattr(cfg, "SMOOTHED_CREST_COL", "d_crest_smooth")
    if np.isfinite(numeric(row, smooth_col)):
        return smooth_col
    return cfg.CREST_COL


def landward_column_for_row(row: Mapping[str, Any]) -> str:
    """Prefer a topographic heel that is safely landward of the crest."""
    crest_col = crest_column_for_row(row)
    d_crest = numeric(row, crest_col)
    min_gap = float(getattr(cfg, "MIN_HEEL_CREST_GAP_M", 2.0))
    for col in [getattr(cfg, "SMOOTHED_HEEL_COL", "d_heel_final_smooth"), getattr(cfg, "HEEL_COL", "d_heel_final")]:
        d_land = numeric(row, col)
        if np.isfinite(d_land) and np.isfinite(d_crest) and d_land < d_crest - min_gap:
            return col
    return getattr(cfg, "POLYGON_LANDWARD_COL", cfg.DUNE_LANDWARD_COL)


def run_model_one_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    profile_idx = row.get(cfg.PROFILE_ID_COL, getattr(row, "name", -1))

    if not is_true(row.get(cfg.IS_DUNE_COL, False)):
        return original_eroded_profile_record(row, "skipped_not_dune", "is_dune is False")

    sim_col = getattr(cfg, "SIMULATE_DUNE_COL", "simulate_dune")
    if sim_col in row and not is_true(row.get(sim_col, False)):
        reason_col = getattr(cfg, "SIMULATE_DUNE_REASON_COL", "simulate_dune_reason")
        return original_eroded_profile_record(row, "skipped_not_simulate_dune", str(row.get(reason_col, "simulate_dune is False")))

    toe_col = toe_column_for_row(row)
    crest_col = crest_column_for_row(row)
    landward_col = landward_column_for_row(row)
    if not np.isfinite(numeric(row, toe_col)):
        return original_eroded_profile_record(row, "skipped_no_toe", f"Missing {toe_col}")
    if not np.isfinite(numeric(row, crest_col)):
        return original_eroded_profile_record(row, "skipped_no_crest", f"Missing {crest_col}")
    if not np.isfinite(numeric(row, landward_col)):
        return original_eroded_profile_record(row, "skipped_no_landward_boundary", f"Missing {landward_col}")

    try:
        model_row, geometry_source = prepare_row_for_model_geometry(row, toe_col, crest_col, landward_col)

        geom = build_translation_geometry(
            model_row,
            model_row,
            d_col=cfg.D_COL,
            z_col=cfg.Z_COL,
            toe_col=toe_col,
            crest_col=crest_col,
            berm_col=cfg.BERM_COL,
            z_toe_col=cfg.Z_TOE_COL,
            z_crest_col=cfg.Z_CREST_COL,
            dune_landward_col="d_model_landward_boundary",
            beach_slope_col=cfg.BEACH_SLOPE_COL if cfg.USE_INPUT_BEACH_SLOPE else None,
            beach_slope_min=cfg.BEACH_SLOPE_MIN,
            beach_slope_max=cfg.BEACH_SLOPE_MAX,
            seaward_window_m=getattr(cfg, "BEACH_SLOPE_SEAWARD_WINDOW_M", 50.0),
            enforce_model_slope_separation=cfg.ENFORCE_MODEL_SLOPE_SEPARATION,
            min_dune_to_beach_slope_ratio=cfg.MIN_DUNE_TO_BEACH_SLOPE_RATIO,
            min_dune_to_beach_slope_gap=cfg.MIN_DUNE_TO_BEACH_SLOPE_GAP,
            max_model_dune_face_slope=cfg.MAX_MODEL_DUNE_FACE_SLOPE,
            min_landward_back_slope_length_m=cfg.MIN_LANDWARD_BACK_SLOPE_LENGTH_M,
            refine_crest=cfg.REFINE_CREST_FROM_PROFILE,
            crest_refine_window_m=cfg.CREST_REFINE_WINDOW_M,
            crest_refine_min_gain_m=cfg.CREST_REFINE_MIN_GAIN_M,
            min_toe_crest_gap_m=cfg.MIN_TOE_CREST_GAP_M,
            min_crest_toe_relief_m=cfg.MIN_CREST_TOE_RELIEF_M,
        )

        model, params, result = simulate_profile_event(
            geom,
            time_s=cfg.TIME_S,
            T=cfg.T,
            TWL=cfg.TWL,
            Ru=cfg.RU,
            H0=cfg.H0,
            base_params=cfg.BASE_PARAMS,
            runup_mode=cfg.RUNUP_MODE,
        )

        proj = merge_modeled_profile_into_real(
            row,
            geom,
            model,
            result,
            t_idx=-1,
            d_col=cfg.D_COL,
            z_col=cfg.Z_COL,
            blend_width_m=cfg.BLEND_WIDTH_M,
            dune_landward_col="d_model_landward_boundary",
            max_vertical_change_m=cfg.MAX_VERTICAL_CHANGE_M,
            max_retreat_m=cfg.MAX_MODEL_RETREAT_M,
            max_crest_lowering_m=cfg.MAX_CREST_LOWERING_M,
        )
        features = translate_modeled_features_to_real(geom, result, t_idx=-1)
        summary = summarize_simulation(geom, params, result, t_idx=-1)

        d_eroded = as_array(proj["d_real"])
        z_eroded = as_array(proj["z_merged"])
        x_eroded, y_eroded = get_aligned_xy_for_profile(row, d_eroded)

        rec = {
            "dune_model_status": "ok",
            "dune_model_message": "",
            "dune_model_ran": True,
            **geometry_source,
            **features,
            "x0_final": summary.get("x0_final", np.nan),
            "z0_final": summary.get("z0_final", np.nan),
            "Ds_final": summary.get("Ds_final", np.nan),
            "tan_beta_f_model": geom.tan_beta_f,
            "tan_beta_f_observed": geom.tan_beta_f_observed,
            "tan_beta_D_model": geom.tan_beta_D,
            "tan_beta_eff_est_model": geom.tan_beta_eff_est,
            "alpha_rep_deg_model": geom.alpha_rep_deg,
            "model_crest_source": geom.crest_source,
            "beach_slope_source": geom.beach_slope_source,
            "landward_crest_width_m_model": geom.landward_crest_width_m,
            "landward_back_slope_m_model": geom.landward_back_slope_m,
            "tan_beta_back_model": geom.tan_beta_back,
            "real_volume_above_0_m2": geom.real_volume_above_0_m2,
            "calc_volume_above_0_m2": geom.calc_volume_above_0_m2,
            "calc_volume_residual_m2": geom.calc_volume_residual_m2,
            "calc_volume_match_status": geom.calc_volume_match_status,
            "calc_initial_volume_above_0_m2": proj.get("calc_initial_volume_above_0_m2", np.nan),
            "calc_final_volume_above_0_m2": proj.get("calc_final_volume_above_0_m2", np.nan),
            "calc_target_volume_change_m2": proj.get("calc_target_volume_change_m2", np.nan),
            "merged_volume_change_m2": proj.get("merged_volume_change_m2", np.nan),
            "volume_scale_factor": proj.get("volume_scale_factor", np.nan),
            "dune_merge_landward_limit_d": proj["landward_limit_d"],
            "dune_merge_initial_landward_limit_d": proj.get("initial_landward_limit_d", np.nan),
            "dune_merge_final_landward_limit_d": proj.get("final_landward_limit_d", np.nan),
            "dune_merge_seaward_limit_d": proj["seaward_limit_d"],
            "dune_merge_datum_offset_m": proj.get("merge_datum_offset_m", np.nan),
            "d_toe_shift_m": proj.get("d_toe_shift_m", np.nan),
            "d_crest_shift_m": proj.get("d_crest_shift_m", np.nan),
            "z_crest_change_m": proj.get("z_crest_change_m", np.nan),
            "d_landward_polygon_model": geom.d_landward0,
            "z_landward_polygon_model": geom.z_landward0,
            "dune_merge_strategy": str(proj.get("merge_strategy", np.array(["translated_final_profile_smooth"]))[0]),
            "volume_eroded_front_m2": proj.get("volume_eroded_front_m2", np.nan),
            "volume_deposited_landward_m2": proj.get("volume_deposited_landward_m2", np.nan),
            "volume_balance_error_m2": proj.get("volume_balance_error_m2", np.nan),
            "overwash_volume_available_m2": proj.get("overwash_volume_available_m2", np.nan),
            "d_dune_eroded": d_eroded,
            "z_dune_eroded": z_eroded,
            "x_dune_eroded": as_array(x_eroded),
            "y_dune_eroded": as_array(y_eroded),
        }

        if cfg.SAVE_TRANSLATION_PLOTS:
            try:
                fig, _ = plot_translation_summary(
                    row,
                    geom,
                    model,
                    result,
                    t_idx=-1,
                    d_col=cfg.D_COL,
                    z_col=cfg.Z_COL,
                    blend_width_m=cfg.BLEND_WIDTH_M,
                    dune_landward_col="d_model_landward_boundary",
                    max_vertical_change_m=cfg.MAX_VERTICAL_CHANGE_M,
                    max_retreat_m=cfg.MAX_MODEL_RETREAT_M,
                    max_crest_lowering_m=cfg.MAX_CREST_LOWERING_M,
                    **cfg.TRANSLATION_PLOT_KWARGS,
                )
                fig.suptitle(f"Profile {profile_idx} | real → model → real", y=0.995)
                fig.savefig(
                    cfg.TRANSLATION_PLOTS_DIR / f"{profile_stem(profile_idx)}_translation.png",
                    dpi=160,
                    bbox_inches="tight",
                )
                plt.close(fig)
            except Exception as exc:
                print(f"  [profile {profile_idx}] translation plot failed: {type(exc).__name__}: {exc}")

        return rec

    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        print(f"  [profile {profile_idx}] model failed: {message}")
        traceback.print_exc(limit=2)
        return original_eroded_profile_record(row, "failed", message)


def run_dune_model_batch(geometry_gdf: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    print("\n[3/3] Running dune model and translating eroded profile back to real profiles")
    if geometry_gdf is None:
        geometry_gdf = normalize_geodataframe_columns(pd.read_pickle(cfg.GEOMETRY_PKL))
    geometry_gdf = ensure_profile_id(geometry_gdf)
    geometry_gdf = add_simulate_dune_columns(geometry_gdf)

    dune_mask = dune_crossing_mask(geometry_gdf)
    sim_mask = simulate_dune_mask(geometry_gdf)
    n_dune = int(dune_mask.sum())
    n_candidates = int(sim_mask.sum())
    print(f"Model candidates crossing dune polygons: {n_dune}/{len(geometry_gdf)}")
    print(f"Profiles marked for simulation: {n_candidates}/{len(geometry_gdf)}")

    records: list[dict[str, Any]] = []
    iterator = progress_iter(
        geometry_gdf.iterrows(),
        total=len(geometry_gdf),
        desc="Running model/translation",
        unit="profile",
    )
    for _, row in iterator:
        if not bool(dune_mask.loc[row.name]):
            records.append(
                original_eroded_profile_record(
                    row,
                    "skipped_not_dune_polygon",
                    "Profile does not cross a finite dune polygon interval",
                )
            )
            continue

        if not bool(sim_mask.loc[row.name]):
            reason_col = getattr(cfg, "SIMULATE_DUNE_REASON_COL", "simulate_dune_reason")
            reason = row.get(reason_col, "simulate_dune is False")
            records.append(
                original_eroded_profile_record(
                    row,
                    "skipped_not_simulate_dune",
                    str(reason),
                )
            )
            continue

        records.append(run_model_one_profile(row))

    model_df = pd.DataFrame(records, index=geometry_gdf.index)

    # Keep output compact: original/proxy/geometry columns + final eroded profile
    # and final toe/crest model diagnostics only. No heavy model time-series.
    out = geometry_gdf.drop(columns=[c for c in model_df.columns if c in geometry_gdf.columns], errors="ignore")
    out = pd.concat([out, model_df], axis=1)
    out = gpd.GeoDataFrame(out, geometry=geometry_gdf.geometry.name, crs=geometry_gdf.crs)
    out.to_pickle(cfg.ERODED_PKL)

    print(f"Saved: {cfg.ERODED_PKL}")
    print(out["dune_model_status"].value_counts(dropna=False))
    return out


# -----------------------------------------------------------------------------
# Plan-view plot
# -----------------------------------------------------------------------------
def _points_from_dz(gdf: gpd.GeoDataFrame, d_col: str, x_col: str, y_col: str, out_col: str) -> gpd.GeoDataFrame:
    rows = []
    for _, row in gdf.iterrows():
        d_target = numeric(row, d_col)
        xx, yy = interpolate_profile_xy(row, d_target)
        if not np.isfinite(xx) or not np.isfinite(yy):
            continue
        rows.append({
            cfg.PROFILE_ID_COL: row[cfg.PROFILE_ID_COL],
            "feature": out_col,
            "geometry": gpd.points_from_xy([xx], [yy])[0],
        })
    if not rows:
        return gpd.GeoDataFrame(
            pd.DataFrame(columns=[cfg.PROFILE_ID_COL, "feature", "geometry"]),
            geometry="geometry",
            crs=gdf.crs,
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def _points_from_eroded(gdf: gpd.GeoDataFrame, d_model_col: str, feature_name: str) -> gpd.GeoDataFrame:
    rows = []
    for _, row in gdf.iterrows():
        d_target = numeric(row, d_model_col)
        if not np.isfinite(d_target):
            continue

        xx = yy = np.nan
        try:
            d = row_numeric_array(row, "d_dune_eroded")
            x = row_numeric_array(row, "x_dune_eroded")
            y = row_numeric_array(row, "y_dune_eroded")
            n = min(len(d), len(x), len(y))
            d, x, y = d[:n], x[:n], y[:n]
            mask = np.isfinite(d) & np.isfinite(x) & np.isfinite(y)
            d, x, y = d[mask], x[mask], y[mask]
            if len(d) >= 2:
                order = np.argsort(d)
                d, x, y = d[order], x[order], y[order]
                d_unique, ii = np.unique(d, return_index=True)
                if d_unique.min() <= d_target <= d_unique.max():
                    xx = float(np.interp(d_target, d_unique, x[ii]))
                    yy = float(np.interp(d_target, d_unique, y[ii]))
        except Exception:
            pass

        # Robust fallback: eroded profiles keep the same plan-view transect, so
        # the original profile X/Y mapping is valid when eroded x/y arrays were
        # produced by older workflow versions or are incomplete.
        if not np.isfinite(xx) or not np.isfinite(yy):
            xx, yy = interpolate_profile_xy(row, d_target)

        if not np.isfinite(xx) or not np.isfinite(yy):
            continue
        rows.append({
            cfg.PROFILE_ID_COL: row[cfg.PROFILE_ID_COL],
            "feature": feature_name,
            "geometry": gpd.points_from_xy([xx], [yy])[0],
        })
    if not rows:
        return gpd.GeoDataFrame(
            pd.DataFrame(columns=[cfg.PROFILE_ID_COL, "feature", "geometry"]),
            geometry="geometry",
            crs=gdf.crs,
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def _line_from_points(points: gpd.GeoDataFrame):
    from shapely.geometry import LineString

    if points.empty:
        return None
    pts = points.sort_values(cfg.PROFILE_ID_COL).geometry.tolist()
    if len(pts) < 2:
        return None
    return LineString([(p.x, p.y) for p in pts])


def _safe_filename(value: Any) -> str:
    """Small filename sanitizer for beach names."""
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ_.-]+", "", text)
    return text or "unknown"


def _find_playa_column(gdf: gpd.GeoDataFrame) -> str | None:
    requested = getattr(cfg, "PLANVIEW_PLAYA_COL", "Playa")
    if requested in gdf.columns:
        return requested
    requested_low = str(requested).lower()
    for col in gdf.columns:
        if str(col).lower() == requested_low:
            return col
    return None


def _resolve_planview_basemap_source():
    source = getattr(cfg, "PLANVIEW_BASEMAP_SOURCE", None)
    if cx is None:
        return source
    if isinstance(source, str):
        key = source.strip().lower()
        if key in {"esri.worldimagery", "worldimagery", "esri_worldimagery"}:
            return cx.providers.Esri.WorldImagery
    return source


def _plot_planview_initial_final_one(gdf: gpd.GeoDataFrame, out_path: Path, title: str) -> None:
    """Draw one plan-view panel for a single beach or for the full dataset."""
    if cfg.X_COL not in gdf.columns or cfg.Y_COL not in gdf.columns:
        print("Plan-view plot skipped: profile X/Y columns not found.")
        return

    gdf_plot = gdf.copy()
    if cfg.IS_DUNE_COL in gdf_plot.columns:
        mask = dune_crossing_mask(gdf_plot)
        gdf_plot = gdf_plot.loc[mask].copy()

    if gdf_plot.empty:
        print(f"Plan-view plot skipped for {title}: no dune-polygon profiles.")
        return

    toe0 = _points_from_dz(gdf_plot, cfg.TOE_COL, cfg.X_COL, cfg.Y_COL, "initial toe")
    crest0 = _points_from_dz(gdf_plot, cfg.CREST_COL, cfg.X_COL, cfg.Y_COL, "initial crest")
    toe1 = _points_from_eroded(gdf_plot, "d_toe_model", "final toe")
    crest1 = _points_from_eroded(gdf_plot, "d_crest_model", "final crest")

    crs_plot = cfg.PLANVIEW_PLOT_CRS
    layers = []
    for pts in [toe0, crest0, toe1, crest1]:
        if not pts.empty:
            layers.append(pts.to_crs(crs_plot))

    if not layers:
        print(f"Plan-view plot skipped for {title}: no valid toe/crest points.")
        return

    all_pts = pd.concat(layers)
    all_pts = gpd.GeoDataFrame(all_pts, geometry="geometry", crs=crs_plot)
    minx, miny, maxx, maxy = all_pts.total_bounds
    dx = max(maxx - minx, 1.0)
    dy = max(maxy - miny, 1.0)
    buf = cfg.PLANVIEW_EXTENT_BUFFER_RATIO

    fig, ax = plt.subplots(figsize=cfg.PLANVIEW_FIGSIZE)

    style = {
        "initial toe": dict(marker="o", markersize=16, label="initial toe"),
        "initial crest": dict(marker="^", markersize=16, label="initial crest"),
        "final toe": dict(marker="o", markersize=22, label="final toe"),
        "final crest": dict(marker="^", markersize=22, label="final crest"),
    }

    for feature, pts in [
        ("initial toe", toe0),
        ("initial crest", crest0),
        ("final toe", toe1),
        ("final crest", crest1),
    ]:
        if pts.empty:
            continue
        pts_p = pts.to_crs(crs_plot).sort_values(cfg.PROFILE_ID_COL)
        line = _line_from_points(pts_p)
        if line is not None:
            gpd.GeoSeries([line], crs=crs_plot).plot(ax=ax, linewidth=1.6, alpha=0.85)
        pts_p.plot(ax=ax, **style[feature])

    ax.set_xlim(minx - dx * buf, maxx + dx * buf)
    ax.set_ylim(miny - dy * buf, maxy + dy * buf)

    if cfg.PLANVIEW_BASEMAP and cx is not None:
        try:
            cx.add_basemap(ax, source=_resolve_planview_basemap_source(), crs=crs_plot)
        except Exception as exc:
            print(f"  basemap failed for {title}: {type(exc).__name__}: {exc}")

    ax.set_title(title)
    ax.set_xlabel(f"X ({crs_plot})")
    ax.set_ylabel(f"Y ({crs_plot})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_planview_initial_final(gdf: gpd.GeoDataFrame) -> None:
    if not cfg.SAVE_PLANVIEW_PLOT:
        return

    if getattr(cfg, "SAVE_GLOBAL_PLANVIEW_PLOT", True):
        print(f"Saving global plan-view toe/crest plot: {cfg.PLANVIEW_PLOT_PATH}")
        _plot_planview_initial_final_one(
            gdf,
            Path(cfg.PLANVIEW_PLOT_PATH),
            "Dune toe and crest: initial detection vs final model",
        )

    if not getattr(cfg, "PLANVIEW_BY_PLAYA", False):
        return

    playa_col = _find_playa_column(gdf)
    if playa_col is None:
        print("Plan-view by beach skipped: beach identifier column not found.")
        return

    out_dir = Path(getattr(cfg, "PLANVIEW_BY_PLAYA_DIR", cfg.PLOTS_DIR / "planview_by_playa"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving plan-view toe/crest plots by {playa_col!r}: {out_dir}")
    values = sorted(gdf[playa_col].dropna().unique(), key=lambda x: str(x))
    for value in progress_iter(values, total=len(values), desc="Saving planviews by beach", unit="beach"):
        g = gdf.loc[gdf[playa_col] == value].copy()
        if g.empty:
            continue
        filename = f"planview_toe_crest_initial_final_{_safe_filename(value)}.png"
        _plot_planview_initial_final_one(
            g,
            out_dir / filename,
            f"{value} | dune toe and crest: initial vs final",
        )

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def run_workflow() -> gpd.GeoDataFrame:
    ensure_dirs()
    clean_plot_outputs()
    proxies = build_dune_proxies()
    geometry = build_dune_geometry(proxies)
    eroded = run_dune_model_batch(geometry)
    plot_planview_initial_final(eroded)

    print("\nWorkflow complete.")
    print(f"01 proxies : {cfg.PROXIES_PKL}")
    print(f"02 geometry: {cfg.GEOMETRY_PKL}")
    print(f"03 eroded  : {cfg.ERODED_PKL}")
    if cfg.SAVE_DETECTION_PLOTS:
        print(f"Detection plots   : {cfg.DETECTION_PLOTS_DIR}")
    if cfg.SAVE_TRANSLATION_PLOTS:
        print(f"Translation plots : {cfg.TRANSLATION_PLOTS_DIR}")
    if cfg.SAVE_PLANVIEW_PLOT:
        print(f"Plan-view plot    : {cfg.PLANVIEW_PLOT_PATH}")
    return eroded


if __name__ == "__main__":
    run_workflow()
