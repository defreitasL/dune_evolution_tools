from __future__ import annotations

"""Run the dune erosion model directly from ``dune_params`` parquet outputs.

This module is the bridge between the parameter-identification package
(``dune_params``) and the event-scale dune erosion model. It deliberately does
**not** run any proxy extraction or toe/crest/heel detection. The expected input
is the final parquet produced by ``dune_params`` with profile arrays and detected
geometry already attached.

Typical use
-----------
Single profile::

    from dune_evolution_tools import DuneToeStormParams
    from dune_evolution_tools.dune_params_workflow import (
        simulate_dune_profile_from_dune_params_parquet,
    )

    d_final, z_final, V_to_beach = simulate_dune_profile_from_dune_params_parquet(
        "cantabria_dune_parameters.parquet",
        profile_id=570,
        time_s=time_s,
        TWL=twl,
        Hs0=hs0,
        T=11.0,
        base_params=DuneToeStormParams(
            Ds=5.0, z0_init=3.0, tan_beta_f=0.05,
            Cs=1.8e-3, A_overwash=3.0,
            crest_erosion=True, k_crest=0.7, crest_width_m=10.0,
        ),
    )

Batch workflow::

    out = run_dune_model_from_dune_params_parquet(
        "cantabria_dune_parameters.parquet",
        time_s=time_s,
        TWL=twl,
        Hs0=hs0,
        T=11.0,
        output_path="cantabria_dune_model_outputs.parquet",
    )

The returned profile is always the complete original profile mesh ``(d, z)``
with the modelled dune segment transferred back to that real profile.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence
import traceback

import numpy as np
import pandas as pd

try:  # GeoPandas is needed for reading/writing GeoParquet, but keep imports lazy-safe.
    import geopandas as gpd
except Exception:  # pragma: no cover - environments without geopandas can still import this module
    gpd = None

from .params import DuneToeStormParams
from .real_profile_bridge import (
    DEFAULT_Z_CANDIDATES,
    build_translation_geometry,
    coerce_numeric_1d,
    get_profile_arrays,
    interp_on_profile,
    merge_modeled_profile_into_real,
    simulate_profile_event,
    summarize_simulation,
    translate_modeled_features_to_real,
)


@dataclass(frozen=True)
class DuneParamsModelConfig:
    """Column names and modelling options for ``dune_params`` → model.

    Defaults match the current Cantabria workflow and the example parquet
    produced by ``dune_params``.
    """

    # Core columns from dune_params parquet
    profile_id_col: str = "id"
    d_col: str = "d"
    z_col: str = "z_corregido"
    z_candidates: tuple[str, ...] = ("z_corregido", "z", "z_full", "z_chill")
    x_col: str = "X"
    y_col: str = "Y"
    is_dune_col: str = "is_dune"

    # Detected/smoothed morphology columns
    toe_col: str = "d_toe_final"
    smoothed_toe_col: str = "d_toe_final_smooth"
    crest_col: str = "d_crest"
    smoothed_crest_col: str = "d_crest_smooth"
    heel_col: str = "d_heel_final"
    smoothed_heel_col: str = "d_heel_final_smooth"
    berm_col: str = "d_berm"
    z_toe_col: str = "z_toe_final"
    z_crest_col: str = "z_crest"

    # Dune polygon interval columns already computed by dune_params
    polygon_landward_col: str = "d_dune_landward_polygon"
    polygon_seaward_col: str = "d_dune_seaward_polygon"

    # Optional simulation gate, recomputed if the parquet does not contain it
    simulate_dune_col: str = "simulate_dune"
    simulate_dune_reason_col: str = "simulate_dune_reason"
    non_erodible_distance_col: str = "dist_lnero"
    simulate_dune_reference_col: str = "d_dune_seaward_polygon"
    simulate_dune_tolerance_m: float = 0.0
    simulate_if_non_erodible_missing: bool = True
    require_dune_polygon_crossing: bool = True

    # Geometry/model controls copied from the Cantabria workflow
    min_heel_crest_gap_m: float = 2.0
    refine_crest_from_profile: bool = True
    crest_refine_window_m: float = 50.0
    crest_refine_min_gain_m: float = 0.02
    min_toe_crest_gap_m: float = 2.0
    min_crest_toe_relief_m: float = 0.20

    use_input_beach_slope: bool = False
    beach_slope_col: str = "mean_beach_slope"
    beach_slope_min: float = 0.01
    beach_slope_max: float = 0.12
    beach_slope_seaward_window_m: float = 50.0
    enforce_model_slope_separation: bool = True
    min_dune_to_beach_slope_ratio: float = 1.15
    min_dune_to_beach_slope_gap: float = 0.005
    max_model_dune_face_slope: float = 1.5
    min_landward_back_slope_length_m: float = 3.0

    blend_width_m: float = 10.0
    max_vertical_change_m: float = np.inf
    max_model_retreat_m: float = 120.0
    max_crest_lowering_m: float = 5.0

    # Output behaviour
    copy_original_profile_when_skipped: bool = True
    keep_traceback_on_failure: bool = False


@dataclass
class DuneParamsProfileOutput:
    """Result for one profile model run."""

    d: np.ndarray
    z: np.ndarray
    volume_eroded_to_beach_m2: float
    status: str
    message: str = ""
    diagnostics: dict[str, Any] | None = None


def _numeric(row: Mapping[str, Any], col: str | None, default: float = np.nan) -> float:
    if col is None:
        return float(default)
    try:
        return float(pd.to_numeric(row.get(col, default), errors="coerce"))
    except Exception:
        return float(default)


def _is_true(value: Any) -> bool:
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


def _as_array(values: Any, dtype=float) -> np.ndarray:
    return np.asarray(values, dtype=dtype).copy()


def _row_numeric_array(row: Mapping[str, Any], col: str) -> np.ndarray:
    if col not in row:
        return np.asarray([], dtype=float)
    return coerce_numeric_1d(row[col])


def _interp_profile_z(row: Mapping[str, Any], d_target: float, cfg: DuneParamsModelConfig) -> float:
    if not np.isfinite(d_target):
        return np.nan
    try:
        d, z, _ = get_profile_arrays(
            row,
            d_col=cfg.d_col,
            z_col=cfg.z_col,
            z_candidates=cfg.z_candidates,
        )
    except Exception:
        return np.nan
    if d.size < 2 or d_target < np.nanmin(d) or d_target > np.nanmax(d):
        return np.nan
    return float(np.interp(float(d_target), d, z))


def _profile_xy_control_points(row: Mapping[str, Any], cfg: DuneParamsModelConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distance/x/y control points for a profile.

    Supports both full X/Y arrays and the common two-endpoint representation.
    """
    if cfg.x_col not in row or cfg.y_col not in row or cfg.d_col not in row:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty

    d0 = _row_numeric_array(row, cfg.d_col)
    x0 = _row_numeric_array(row, cfg.x_col)
    y0 = _row_numeric_array(row, cfg.y_col)

    d0 = d0[np.isfinite(d0)]
    x0 = x0[np.isfinite(x0)]
    y0 = y0[np.isfinite(y0)]

    if len(d0) < 2 or len(x0) < 2 or len(y0) < 2:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty

    if len(x0) == len(d0) and len(y0) == len(d0):
        d_ctrl, x_ctrl, y_ctrl = d0, x0, y0
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
        empty = np.asarray([], dtype=float)
        return empty, empty, empty

    order = np.argsort(d_ctrl)
    d_ctrl, x_ctrl, y_ctrl = d_ctrl[order], x_ctrl[order], y_ctrl[order]
    d_unique, idx = np.unique(d_ctrl, return_index=True)
    return d_unique, x_ctrl[idx], y_ctrl[idx]


def _get_aligned_xy_for_profile(row: Mapping[str, Any], d_ref: np.ndarray, cfg: DuneParamsModelConfig) -> tuple[np.ndarray, np.ndarray]:
    d_ctrl, x_ctrl, y_ctrl = _profile_xy_control_points(row, cfg)
    if len(d_ctrl) < 2:
        return np.full_like(d_ref, np.nan, dtype=float), np.full_like(d_ref, np.nan, dtype=float)
    x = np.full_like(d_ref, np.nan, dtype=float)
    y = np.full_like(d_ref, np.nan, dtype=float)
    valid = (d_ref >= d_ctrl.min()) & (d_ref <= d_ctrl.max())
    x[valid] = np.interp(d_ref[valid], d_ctrl, x_ctrl)
    y[valid] = np.interp(d_ref[valid], d_ctrl, y_ctrl)
    return x, y


def _has_finite_polygon_interval(row: Mapping[str, Any], cfg: DuneParamsModelConfig) -> bool:
    d_land = _numeric(row, cfg.polygon_landward_col)
    d_sea = _numeric(row, cfg.polygon_seaward_col)
    return bool(np.isfinite(d_land) and np.isfinite(d_sea) and d_sea > d_land)


def _simulation_gate(row: Mapping[str, Any], cfg: DuneParamsModelConfig) -> tuple[bool, str]:
    """Return whether this row should enter the erosion model and why."""
    if cfg.simulate_dune_col in row:
        if _is_true(row.get(cfg.simulate_dune_col)):
            return True, str(row.get(cfg.simulate_dune_reason_col, "simulate_dune"))
        return False, str(row.get(cfg.simulate_dune_reason_col, "simulate_dune is False"))

    if cfg.is_dune_col in row and not _is_true(row.get(cfg.is_dune_col)):
        return False, "is_dune is False"

    if cfg.require_dune_polygon_crossing and not _has_finite_polygon_interval(row, cfg):
        return False, "profile does not cross a finite dune polygon interval"

    ref = _numeric(row, cfg.simulate_dune_reference_col)
    lnero = _numeric(row, cfg.non_erodible_distance_col)

    if not np.isfinite(ref):
        return False, f"missing {cfg.simulate_dune_reference_col}"

    if not np.isfinite(lnero):
        if cfg.simulate_if_non_erodible_missing:
            return True, f"simulate_missing_{cfg.non_erodible_distance_col}"
        return False, f"missing {cfg.non_erodible_distance_col}"

    if ref >= lnero - float(cfg.simulate_dune_tolerance_m):
        return True, "simulate_dune"

    return False, "dune_behind_non_erodible_line"


def _choose_toe_col(row: Mapping[str, Any], cfg: DuneParamsModelConfig) -> str:
    if np.isfinite(_numeric(row, cfg.smoothed_toe_col)):
        return cfg.smoothed_toe_col
    return cfg.toe_col


def _choose_crest_col(row: Mapping[str, Any], cfg: DuneParamsModelConfig) -> str:
    if np.isfinite(_numeric(row, cfg.smoothed_crest_col)):
        return cfg.smoothed_crest_col
    return cfg.crest_col


def _choose_landward_col(row: Mapping[str, Any], cfg: DuneParamsModelConfig) -> str:
    crest_col = _choose_crest_col(row, cfg)
    d_crest = _numeric(row, crest_col)
    for col in (cfg.smoothed_heel_col, cfg.heel_col):
        d_land = _numeric(row, col)
        if np.isfinite(d_land) and np.isfinite(d_crest) and d_land < d_crest - float(cfg.min_heel_crest_gap_m):
            return col
    return cfg.polygon_landward_col


def _prepare_row_for_model_geometry(
    row: Mapping[str, Any],
    toe_col: str,
    crest_col: str,
    landward_col: str,
    cfg: DuneParamsModelConfig,
) -> tuple[pd.Series, dict[str, Any]]:
    """Prepare model-control columns without changing stored diagnostics."""
    model_row = row.copy() if hasattr(row, "copy") else pd.Series(dict(row))

    if cfg.profile_id_col in model_row and "profile_idx" not in model_row:
        model_row["profile_idx"] = model_row[cfg.profile_id_col]

    toe_d = _numeric(model_row, toe_col)
    toe_z = _interp_profile_z(model_row, toe_d, cfg)
    if not np.isfinite(toe_z):
        toe_z = _numeric(model_row, cfg.z_toe_col)

    crest_d = _numeric(model_row, crest_col)
    crest_z = _interp_profile_z(model_row, crest_d, cfg)
    if not np.isfinite(crest_z):
        crest_z = _numeric(model_row, cfg.z_crest_col)

    landward_d = _numeric(model_row, landward_col)
    landward_z = _interp_profile_z(model_row, landward_d, cfg)

    model_row[toe_col] = toe_d
    model_row[cfg.z_toe_col] = toe_z
    model_row[crest_col] = crest_d
    model_row[cfg.z_crest_col] = crest_z
    model_row["d_model_landward_boundary"] = landward_d

    toe_source = "smooth_toe" if toe_col == cfg.smoothed_toe_col else "detected_toe"
    crest_source = "smooth_crest" if crest_col == cfg.smoothed_crest_col else "detected_crest"
    if landward_col == cfg.smoothed_heel_col:
        land_source = "smooth_heel"
    elif landward_col == cfg.heel_col:
        land_source = "detected_heel"
    else:
        land_source = landward_col

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


def _make_default_base_params(base_params: DuneToeStormParams | None) -> DuneToeStormParams:
    if base_params is not None:
        return base_params
    # Geometry values are overwritten profile by profile by build_params_for_profile().
    return DuneToeStormParams(
        Ds=5.0,
        z0_init=3.0,
        tan_beta_f=0.05,
        Cs=1.8e-3,
        A_overwash=3.0,
        crest_erosion=True,
        k_crest=0.7,
        crest_width_m=10.0,
        use_profile_mesh=False,
    )


def _as_1d_or_full(value: Sequence[float] | float | None, n: int, *, name: str, default: float | None = None) -> np.ndarray | None:
    if value is None:
        if default is None:
            return None
        return np.full(n, float(default), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    arr = arr.ravel()
    if arr.size != n:
        raise ValueError(f"{name} must have length {n}, got {arr.size}.")
    return arr


def _prepare_forcing(
    *,
    time_s: Sequence[float] | None,
    T: Sequence[float] | float | None,
    TWL: Sequence[float] | None,
    Hs0: Sequence[float] | float | None,
    Ru: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Coerce forcing to 1D arrays accepted by the model."""
    if TWL is not None:
        n = np.asarray(TWL, dtype=float).size
    elif Ru is not None:
        n = np.asarray(Ru, dtype=float).size
    elif Hs0 is not None:
        n = np.asarray(Hs0, dtype=float).size
    else:
        raise ValueError("Provide at least TWL, Ru or Hs0/H0 forcing.")

    if n < 2:
        raise ValueError("Forcing time series must contain at least two samples.")

    if time_s is None:
        tt = np.arange(n, dtype=float) * 3600.0
    else:
        tt = np.asarray(time_s, dtype=float).ravel()
        if tt.size != n:
            raise ValueError(f"time_s must have length {n}, got {tt.size}.")

    T_arr = _as_1d_or_full(T, n, name="T", default=11.0)
    TWL_arr = _as_1d_or_full(TWL, n, name="TWL")
    Hs0_arr = _as_1d_or_full(Hs0, n, name="Hs0")
    Ru_arr = _as_1d_or_full(Ru, n, name="Ru")

    if not np.all(np.isfinite(tt)) or not np.all(np.diff(tt) >= 0.0):
        raise ValueError("time_s must be finite and non-decreasing.")
    if not np.all(np.isfinite(T_arr)):
        raise ValueError("T contains non-finite values.")
    for name, arr in (("TWL", TWL_arr), ("Hs0", Hs0_arr), ("Ru", Ru_arr)):
        if arr is not None and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")

    return tt, T_arr, TWL_arr, Hs0_arr, Ru_arr


def _integrate_flux(time_s: Sequence[float], q: Sequence[float] | None) -> float:
    if q is None:
        return np.nan
    tt = np.asarray(time_s, dtype=float).ravel()
    qq = np.asarray(q, dtype=float).ravel()
    n = min(tt.size, qq.size)
    if n < 2:
        return 0.0
    tt = tt[:n]
    qq = np.nan_to_num(qq[:n], nan=0.0, posinf=0.0, neginf=0.0)
    return float(max(0.0, np.trapz(qq, tt)))


def _original_profile_output(
    row: Mapping[str, Any],
    status: str,
    message: str,
    cfg: DuneParamsModelConfig,
) -> DuneParamsProfileOutput:
    try:
        d_real, z_real, _ = get_profile_arrays(
            row,
            d_col=cfg.d_col,
            z_col=cfg.z_col,
            z_candidates=cfg.z_candidates,
        )
    except Exception:
        d_real = np.asarray([], dtype=float)
        z_real = np.asarray([], dtype=float)
    return DuneParamsProfileOutput(
        d=d_real,
        z=z_real,
        volume_eroded_to_beach_m2=0.0,
        status=status,
        message=message,
        diagnostics={"dune_model_ran": False},
    )


def simulate_dune_profile_from_dune_params_row(
    row: Mapping[str, Any],
    *,
    time_s: Sequence[float] | None = None,
    TWL: Sequence[float] | None = None,
    Hs0: Sequence[float] | float | None = None,
    T: Sequence[float] | float | None = 11.0,
    Ru: Sequence[float] | None = None,
    base_params: DuneToeStormParams | None = None,
    config: DuneParamsModelConfig | None = None,
    runup_mode: str = "stockdon",
    raise_on_failure: bool = False,
) -> DuneParamsProfileOutput:
    """Model one already-detected dune profile from a ``dune_params`` row.

    Parameters
    ----------
    row
        One row from the ``dune_params`` output parquet. It must already contain
        the profile arrays and detected dune parameters.
    time_s, TWL, Hs0, T
        Storm forcing. ``T`` can be scalar or a time series. If ``time_s`` is not
        given, hourly spacing is assumed. ``Hs0`` is passed to the model as the
        offshore/deep-water wave-height series used by ``Cs_mode='larson2004_eq37'``
        and by wave/runup modes; the dune crest level ``Ds`` itself is taken from
        the detected/crest geometry.
    base_params
        Event model parameters. Geometry-dependent values ``Ds``, ``z0_init``
        and ``tan_beta_f`` are overwritten profile by profile.
    config
        Column names and workflow settings.
    runup_mode
        Currently only used if no ``TWL``/``Ru`` is given and the model is forced
        from ``Hs0`` directly.

    Returns
    -------
    DuneParamsProfileOutput
        Contains the complete final profile arrays ``d`` and ``z`` plus
        ``volume_eroded_to_beach_m2``. The volume is the time integral of the
        modelled seaward sediment flux ``qS`` [m² per metre alongshore].
    """
    cfg = config or DuneParamsModelConfig()
    params0 = _make_default_base_params(base_params)

    try:
        gate, gate_reason = _simulation_gate(row, cfg)
        if not gate:
            if cfg.copy_original_profile_when_skipped:
                return _original_profile_output(row, "skipped", gate_reason, cfg)
            return DuneParamsProfileOutput(
                d=np.asarray([], dtype=float),
                z=np.asarray([], dtype=float),
                volume_eroded_to_beach_m2=0.0,
                status="skipped",
                message=gate_reason,
                diagnostics={"dune_model_ran": False},
            )

        toe_col = _choose_toe_col(row, cfg)
        crest_col = _choose_crest_col(row, cfg)
        landward_col = _choose_landward_col(row, cfg)

        for col, label in ((toe_col, "toe"), (crest_col, "crest"), (landward_col, "landward boundary")):
            if not np.isfinite(_numeric(row, col)):
                msg = f"Missing {label} column value: {col}"
                return _original_profile_output(row, "skipped", msg, cfg)

        model_row, geometry_source = _prepare_row_for_model_geometry(row, toe_col, crest_col, landward_col, cfg)
        tt, T_arr, TWL_arr, Hs0_arr, Ru_arr = _prepare_forcing(time_s=time_s, T=T, TWL=TWL, Hs0=Hs0, Ru=Ru)

        geom = build_translation_geometry(
            model_row,
            model_row,
            d_col=cfg.d_col,
            z_col=cfg.z_col,
            z_candidates=cfg.z_candidates,
            toe_col=toe_col,
            crest_col=crest_col,
            berm_col=cfg.berm_col,
            z_toe_col=cfg.z_toe_col,
            z_crest_col=cfg.z_crest_col,
            dune_landward_col="d_model_landward_boundary",
            beach_slope_col=cfg.beach_slope_col if cfg.use_input_beach_slope else None,
            beach_slope_min=cfg.beach_slope_min,
            beach_slope_max=cfg.beach_slope_max,
            seaward_window_m=cfg.beach_slope_seaward_window_m,
            enforce_model_slope_separation=cfg.enforce_model_slope_separation,
            min_dune_to_beach_slope_ratio=cfg.min_dune_to_beach_slope_ratio,
            min_dune_to_beach_slope_gap=cfg.min_dune_to_beach_slope_gap,
            max_model_dune_face_slope=cfg.max_model_dune_face_slope,
            min_landward_back_slope_length_m=cfg.min_landward_back_slope_length_m,
            refine_crest=cfg.refine_crest_from_profile,
            crest_refine_window_m=cfg.crest_refine_window_m,
            crest_refine_min_gain_m=cfg.crest_refine_min_gain_m,
            min_toe_crest_gap_m=cfg.min_toe_crest_gap_m,
            min_crest_toe_relief_m=cfg.min_crest_toe_relief_m,
        )

        model, params, result = simulate_profile_event(
            geom,
            time_s=tt,
            T=T_arr,
            TWL=TWL_arr,
            Ru=Ru_arr,
            H0=Hs0_arr,
            base_params=params0,
            runup_mode=runup_mode,
        )

        proj = merge_modeled_profile_into_real(
            row,
            geom,
            model,
            result,
            t_idx=-1,
            d_col=cfg.d_col,
            z_col=cfg.z_col,
            blend_width_m=cfg.blend_width_m,
            dune_landward_col="d_model_landward_boundary",
            max_vertical_change_m=cfg.max_vertical_change_m,
            max_retreat_m=cfg.max_model_retreat_m,
            max_crest_lowering_m=cfg.max_crest_lowering_m,
        )

        d_out = _as_array(proj["d_real"])
        z_out = _as_array(proj["z_merged"])
        V_to_beach = _integrate_flux(result.get("time_s", tt), result.get("qS"))
        V_total = _integrate_flux(result.get("time_s", tt), result.get("qD"))
        V_landward = _integrate_flux(result.get("time_s", tt), result.get("qL"))

        features = translate_modeled_features_to_real(geom, result, t_idx=-1)
        summary = summarize_simulation(geom, params, result, t_idx=-1)
        x_out, y_out = _get_aligned_xy_for_profile(row, d_out, cfg)

        diagnostics: dict[str, Any] = {
            "dune_model_ran": True,
            "simulation_gate_reason": gate_reason,
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
            "dune_merge_landward_limit_d": proj.get("landward_limit_d", np.nan),
            "dune_merge_initial_landward_limit_d": proj.get("initial_landward_limit_d", np.nan),
            "dune_merge_final_landward_limit_d": proj.get("final_landward_limit_d", np.nan),
            "dune_merge_seaward_limit_d": proj.get("seaward_limit_d", np.nan),
            "d_toe_shift_m": proj.get("d_toe_shift_m", np.nan),
            "d_crest_shift_m": proj.get("d_crest_shift_m", np.nan),
            "z_crest_change_m": proj.get("z_crest_change_m", np.nan),
            "d_landward_polygon_model": geom.d_landward0,
            "z_landward_polygon_model": geom.z_landward0,
            "dune_merge_strategy": str(proj.get("merge_strategy", np.array([""]))[0]),
            "volume_eroded_to_beach_m2": V_to_beach,
            "volume_eroded_total_m2": V_total,
            "volume_overwashed_landward_m2": V_landward,
            "volume_eroded_front_m2": proj.get("volume_eroded_front_m2", np.nan),
            "volume_deposited_landward_m2": proj.get("volume_deposited_landward_m2", np.nan),
            "volume_balance_error_m2": proj.get("volume_balance_error_m2", np.nan),
            "x_dune_eroded": _as_array(x_out),
            "y_dune_eroded": _as_array(y_out),
        }

        return DuneParamsProfileOutput(
            d=d_out,
            z=z_out,
            volume_eroded_to_beach_m2=V_to_beach,
            status="ok",
            message="",
            diagnostics=diagnostics,
        )

    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        if raise_on_failure:
            raise
        out = _original_profile_output(row, "failed", msg, cfg)
        if cfg.keep_traceback_on_failure:
            out.diagnostics = dict(out.diagnostics or {})
            out.diagnostics["traceback"] = traceback.format_exc(limit=5)
        return out


def _read_dune_params_table(parquet_path: str | Path):
    path = Path(parquet_path)
    if gpd is not None:
        try:
            return gpd.read_parquet(path)
        except Exception:
            pass
    return pd.read_parquet(path)


def _select_profile_row(df: pd.DataFrame, *, profile_id: Any = None, row_index: int | None = None, cfg: DuneParamsModelConfig):
    if profile_id is not None:
        if cfg.profile_id_col not in df.columns:
            raise KeyError(f"profile_id was given, but column {cfg.profile_id_col!r} is not in the parquet.")
        sel = df[df[cfg.profile_id_col] == profile_id]
        if len(sel) != 1:
            raise ValueError(f"Expected exactly one profile with {cfg.profile_id_col}={profile_id!r}, found {len(sel)}.")
        return sel.iloc[0]

    if row_index is not None:
        return df.iloc[int(row_index)]

    if len(df) == 1:
        return df.iloc[0]

    raise ValueError("The parquet contains more than one profile. Provide profile_id=... or row_index=....")


def simulate_dune_profile_from_dune_params_parquet(
    parquet_path: str | Path,
    *,
    profile_id: Any = None,
    row_index: int | None = None,
    time_s: Sequence[float] | None = None,
    TWL: Sequence[float] | None = None,
    Hs0: Sequence[float] | float | None = None,
    T: Sequence[float] | float | None = 11.0,
    Ru: Sequence[float] | None = None,
    base_params: DuneToeStormParams | None = None,
    config: DuneParamsModelConfig | None = None,
    runup_mode: str = "stockdon",
    return_diagnostics: bool = False,
    raise_on_failure: bool = False,
):
    """Run one profile from a ``dune_params`` parquet and return final vectors:
        d_final, z_final, volume_eroded_to_beach_m2
    """
    cfg = config or DuneParamsModelConfig()
    df = _read_dune_params_table(parquet_path)
    row = _select_profile_row(df, profile_id=profile_id, row_index=row_index, cfg=cfg)
    out = simulate_dune_profile_from_dune_params_row(
        row,
        time_s=time_s,
        TWL=TWL,
        Hs0=Hs0,
        T=T,
        Ru=Ru,
        base_params=base_params,
        config=cfg,
        runup_mode=runup_mode,
        raise_on_failure=raise_on_failure,
    )
    if return_diagnostics:
        return out.d, out.z, out.volume_eroded_to_beach_m2, {
            "status": out.status,
            "message": out.message,
            **(out.diagnostics or {}),
        }
    return out.d, out.z, out.volume_eroded_to_beach_m2


def _record_from_output(output: DuneParamsProfileOutput, cfg: DuneParamsModelConfig) -> dict[str, Any]:
    diag = dict(output.diagnostics or {})
    rec = {
        "dune_model_status": output.status,
        "dune_model_message": output.message,
        "dune_model_ran": bool(diag.pop("dune_model_ran", output.status == "ok")),
        "d_dune_eroded": output.d,
        "z_dune_eroded": output.z,
        "volume_eroded_to_beach_m2": output.volume_eroded_to_beach_m2,
    }
    rec.update(diag)
    return rec


def run_dune_model_from_dune_params_parquet(
    parquet_path: str | Path,
    *,
    time_s: Sequence[float] | None = None,
    TWL: Sequence[float] | None = None,
    Hs0: Sequence[float] | float | None = None,
    T: Sequence[float] | float | None = 11.0,
    Ru: Sequence[float] | None = None,
    base_params: DuneToeStormParams | None = None,
    config: DuneParamsModelConfig | None = None,
    runup_mode: str = "stockdon",
    output_path: str | Path | None = None,
    show_progress: bool = True,
):
    """Run the dune model for every row in a ``dune_params`` parquet.

    The output is the input table plus model columns, including the full final
    profile arrays ``d_dune_eroded`` and ``z_dune_eroded`` and the seaward volume
    ``volume_eroded_to_beach_m2``.
    """
    cfg = config or DuneParamsModelConfig()
    df = _read_dune_params_table(parquet_path)

    iterator = df.iterrows()
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=len(df), desc="Dune model from dune_params", unit="profile", dynamic_ncols=True)
        except Exception:  # pragma: no cover
            pass

    records: list[dict[str, Any]] = []
    for _, row in iterator:
        output = simulate_dune_profile_from_dune_params_row(
            row,
            time_s=time_s,
            TWL=TWL,
            Hs0=Hs0,
            T=T,
            Ru=Ru,
            base_params=base_params,
            config=cfg,
            runup_mode=runup_mode,
            raise_on_failure=False,
        )
        records.append(_record_from_output(output, cfg))

    model_df = pd.DataFrame(records, index=df.index)
    out = df.drop(columns=[c for c in model_df.columns if c in df.columns], errors="ignore")
    out = pd.concat([out, model_df], axis=1)

    if gpd is not None and hasattr(df, "geometry"):
        out = gpd.GeoDataFrame(out, geometry=df.geometry.name, crs=getattr(df, "crs", None))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            if gpd is not None and isinstance(out, gpd.GeoDataFrame):
                out.to_parquet(output_path)
            else:
                out.to_parquet(output_path)
        elif suffix in {".pkl", ".pickle"}:
            out.to_pickle(output_path)
        elif suffix == ".csv":
            # CSV cannot represent profile arrays cleanly, but is useful for diagnostics.
            out.to_csv(output_path, index=False)
        else:
            raise ValueError("output_path must end in .parquet, .pkl/.pickle or .csv")

    return out


__all__ = [
    "DuneParamsModelConfig",
    "DuneParamsProfileOutput",
    "simulate_dune_profile_from_dune_params_row",
    "simulate_dune_profile_from_dune_params_parquet",
    "run_dune_model_from_dune_params_parquet",
]
