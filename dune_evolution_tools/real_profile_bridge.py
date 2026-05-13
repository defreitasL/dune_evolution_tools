from __future__ import annotations

"""Bridge between observed profiles and the event-scale dune erosion model.

The workflow in this file is intentionally narrow.  A real profile is converted
into one calculation profile, the storm model is run on that profile, and the
modelled change is transferred back to the original profile mesh.

Coordinate convention
---------------------
The Berria profiles use ``d`` increasing seaward.  The dune model uses a local
axis ``x`` increasing landward.  The bridge therefore uses::

    x = d_toe - d

so the initial frontal dune toe is located at ``x = 0`` and landward points have
positive ``x``.

Calculation geometry
--------------------
The calculation profile is a volume-matched trapezoid:

* frontal toe: smoothed/detected toe used by the workflow;
* crest: detected/refined crest from the real profile;
* landward boundary: mapped dune-polygon landward intersection;
* crest-platform width: solved so the volume above z=0 matches the real profile
  volume inside the numerical dune domain as closely as possible.

If the real volume cannot be matched with those fixed toe/crest/landward points,
the closest admissible trapezoid is used and the residual is written to the
output diagnostics.  This is deliberate: the code should report an incompatible
geometry instead of hiding it by moving physical control points.
"""

from dataclasses import asdict, dataclass, fields, replace
import math
import re
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dune_evolution_tools.model import DuneToeStormModel
from dune_evolution_tools.params import DuneToeStormParams


DEFAULT_Z_CANDIDATES = ("z", "z_full", "z_chill", "z_corregido")
NP_FLOAT64_RE = re.compile(r"np\.float64\(([-+0-9.eE]+)\)")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class TranslationGeometry:
    """Profile-specific geometry used by the dune model and real-profile merge."""

    profile_idx: int | None
    d_toe0: float
    z_toe0: float
    d_crest0: float
    z_crest0: float
    d_landward0: float
    z_landward0: float
    d_berm0: float = np.nan
    tan_beta_f: float = np.nan
    tan_beta_f_observed: float = np.nan
    tan_beta_D: float = np.nan
    tan_beta_eff_est: float = np.nan
    alpha_rep_deg: float = np.nan
    x_crest_obs: float = np.nan
    x_landward_obs: float = np.nan
    landward_crest_width_m: float = 0.0
    landward_back_slope_m: float = 0.0
    landward_back_buffer_m: float = 0.0
    tan_beta_back: float = np.nan
    z_back: float = np.nan
    real_volume_above_0_m2: float = np.nan
    calc_volume_above_0_m2: float = np.nan
    calc_volume_residual_m2: float = np.nan
    calc_volume_match_status: str = "unknown"
    z_col: str = "z"
    crest_source: str = "detected_crest"
    beach_slope_source: str = "profile_estimate"


# -----------------------------------------------------------------------------
# Generic profile parsing
# -----------------------------------------------------------------------------


def _get_first_numeric(*items: tuple[Mapping[str, Any] | None, str | None]) -> float:
    """Return the first finite number found in a sequence of mapping/column pairs."""
    for mapping, key in items:
        if mapping is None or key is None:
            continue
        try:
            value = mapping.get(key)
        except AttributeError:
            continue
        value = float(pd.to_numeric(value, errors="coerce"))
        if np.isfinite(value):
            return value
    return np.nan


def coerce_numeric_1d(obj: Any) -> np.ndarray:
    """Convert arrays, lists or stringified arrays to a one-dimensional float array."""
    if obj is None:
        return np.asarray([], dtype=float)

    if isinstance(obj, np.ndarray):
        return np.asarray(obj, dtype=float).ravel()

    if isinstance(obj, (list, tuple, pd.Series)):
        return np.asarray(obj, dtype=float).ravel()

    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return np.asarray([], dtype=float)

        matches = NP_FLOAT64_RE.findall(s)
        if matches:
            return np.asarray([float(v) for v in matches], dtype=float)

        s = s.replace("np.float32(", "").replace("np.float64(", "").replace(")", "")
        vals = FLOAT_RE.findall(s)
        return np.asarray([float(v) for v in vals], dtype=float)

    return np.asarray(obj, dtype=float).ravel()


def infer_z_column(
    row: Mapping[str, Any],
    z_col: str | None = None,
    z_candidates: Sequence[str] = DEFAULT_Z_CANDIDATES,
) -> str:
    if z_col is not None and z_col in row:
        return z_col

    for cand in z_candidates:
        if cand in row:
            return cand

    raise KeyError(f"Could not infer elevation column. Checked: {list(z_candidates)!r}.")


def get_profile_arrays(
    row: Mapping[str, Any],
    *,
    d_col: str = "d",
    z_col: str | None = None,
    z_candidates: Sequence[str] = DEFAULT_Z_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return clean ``d`` and ``z`` arrays sorted by increasing ``d``."""
    if d_col not in row:
        raise KeyError(f"Column {d_col!r} not found in profile row.")

    z_name = infer_z_column(row, z_col=z_col, z_candidates=z_candidates)
    d = coerce_numeric_1d(row[d_col])
    z = coerce_numeric_1d(row[z_name])

    n = min(len(d), len(z))
    d = d[:n]
    z = z[:n]

    mask = np.isfinite(d) & np.isfinite(z)
    d = d[mask]
    z = z[mask]

    if d.size < 2:
        raise ValueError("Profile contains fewer than two valid points.")

    order = np.argsort(d)
    d = d[order]
    z = z[order]
    d_unique, idx = np.unique(d, return_index=True)
    d = d_unique
    z = z[idx]

    if d.size < 2:
        raise ValueError("Profile is degenerate after cleaning/sorting.")

    return d, z, z_name


def interp_on_profile(d: np.ndarray, z: np.ndarray, d_target: float) -> float:
    """Interpolate elevation at ``d_target`` inside the observed profile extent."""
    if not np.isfinite(d_target) or d_target < np.nanmin(d) or d_target > np.nanmax(d):
        return np.nan
    return float(np.interp(float(d_target), d, z))


def real_d_to_local_x(d_real: np.ndarray | float, d_toe0: float) -> np.ndarray | float:
    return float(d_toe0) - np.asarray(d_real)


def local_x_to_real_d(x_local: np.ndarray | float, d_toe0: float) -> np.ndarray | float:
    return float(d_toe0) - np.asarray(x_local)


# -----------------------------------------------------------------------------
# Geometry and volume matching
# -----------------------------------------------------------------------------


def _trapz_area(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    xx = np.asarray(x[mask], dtype=float)
    yy = np.asarray(y[mask], dtype=float)
    order = np.argsort(xx)
    return float(np.trapz(yy[order], xx[order]))


def _volume_above_level(d: np.ndarray, z: np.ndarray, *, level: float = 0.0) -> float:
    """Area per metre above a reference level on a 1D profile."""
    return _trapz_area(d, np.maximum(np.asarray(z, dtype=float) - float(level), 0.0))


def _extract_domain_profile(
    d: np.ndarray,
    z: np.ndarray,
    *,
    d_landward: float,
    d_toe: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Profile subset between the landward polygon limit and the frontal toe."""
    if not (np.isfinite(d_landward) and np.isfinite(d_toe)) or d_landward >= d_toe:
        raise ValueError("Invalid dune numerical domain: expected d_landward < d_toe.")

    lo = max(float(d_landward), float(np.nanmin(d)))
    hi = min(float(d_toe), float(np.nanmax(d)))
    if lo >= hi:
        raise ValueError("The dune numerical domain does not overlap the observed profile.")

    interior = (d > lo) & (d < hi) & np.isfinite(z)
    dd = np.concatenate(([lo], d[interior], [hi]))
    zz = np.concatenate(([np.interp(lo, d, z)], z[interior], [np.interp(hi, d, z)]))
    return dd, zz


def refine_crest_on_profile(
    d: np.ndarray,
    z: np.ndarray,
    *,
    d_crest: float,
    d_toe: float,
    d_landward: float,
    window_m: float = 8.0,
    min_gain_m: float = 0.02,
) -> tuple[float, float]:
    """Snap the crest to the highest nearby observed point when the gain is meaningful."""
    if not np.isfinite(d_crest):
        return np.nan, np.nan

    lo = max(float(d_crest) - float(window_m), float(d_landward))
    hi = min(float(d_crest) + float(window_m), float(d_toe))
    mask = (d >= lo) & (d <= hi) & np.isfinite(z)
    current_z = interp_on_profile(d, z, d_crest)

    if mask.sum() < 3:
        return float(d_crest), float(current_z)

    idx = np.where(mask)[0]
    i_best = idx[int(np.nanargmax(z[mask]))]
    if np.isfinite(current_z) and z[i_best] < current_z + float(min_gain_m):
        return float(d_crest), float(current_z)

    return float(d[i_best]), float(z[i_best])




def choose_crest_inside_domain(
    d: np.ndarray,
    z: np.ndarray,
    *,
    d_crest: float,
    d_toe: float,
    d_landward: float,
    min_toe_crest_gap_m: float = 2.0,
    landward_buffer_m: float = 1.0,
) -> tuple[float, float, str]:
    """Return a crest position that is usable by the calculation geometry.

    The detector is normally good, but the calculation mesh is stricter than a
    plotting diagnostic: it needs ``d_landward < d_crest < d_toe`` with a finite
    dune face.  When the detected crest violates that ordering, the safest
    fallback is the highest observed point inside the numerical dune domain.
    """
    d = np.asarray(d, dtype=float)
    z = np.asarray(z, dtype=float)
    source = "detected_crest"

    valid_detected = (
        np.isfinite(d_crest)
        and np.isfinite(d_toe)
        and np.isfinite(d_landward)
        and d_landward < d_crest < d_toe - float(min_toe_crest_gap_m)
    )
    if valid_detected:
        zc = interp_on_profile(d, z, d_crest)
        if np.isfinite(zc):
            return float(d_crest), float(zc), source

    lo = float(d_landward) + max(0.0, float(landward_buffer_m))
    hi = float(d_toe) - max(0.0, float(min_toe_crest_gap_m))
    mask = (d >= lo) & (d <= hi) & np.isfinite(z)
    if mask.sum() < 3:
        raise ValueError("Could not recover a valid crest inside the dune numerical domain.")

    idx = np.where(mask)[0]
    i_best = idx[int(np.nanargmax(z[mask]))]
    return float(d[i_best]), float(z[i_best]), "domain_max_crest"

def estimate_beachface_slope(
    d: np.ndarray,
    z: np.ndarray,
    *,
    d_toe: float,
    d_berm: float | None = None,
    seaward_window_m: float = 55.0,
    seaward_offset_m: float = 2.0,
    min_points: int = 8,
    min_slope: float = 1e-3,
    max_slope: float = 0.35,
) -> float:
    """Robustly estimate the beachface slope used by the storm model.

    The retreat formulation is very sensitive to ``tan_beta_f``.  A single
    noisy point near the dune toe or a berm point far down the profile can move
    the result a lot, so the estimate is deliberately local and robust:

    * start a few metres seaward of the dune toe, avoiding the toe kink itself;
    * use a fixed seaward window, clipped by the berm only when it falls inside
      that window;
    * smooth the selected elevations with a short rolling median;
    * combine a robust line fit with the median local gradient.

    The returned value is a positive slope magnitude.
    """
    d = np.asarray(d, dtype=float)
    z = np.asarray(z, dtype=float)
    if not np.isfinite(d_toe):
        raise ValueError("Cannot estimate beach slope without a finite toe position.")

    left = float(d_toe) + max(0.0, float(seaward_offset_m))
    right = float(d_toe) + max(float(seaward_window_m), float(seaward_offset_m) + 1.0)
    if d_berm is not None and np.isfinite(d_berm) and d_berm > left:
        right = min(right, float(d_berm))

    mask = (d >= left) & (d <= right) & np.isfinite(z)
    if mask.sum() < max(3, min_points):
        # Fall back to the first finite samples seaward of the toe.  This keeps
        # short profiles usable while still avoiding landward dune-face points.
        i0 = int(np.searchsorted(d, left))
        i1 = min(len(d), max(i0 + min_points, int(np.searchsorted(d, float(d_toe) + seaward_window_m))))
        mask = np.zeros_like(d, dtype=bool)
        mask[i0:i1] = np.isfinite(z[i0:i1])

    if mask.sum() < 3:
        raise ValueError("Not enough seaward profile points to estimate beachface slope.")

    dd = d[mask].astype(float)
    zz = z[mask].astype(float)
    order = np.argsort(dd)
    dd = dd[order]
    zz = zz[order]

    if dd.size >= 7:
        zz_s = pd.Series(zz).rolling(5, center=True, min_periods=1).median().to_numpy(dtype=float)
    else:
        zz_s = zz

    # Trim strong elevation outliers before the line fit.  The percentiles are
    # intentionally loose; the goal is not to remove morphology, only spikes.
    qlo, qhi = np.nanpercentile(zz_s, [5, 95])
    fit_mask = np.isfinite(zz_s) & (zz_s >= qlo) & (zz_s <= qhi)
    if fit_mask.sum() < 3:
        fit_mask = np.isfinite(zz_s)

    slope_fit = float(np.polyfit(dd[fit_mask], zz_s[fit_mask], 1)[0])

    grad = np.gradient(zz_s, dd)
    grad = grad[np.isfinite(grad)]
    slope_grad = float(np.nanmedian(grad)) if grad.size else slope_fit

    # The profile coordinate increases seaward, so a regular beachface often has
    # a negative dz/dd.  The model needs the magnitude.
    candidates = np.asarray([abs(slope_fit), abs(slope_grad)], dtype=float)
    slope = float(np.nanmedian(candidates[np.isfinite(candidates)]))
    return float(np.clip(slope, min_slope, max_slope))


def effective_dune_slope(tan_beta_f: float, tan_beta_D: float) -> float:
    """Larson effective slope, returned as inf when the expression is singular."""
    tan_beta_f = float(tan_beta_f)
    tan_beta_D = float(tan_beta_D)
    if tan_beta_f <= 0.0 or tan_beta_D <= tan_beta_f:
        return np.inf
    return float(1.0 / (1.0 / tan_beta_f - 1.0 / tan_beta_D))


def _calculation_profile_local(
    *,
    z_toe: float,
    z_crest: float,
    z_landward: float,
    x_crest: float,
    x_landward: float,
    crest_width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the local trapezoid points used as model initial geometry."""
    W = float(np.clip(crest_width_m, 0.0, max(0.0, x_landward - x_crest)))
    x_plateau_end = min(float(x_landward), float(x_crest + W))

    x = [0.0, float(x_crest)]
    z = [float(z_toe), float(z_crest)]

    if x_plateau_end > x_crest + 1e-9:
        x.append(float(x_plateau_end))
        z.append(float(z_crest))

    if x_landward > x_plateau_end + 1e-9:
        x.append(float(x_landward))
        z.append(float(z_landward))
    elif abs(x[-1] - x_landward) > 1e-9:
        x.append(float(x_landward))
        z.append(float(z_landward))
    else:
        z[-1] = float(z_landward)

    return np.asarray(x, dtype=float), np.asarray(z, dtype=float)


def _calc_trapezoid_volume(
    *,
    z_toe: float,
    z_crest: float,
    z_landward: float,
    x_crest: float,
    x_landward: float,
    crest_width_m: float,
) -> float:
    x, z = _calculation_profile_local(
        z_toe=z_toe,
        z_crest=z_crest,
        z_landward=z_landward,
        x_crest=x_crest,
        x_landward=x_landward,
        crest_width_m=crest_width_m,
    )
    return _volume_above_level(x, z, level=0.0)


def solve_volume_matched_trapezoid(
    *,
    target_volume_m2: float,
    z_toe: float,
    z_crest: float,
    z_landward: float,
    x_crest: float,
    x_landward: float,
    min_back_slope_length_m: float = 2.0,
    n_iter: int = 60,
) -> dict[str, float | str]:
    """Solve the crest-platform width that best matches the observed dune volume.

    Toe, crest and landward boundary are fixed.  The only available degree of
    freedom is the flat crest-platform width, with the remaining landward length
    assigned to the back slope.  If the target volume is outside the reachable
    range, the nearest admissible endpoint is returned.
    """
    if x_crest <= 0.0 or x_landward <= x_crest:
        raise ValueError("Invalid local dune geometry: expected 0 < x_crest < x_landward.")

    W_min = 0.0
    back_len_min = max(0.0, float(min_back_slope_length_m))
    available_back_len = max(0.0, float(x_landward - x_crest))
    W_max = max(0.0, available_back_len - back_len_min)

    V_min = _calc_trapezoid_volume(
        z_toe=z_toe,
        z_crest=z_crest,
        z_landward=z_landward,
        x_crest=x_crest,
        x_landward=x_landward,
        crest_width_m=W_min,
    )
    V_max = _calc_trapezoid_volume(
        z_toe=z_toe,
        z_crest=z_crest,
        z_landward=z_landward,
        x_crest=x_crest,
        x_landward=x_landward,
        crest_width_m=W_max,
    )

    target = float(target_volume_m2)
    loV, hiV = sorted((V_min, V_max))
    if target <= loV:
        W = W_min if V_min <= V_max else W_max
        status = "below_reachable_volume"
    elif target >= hiV:
        W = W_max if V_max >= V_min else W_min
        status = "above_reachable_volume"
    else:
        lo, hi = W_min, W_max
        increasing = V_max >= V_min
        for _ in range(int(n_iter)):
            mid = 0.5 * (lo + hi)
            Vm = _calc_trapezoid_volume(
                z_toe=z_toe,
                z_crest=z_crest,
                z_landward=z_landward,
                x_crest=x_crest,
                x_landward=x_landward,
                crest_width_m=mid,
            )
            if (Vm < target) == increasing:
                lo = mid
            else:
                hi = mid
        W = 0.5 * (lo + hi)
        status = "matched"

    V = _calc_trapezoid_volume(
        z_toe=z_toe,
        z_crest=z_crest,
        z_landward=z_landward,
        x_crest=x_crest,
        x_landward=x_landward,
        crest_width_m=W,
    )
    back_len = max(0.0, float(x_landward - x_crest - W))
    if back_len > 1e-9:
        # Signed slope of the landward segment.  A negative value is valid here:
        # it means the polygon landward limit is higher than the frontal crest.
        # The model core does not use this segment in the ODE; it is needed to
        # reconstruct and merge the calculation profile consistently.
        tan_beta_back = float((z_crest - z_landward) / back_len)
    else:
        tan_beta_back = 0.0

    return {
        "crest_width_m": float(W),
        "back_slope_length_m": float(back_len),
        "tan_beta_back": float(tan_beta_back),
        "calc_volume_m2": float(V),
        "volume_residual_m2": float(V - target),
        "status": status,
    }


def validate_translation_geometry(
    *,
    d_toe: float,
    z_toe: float,
    d_crest: float,
    z_crest: float,
    d_landward: float,
    z_landward: float,
    min_toe_crest_gap_m: float = 2.0,
    min_crest_toe_relief_m: float = 0.20,
) -> None:
    """Fail early for geometries that cannot support the calculation profile."""
    values = (d_toe, z_toe, d_crest, z_crest, d_landward, z_landward)
    if not all(np.isfinite(v) for v in values):
        raise ValueError("Toe, crest and landward boundary must all have finite d/z values.")

    if not d_landward < d_crest < d_toe:
        raise ValueError(
            "Invalid dune-domain ordering. Expected d_landward < d_crest < d_toe "
            "because d increases seaward."
        )

    gap = float(d_toe - d_crest)
    if gap < float(min_toe_crest_gap_m):
        raise ValueError(f"Toe/crest gap is too small for a stable model domain: {gap:g} m.")

    relief = float(z_crest - z_toe)
    if relief < float(min_crest_toe_relief_m):
        raise ValueError(f"Dune relief is too small: crest-to-toe relief is {relief:g} m.")


def build_translation_geometry(
    profile_row: Mapping[str, Any],
    detected_row: Mapping[str, Any],
    *,
    d_col: str = "d",
    z_col: str | None = None,
    z_candidates: Sequence[str] = DEFAULT_Z_CANDIDATES,
    toe_col: str = "d_toe_final_smooth",
    crest_col: str = "d_crest",
    heel_col: str = "d_heel_final",  # kept for API compatibility; not used by the mesh builder
    berm_col: str = "d_berm",
    z_toe_col: str = "z_toe_final",  # kept as fallback only
    z_crest_col: str = "z_crest",    # kept as fallback only
    z_heel_col: str = "z_heel_final",  # kept for API compatibility
    seaward_window_m: float = 40.0,
    default_landward_back_buffer_m: float = 0.0,
    alpha_rep_deg_override: float | None = None,
    crest_width_m: float | None = None,  # ignored; width is solved from volume
    crest_width_col: str | None = None,  # ignored; width is solved from volume
    dune_landward_col: str | None = "d_dune_landward_polygon",
    use_polygon_landward_as_heel: bool = True,  # API compatibility
    default_back_slope: float = 0.10,           # API compatibility fallback
    min_back_slope_m: float = 0.0,              # API compatibility
    beach_slope_col: str | None = None,
    beach_slope_min: float = 0.001,
    beach_slope_max: float = 0.35,
    enforce_model_slope_separation: bool = True,
    min_dune_to_beach_slope_ratio: float = 1.15,
    min_dune_to_beach_slope_gap: float = 0.005,
    max_model_dune_face_slope: float = 1.5,
    min_landward_back_slope_length_m: float = 2.0,
    refine_crest: bool = True,
    crest_refine_window_m: float = 8.0,
    crest_refine_min_gain_m: float = 0.02,
    min_toe_crest_gap_m: float = 2.0,
    min_crest_toe_relief_m: float = 0.20,
) -> TranslationGeometry:
    """Build a volume-matched calculation geometry for one profile."""
    profile_idx = detected_row.get("profile_idx", getattr(profile_row, "name", None))
    d, z, z_name = get_profile_arrays(profile_row, d_col=d_col, z_col=z_col, z_candidates=z_candidates)

    d_toe0 = _get_first_numeric((detected_row, toe_col), (profile_row, toe_col))
    d_crest0 = _get_first_numeric((detected_row, crest_col), (profile_row, crest_col))
    d_landward0 = _get_first_numeric((detected_row, dune_landward_col), (profile_row, dune_landward_col))
    d_berm0 = _get_first_numeric((detected_row, berm_col), (profile_row, berm_col))

    if not np.isfinite(d_toe0):
        raise ValueError(f"Missing model toe column {toe_col!r}.")
    if not np.isfinite(d_crest0):
        raise ValueError(f"Missing crest column {crest_col!r}.")
    if not np.isfinite(d_landward0):
        raise ValueError(f"Missing landward dune boundary column {dune_landward_col!r}.")

    z_toe0 = interp_on_profile(d, z, d_toe0)
    if not np.isfinite(z_toe0):
        z_toe0 = _get_first_numeric((detected_row, z_toe_col), (profile_row, z_toe_col))

    d_crest0, z_crest0, crest_source = choose_crest_inside_domain(
        d,
        z,
        d_crest=d_crest0,
        d_toe=d_toe0,
        d_landward=d_landward0,
        min_toe_crest_gap_m=min_toe_crest_gap_m,
    )
    if not np.isfinite(z_crest0):
        z_crest0 = _get_first_numeric((detected_row, z_crest_col), (profile_row, z_crest_col))

    z_landward0 = interp_on_profile(d, z, d_landward0)

    if refine_crest:
        d_crest_ref, z_crest_ref = refine_crest_on_profile(
            d,
            z,
            d_crest=d_crest0,
            d_toe=d_toe0,
            d_landward=d_landward0,
            window_m=crest_refine_window_m,
            min_gain_m=crest_refine_min_gain_m,
        )
        if np.isfinite(d_crest_ref) and np.isfinite(z_crest_ref):
            if abs(float(d_crest_ref) - float(d_crest0)) > 1.0e-9:
                crest_source = crest_source + "+local_max"
            d_crest0 = d_crest_ref
            z_crest0 = z_crest_ref

    validate_translation_geometry(
        d_toe=d_toe0,
        z_toe=z_toe0,
        d_crest=d_crest0,
        z_crest=z_crest0,
        d_landward=d_landward0,
        z_landward=z_landward0,
        min_toe_crest_gap_m=min_toe_crest_gap_m,
        min_crest_toe_relief_m=min_crest_toe_relief_m,
    )

    d_dom, z_dom = _extract_domain_profile(d, z, d_landward=d_landward0, d_toe=d_toe0)
    real_volume = _volume_above_level(d_dom, z_dom, level=0.0)

    x_crest_obs = float(d_toe0 - d_crest0)
    x_landward_obs = float(d_toe0 - d_landward0)
    tan_beta_D = float((z_crest0 - z_toe0) / x_crest_obs)
    tan_beta_D = float(np.clip(tan_beta_D, 1e-4, max_model_dune_face_slope))

    volume_solution = solve_volume_matched_trapezoid(
        target_volume_m2=real_volume,
        z_toe=z_toe0,
        z_crest=z_crest0,
        z_landward=z_landward0,
        x_crest=x_crest_obs,
        x_landward=x_landward_obs,
        min_back_slope_length_m=min_landward_back_slope_length_m,
    )

    tan_beta_f_observed = _get_first_numeric((detected_row, beach_slope_col), (profile_row, beach_slope_col)) if beach_slope_col else np.nan
    if np.isfinite(tan_beta_f_observed):
        tan_beta_f_observed = float(np.clip(abs(tan_beta_f_observed), beach_slope_min, beach_slope_max))
        beach_slope_source = "input_column"
    else:
        tan_beta_f_observed = estimate_beachface_slope(
            d,
            z,
            d_toe=d_toe0,
            d_berm=d_berm0 if np.isfinite(d_berm0) else None,
            seaward_window_m=seaward_window_m,
            min_slope=beach_slope_min,
            max_slope=beach_slope_max,
        )
        beach_slope_source = "robust_profile_estimate"

    tan_beta_f = float(tan_beta_f_observed)
    if enforce_model_slope_separation:
        max_safe_f = min(
            tan_beta_D / max(float(min_dune_to_beach_slope_ratio), 1.001),
            tan_beta_D - max(float(min_dune_to_beach_slope_gap), 0.0),
        )
        if np.isfinite(max_safe_f) and max_safe_f > beach_slope_min:
            tan_beta_f = min(tan_beta_f, max_safe_f)
        tan_beta_f = float(np.clip(tan_beta_f, beach_slope_min, beach_slope_max))

    tan_beta_eff_est = effective_dune_slope(tan_beta_f, tan_beta_D)
    alpha_rep_deg = float(alpha_rep_deg_override) if alpha_rep_deg_override is not None else float(np.degrees(np.arctan(tan_beta_D)))

    return TranslationGeometry(
        profile_idx=profile_idx,
        d_toe0=float(d_toe0),
        z_toe0=float(z_toe0),
        d_crest0=float(d_crest0),
        z_crest0=float(z_crest0),
        d_landward0=float(d_landward0),
        z_landward0=float(z_landward0),
        d_berm0=float(d_berm0) if np.isfinite(d_berm0) else np.nan,
        tan_beta_f=float(tan_beta_f),
        tan_beta_f_observed=float(tan_beta_f_observed),
        tan_beta_D=float(tan_beta_D),
        tan_beta_eff_est=float(tan_beta_eff_est),
        alpha_rep_deg=float(alpha_rep_deg),
        x_crest_obs=float(x_crest_obs),
        x_landward_obs=float(x_landward_obs),
        landward_crest_width_m=float(volume_solution["crest_width_m"]),
        landward_back_slope_m=float(volume_solution["back_slope_length_m"]),
        landward_back_buffer_m=float(default_landward_back_buffer_m),
        tan_beta_back=float(volume_solution["tan_beta_back"]),
        z_back=float(z_landward0),
        real_volume_above_0_m2=float(real_volume),
        calc_volume_above_0_m2=float(volume_solution["calc_volume_m2"]),
        calc_volume_residual_m2=float(volume_solution["volume_residual_m2"]),
        calc_volume_match_status=str(volume_solution["status"]),
        z_col=z_name,
        crest_source=str(crest_source),
        beach_slope_source=str(beach_slope_source),
    )


# -----------------------------------------------------------------------------
# Model construction and simulation
# -----------------------------------------------------------------------------


def _params_field_names() -> set[str]:
    return {f.name for f in fields(DuneToeStormParams)}


def build_params_for_profile(
    geometry: TranslationGeometry,
    *,
    base_params: DuneToeStormParams | None = None,
    **overrides: Any,
) -> DuneToeStormParams:
    """Create model parameters from the volume-matched profile geometry."""
    values = dict(
        Ds=float(geometry.z_crest0),
        z0_init=float(geometry.z_toe0),
        tan_beta_f=float(geometry.tan_beta_f),
        alpha_rep_deg=float(geometry.alpha_rep_deg),
        seaward_buffer_m=40.0,
        landward_crest_width_m=float(geometry.landward_crest_width_m),
        landward_back_slope_m=float(geometry.landward_back_slope_m),
        landward_back_buffer_m=float(geometry.landward_back_buffer_m),
        tan_beta_back=float(geometry.tan_beta_back),
        z_back=float(geometry.z_back),
        use_profile_mesh=False,
    )
    values.update(overrides)
    allowed = _params_field_names()
    values = {k: v for k, v in values.items() if k in allowed}

    if base_params is None:
        return DuneToeStormParams(**values)
    return replace(base_params, **values)


def simulate_profile_event(
    geometry: TranslationGeometry,
    *,
    time_s: Sequence[float],
    T: Sequence[float],
    TWL: Sequence[float] | None = None,
    Ru: Sequence[float] | None = None,
    H0: Sequence[float] | None = None,
    base_params: DuneToeStormParams | None = None,
    runup_mode: str = "stockdon",
    **param_overrides: Any,
) -> tuple[DuneToeStormModel, DuneToeStormParams, dict[str, np.ndarray]]:
    """Run the dune erosion model for one translated real profile."""
    params = build_params_for_profile(geometry, base_params=base_params, **param_overrides)
    model = DuneToeStormModel(params)

    time_s = np.asarray(time_s, dtype=float)
    T = np.asarray(T, dtype=float)

    if TWL is not None:
        result = model.simulate_from_twl(
            time_s=time_s,
            TWL=np.asarray(TWL, dtype=float),
            T=T,
            H0_for_Cs=np.asarray(H0, dtype=float) if H0 is not None else None,
        )
    elif Ru is not None:
        result = model.simulate_from_runup(
            time_s=time_s,
            Ru=np.asarray(Ru, dtype=float),
            T=T,
            H0_for_Cs=np.asarray(H0, dtype=float) if H0 is not None else None,
        )
    elif H0 is not None:
        result = model.simulate_from_waves(time_s=time_s, H0=np.asarray(H0, dtype=float), T=T, runup_mode=runup_mode)
    else:
        raise ValueError("Provide one forcing among TWL, Ru or H0.")

    return model, params, result


# -----------------------------------------------------------------------------
# Profile projection and merge
# -----------------------------------------------------------------------------


def _normalise_time_index(n: int, t_idx: int) -> int:
    idx = int(t_idx)
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        raise IndexError(f"t_idx={t_idx} is out of bounds for series length {n}")
    return idx


def get_modeled_profile_local(
    model: DuneToeStormModel,
    result: Mapping[str, np.ndarray],
    *,
    t_idx: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the modelled calculation profile in local ``x,z`` coordinates."""
    z0 = np.asarray(result["z0"], dtype=float)
    x0 = np.asarray(result["x0"], dtype=float)
    Ds_ts = np.asarray(result.get("Ds_ts", np.full_like(z0, float(model.params.Ds))), dtype=float)
    tan_beta_D = float(np.asarray(result["tan_beta_D"], dtype=float).ravel()[0])
    idx = _normalise_time_index(z0.size, t_idx)

    x, z = model.build_profile_xy(
        float(z0[idx]),
        float(x0[idx]),
        float(Ds_ts[idx]),
        float(model.params.tan_beta_f),
        tan_beta_D,
        seaward_buffer_m=float(getattr(model.params, "seaward_buffer_m", 40.0)),
        landward_crest_width_m=float(getattr(model.params, "landward_crest_width_m", 0.0)),
        landward_back_slope_m=float(getattr(model.params, "landward_back_slope_m", 0.0)),
        landward_back_buffer_m=float(getattr(model.params, "landward_back_buffer_m", 0.0)),
        z_back=float(getattr(model.params, "z_back", 0.0)),
        tan_beta_back=float(getattr(model.params, "tan_beta_back", -1.0)),
    )
    return np.asarray(x, dtype=float), np.asarray(z, dtype=float)


def translate_modeled_features_to_real(
    geometry: TranslationGeometry,
    result: Mapping[str, np.ndarray],
    *,
    t_idx: int = -1,
) -> dict[str, float]:
    """Translate modelled toe and crest diagnostics back to real ``d`` coordinates."""
    z0 = np.asarray(result["z0"], dtype=float)
    x0 = np.asarray(result["x0"], dtype=float)
    Ds_ts = np.asarray(result.get("Ds_ts", np.full_like(z0, geometry.z_crest0)), dtype=float)
    tan_beta_D = float(np.asarray(result["tan_beta_D"], dtype=float).ravel()[0])
    idx = _normalise_time_index(z0.size, t_idx)

    x_toe = float(x0[idx])
    z_toe = float(z0[idx])
    Ds_i = float(Ds_ts[idx])
    x_crest = x_toe + (Ds_i - z_toe) / tan_beta_D if tan_beta_D > 0 else np.nan

    return {
        "d_toe_model": float(local_x_to_real_d(x_toe, geometry.d_toe0)),
        "z_toe_model": z_toe,
        "d_crest_model": float(local_x_to_real_d(x_crest, geometry.d_toe0)),
        "z_crest_model": Ds_i,
        "x_toe_local": x_toe,
        "x_crest_local": float(x_crest),
    }


def _interp_model_to_real_grid(
    d_real: np.ndarray,
    geometry: TranslationGeometry,
    model: DuneToeStormModel,
    result: Mapping[str, np.ndarray],
    *,
    t_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_local, z_model = get_modeled_profile_local(model, result, t_idx=t_idx)
    d_model = np.asarray(local_x_to_real_d(x_local, geometry.d_toe0), dtype=float)
    order = np.argsort(d_model)
    d_model = d_model[order]
    z_model = z_model[order]
    d_unique, idx = np.unique(d_model, return_index=True)
    z_unique = z_model[idx]

    z_on_real = np.full_like(d_real, np.nan, dtype=float)
    mask = (d_real >= d_unique.min()) & (d_real <= d_unique.max())
    z_on_real[mask] = np.interp(d_real[mask], d_unique, z_unique)
    return d_unique, z_unique, z_on_real


def _edge_taper(d: np.ndarray, left: float, right: float, width: float) -> np.ndarray:
    """Cosine taper equal to zero at both merge limits and one in the interior."""
    d = np.asarray(d, dtype=float)
    w = np.ones_like(d, dtype=float)
    width = float(width)
    if width <= 0.0 or right <= left:
        return w

    width = min(width, 0.5 * (right - left))
    left_zone = (d >= left) & (d < left + width)
    right_zone = (d > right - width) & (d <= right)

    if np.any(left_zone):
        r = np.clip((d[left_zone] - left) / width, 0.0, 1.0)
        w[left_zone] = 0.5 - 0.5 * np.cos(np.pi * r)
    if np.any(right_zone):
        r = np.clip((right - d[right_zone]) / width, 0.0, 1.0)
        w[right_zone] = 0.5 - 0.5 * np.cos(np.pi * r)

    return np.clip(w, 0.0, 1.0)


def _profile_volume_change(d: np.ndarray, z_before: np.ndarray, z_after: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 2:
        return 0.0
    return _volume_above_level(d[mask], z_after[mask], level=0.0) - _volume_above_level(d[mask], z_before[mask], level=0.0)


def check_modeled_result_quality(
    geometry: TranslationGeometry,
    result: Mapping[str, np.ndarray],
    *,
    t_idx: int = -1,
    max_retreat_m: float = 120.0,
    max_crest_lowering_m: float = 5.0,
) -> None:
    """Reject numerical blow-ups before they reach the real profile."""
    for key in ("z0", "x0", "Ds_ts", "tan_beta_D"):
        if key in result and not np.all(np.isfinite(np.asarray(result[key], dtype=float))):
            raise ValueError(f"Model output {key!r} contains non-finite values.")

    feat = translate_modeled_features_to_real(geometry, result, t_idx=t_idx)
    retreat = float(geometry.d_toe0 - feat["d_toe_model"])
    if np.isfinite(max_retreat_m) and abs(retreat) > float(max_retreat_m):
        raise ValueError(f"Modelled toe movement is outside QC limits: {retreat:g} m.")

    crest_change = float(feat["z_crest_model"] - geometry.z_crest0)
    if np.isfinite(max_crest_lowering_m) and crest_change < -float(max_crest_lowering_m):
        raise ValueError(f"Modelled crest lowering is outside QC limits: {crest_change:g} m.")


def merge_modeled_profile_into_real(
    profile_row: Mapping[str, Any],
    geometry: TranslationGeometry,
    model: DuneToeStormModel,
    result: Mapping[str, np.ndarray],
    *,
    t_idx: int = -1,
    d_col: str = "d",
    z_col: str | None = None,
    landward_limit_d: float | None = None,
    seaward_limit_d: float | None = None,
    blend_width_m: float = 10.0,
    merge_from_initial_toe_landward: bool = True,  # kept for compatibility; always true
    merge_strategy: str = "parametric_retreat",    # kept for compatibility; single strategy
    dune_landward_col: str | None = "d_dune_landward_polygon",
    deposition_volume_mode: str = "parametric_retreat",  # kept for compatibility; not used
    overwash_deposition_fraction: float = 1.0,     # kept for compatibility; not used
    max_vertical_change_m: float = np.inf,
    max_deposition_thickness_m: float = 2.0,       # kept for compatibility; not used
    max_retreat_m: float = 120.0,
    max_crest_lowering_m: float = 5.0,
    validate_model_result: bool = True,
) -> dict[str, np.ndarray]:
    """Apply the modelled toe/crest retreat to the observed profile.

    The calculation trapezoid is useful for running the storm model, but it is
    still an abstraction.  For the final real-profile product we therefore use
    the model in a reduced, more robust way:

    * the model gives the final toe position, final crest position and crest
      lowering;
    * the observed dune profile is horizontally warped between the landward
      polygon edge, crest and frontal toe so those three control points move to
      their modelled positions;
    * the newly exposed beachface seaward of the retreated toe follows the
      model foreshore slope;
    * only the two outer edges are blended back to the unchanged survey profile.

    This keeps the real along-profile texture where it is useful, while avoiding
    the artefacts produced by copying the full idealised trapezoid onto the
    survey mesh.
    """
    if validate_model_result:
        check_modeled_result_quality(
            geometry,
            result,
            t_idx=t_idx,
            max_retreat_m=max_retreat_m,
            max_crest_lowering_m=max_crest_lowering_m,
        )

    d_real, z_real, _ = get_profile_arrays(profile_row, d_col=d_col, z_col=z_col or geometry.z_col)
    d_model0, z_model0, z_model0_on_real = _interp_model_to_real_grid(d_real, geometry, model, result, t_idx=0)
    d_modelf, z_modelf, z_modelf_on_real = _interp_model_to_real_grid(d_real, geometry, model, result, t_idx=t_idx)
    feat = translate_modeled_features_to_real(geometry, result, t_idx=t_idx)

    d_toe_f = float(feat["d_toe_model"])
    z_toe_f = float(feat["z_toe_model"])
    d_crest_f = float(feat["d_crest_model"])
    z_crest_f = float(feat["z_crest_model"])

    if not (np.isfinite(d_toe_f) and np.isfinite(z_toe_f) and np.isfinite(d_crest_f) and np.isfinite(z_crest_f)):
        raise ValueError("Modelled toe/crest features are not finite.")
    if not (d_crest_f < d_toe_f):
        raise ValueError("Modelled geometry is inverted: final crest is not landward of final toe.")

    dmin = float(np.nanmin(d_real))
    dmax = float(np.nanmax(d_real))
    blend_width_m = max(float(blend_width_m), 0.0)

    # Horizontal movement is negative for landward retreat because d increases
    # seaward.  The landward edge is moved with the crest; this lets the eroded
    # dune occupy space behind the initial numerical domain instead of stopping
    # artificially at the original polygon edge.
    toe_shift = float(d_toe_f - geometry.d_toe0)
    crest_shift = float(d_crest_f - geometry.d_crest0)
    landward_shift = crest_shift
    d_landward_f = float(geometry.d_landward0 + landward_shift)

    # Source profile segment to warp.  Exact control points are inserted so the
    # interpolation honours toe/crest/landward coordinates even when the survey
    # grid does not sample them exactly.
    src_mask = (
        (d_real >= geometry.d_landward0)
        & (d_real <= geometry.d_toe0)
        & np.isfinite(d_real)
        & np.isfinite(z_real)
    )
    if src_mask.sum() < 2:
        raise ValueError("Observed dune segment is too short for parametric retreat merge.")

    control_d = np.asarray([geometry.d_landward0, geometry.d_crest0, geometry.d_toe0], dtype=float)
    control_shift = np.asarray([landward_shift, crest_shift, toe_shift], dtype=float)
    control_dz = np.asarray([0.0, z_crest_f - geometry.z_crest0, z_toe_f - geometry.z_toe0], dtype=float)

    d_src = np.concatenate([d_real[src_mask], control_d])
    z_src = np.concatenate([z_real[src_mask], [geometry.z_landward0, geometry.z_crest0, geometry.z_toe0]])

    order = np.argsort(d_src)
    d_src = d_src[order]
    z_src = z_src[order]
    d_src_unique, unique_idx = np.unique(d_src, return_index=True)
    z_src_unique = z_src[unique_idx]

    displacement = np.interp(d_src_unique, control_d, control_shift)
    dz_control = np.interp(d_src_unique, control_d, control_dz)
    d_body_final = d_src_unique + displacement
    z_body_final = z_src_unique + dz_control

    # The variable displacement can very occasionally make two neighbouring
    # source points collapse onto the same final d. Average those duplicates
    # before interpolation.
    body_df = pd.DataFrame({"d": d_body_final, "z": z_body_final})
    body_df = body_df[np.isfinite(body_df["d"]) & np.isfinite(body_df["z"])]
    body_df = body_df.groupby("d", as_index=False, sort=True)["z"].mean()
    if len(body_df) < 2:
        raise ValueError("Warped dune body has fewer than two valid points.")

    d_body_final = body_df["d"].to_numpy(dtype=float)
    z_body_final = body_df["z"].to_numpy(dtype=float)

    body_target = np.full_like(z_real, np.nan, dtype=float)
    body_mask = (d_real >= d_body_final.min()) & (d_real <= min(d_body_final.max(), d_toe_f))
    if np.any(body_mask):
        body_target[body_mask] = np.interp(d_real[body_mask], d_body_final, z_body_final)

    # Seaward of the retreated toe, rebuild the eroded beachface with the model
    # foreshore slope.  It is kept fully active until the initial toe, then
    # blended out over the seaward edge so the bathymetry stays unchanged.
    seaward_core = float(geometry.d_toe0)
    seaward = float(seaward_core + blend_width_m if seaward_limit_d is None else seaward_limit_d)
    seaward = min(seaward, dmax)
    if seaward <= d_toe_f:
        seaward = min(max(d_toe_f + blend_width_m, seaward_core), dmax)

    beach_target = np.full_like(z_real, np.nan, dtype=float)
    beach_mask = (d_real >= d_toe_f) & (d_real <= seaward)
    if np.any(beach_mask):
        beach_target[beach_mask] = z_toe_f - float(geometry.tan_beta_f) * (d_real[beach_mask] - d_toe_f)

    target_z = np.where(np.isfinite(body_target), body_target, beach_target)
    valid_target = np.isfinite(target_z) & np.isfinite(z_real)

    if valid_target.sum() < 2:
        raise ValueError("Parametric retreat target does not overlap the real profile.")

    landward = float(d_body_final.min() if landward_limit_d is None else landward_limit_d)
    landward = max(min(landward, seaward), dmin)
    if landward >= seaward:
        raise ValueError("Merge domain is empty after clipping to the observed profile extent.")

    # Blend only at the two outer edges.  Inside the morphodynamic core the
    # parametric target is written directly.
    weight = np.zeros_like(z_real, dtype=float)
    weight[valid_target] = 1.0

    if blend_width_m > 0.0:
        left_zone = valid_target & (d_real >= landward) & (d_real < landward + blend_width_m)
        if np.any(left_zone):
            r = np.clip((d_real[left_zone] - landward) / blend_width_m, 0.0, 1.0)
            weight[left_zone] *= 0.5 - 0.5 * np.cos(np.pi * r)

        right_zone = valid_target & (d_real > seaward_core) & (d_real <= seaward)
        if np.any(right_zone):
            r = np.clip((seaward - d_real[right_zone]) / max(seaward - seaward_core, 1.0e-9), 0.0, 1.0)
            weight[right_zone] *= 0.5 - 0.5 * np.cos(np.pi * r)

    z_merged = z_real.copy()
    z_merged[valid_target] = (
        (1.0 - weight[valid_target]) * z_real[valid_target]
        + weight[valid_target] * target_z[valid_target]
    )

    # Optional emergency guard. It is disabled by default because the merge is
    # already controlled by physical toe/crest displacements.
    dz_applied = z_merged - z_real
    if np.isfinite(max_vertical_change_m) and max_vertical_change_m > 0.0:
        limit = abs(float(max_vertical_change_m))
        dz_applied[valid_target] = np.clip(dz_applied[valid_target], -limit, limit)
        z_merged[valid_target] = z_real[valid_target] + dz_applied[valid_target]

    if not np.all(np.isfinite(z_merged[np.isfinite(z_real)])):
        raise ValueError("Merged profile contains non-finite elevations.")

    calc_mask0 = (d_model0 >= geometry.d_landward0) & (d_model0 <= geometry.d_toe0)
    calc_maskf = (d_modelf >= min(d_landward_f, d_modelf.min())) & (d_modelf <= max(seaward_core, d_toe_f))
    calc_initial_volume = _volume_above_level(d_model0[calc_mask0], z_model0[calc_mask0], level=0.0)
    calc_final_volume = _volume_above_level(d_modelf[calc_maskf], z_modelf[calc_maskf], level=0.0)
    target_volume_change = float(calc_final_volume - calc_initial_volume)
    merged_volume_change = _profile_volume_change(d_real, z_real, z_merged, valid_target)

    dz_model = np.full_like(z_real, np.nan, dtype=float)
    dz_model[valid_target] = target_z[valid_target] - z_real[valid_target]

    return {
        "d_real": d_real,
        "z_real": z_real,
        "d_model": d_modelf,
        "z_model": z_modelf,
        "z_model_on_real": z_modelf_on_real,
        "d_model_initial": d_model0,
        "z_model_initial": z_model0,
        "z_model_initial_on_real": z_model0_on_real,
        "z_parametric_target_on_real": target_z,
        "z_merged": z_merged,
        "merge_mask": valid_target,
        "merge_weight": weight,
        "landward_limit_d": float(landward),
        "initial_landward_limit_d": float(geometry.d_landward0),
        "final_landward_limit_d": float(d_landward_f),
        "seaward_limit_d": float(seaward),
        "merge_strategy": np.array(["parametric_retreat_beachface"]),
        "dz_model_raw": dz_model,
        "dz_applied": dz_applied,
        "volume_eroded_front_m2": float(max(0.0, -merged_volume_change)),
        "volume_deposited_landward_m2": float(max(0.0, merged_volume_change)),
        "volume_balance_error_m2": float(merged_volume_change - target_volume_change),
        "overwash_volume_available_m2": np.nan,
        "calc_initial_volume_above_0_m2": float(calc_initial_volume),
        "calc_final_volume_above_0_m2": float(calc_final_volume),
        "calc_target_volume_change_m2": float(target_volume_change),
        "merged_volume_change_m2": float(merged_volume_change),
        "volume_scale_factor": 1.0,
        "merge_datum_offset_m": 0.0,
        "d_toe_shift_m": float(toe_shift),
        "d_crest_shift_m": float(crest_shift),
        "z_crest_change_m": float(z_crest_f - geometry.z_crest0),
    }

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def plot_translation_summary(
    profile_row: Mapping[str, Any],
    geometry: TranslationGeometry,
    model: DuneToeStormModel,
    result: Mapping[str, np.ndarray],
    *,
    t_idx: int = -1,
    d_col: str = "d",
    z_col: str | None = None,
    blend_width_m: float = 8.0,
    merge_from_initial_toe_landward: bool = True,
    merge_strategy: str = "model_delta",
    dune_landward_col: str | None = "d_dune_landward_polygon",
    deposition_volume_mode: str = "model_delta",
    overwash_deposition_fraction: float = 1.0,
    max_vertical_change_m: float = 6.0,
    max_deposition_thickness_m: float = 2.0,
    max_retreat_m: float = 120.0,
    max_crest_lowering_m: float = 5.0,
    clip_modeled_plot_to_merge_window: bool = False,
    plot_proxy: bool = True,
    proxy_col: str = "Y_df_AI_proxy",
    proxy_seaward_col: str = "d_dune_seaward_polygon",
    proxy_landward_col: str = "d_dune_landward_polygon",
    figsize: tuple[float, float] = (12, 8),
    xlim: tuple[float, float] = (0, 1000),
    ylim: tuple[float, float] = (-7, 15),
):
    """Diagnostic plot for local calculation geometry and real-profile transfer."""
    proj = merge_modeled_profile_into_real(
        profile_row,
        geometry,
        model,
        result,
        t_idx=t_idx,
        d_col=d_col,
        z_col=z_col,
        blend_width_m=blend_width_m,
        max_vertical_change_m=max_vertical_change_m,
        max_retreat_m=max_retreat_m,
        max_crest_lowering_m=max_crest_lowering_m,
    )
    feat = translate_modeled_features_to_real(geometry, result, t_idx=t_idx)
    x_ini, z_ini = get_modeled_profile_local(model, result, t_idx=0)
    x_fin, z_fin = get_modeled_profile_local(model, result, t_idx=t_idx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=False, gridspec_kw={"hspace": 0.20})

    ax1.plot(x_ini, z_ini, lw=1.8, color="0.45", label="calculation profile, initial")
    ax1.plot(x_fin, z_fin, lw=2.1, color="tab:orange", label="calculation profile, final")
    ax1.scatter([0.0], [geometry.z_toe0], s=55, zorder=5, label="initial toe")
    ax1.scatter([geometry.x_crest_obs], [geometry.z_crest0], s=55, zorder=5, label="initial crest")
    ax1.scatter([geometry.x_landward_obs], [geometry.z_landward0], s=55, zorder=5, label="landward boundary")
    ax1.axhline(0.0, ls=":", lw=1.0, color="0.5")
    ax1.set_xlabel("Local x (m, positive landward)")
    ax1.set_ylabel("z (m)")
    ax1.set_title(
        "Calculation geometry | "
        f"Vreal={geometry.real_volume_above_0_m2:.2f} m²/m, "
        f"Vcalc={geometry.calc_volume_above_0_m2:.2f} m²/m "
        f"({geometry.calc_volume_match_status})"
    )
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", ncol=2)

    ax2.plot(proj["d_real"], proj["z_real"], color="0.35", lw=1.8, label="real profile")
    ax2.plot(proj["d_model_initial"], proj["z_model_initial"], color="0.55", lw=1.5, ls="--", label="initial calc profile mapped")
    ax2.plot(proj["d_model"], proj["z_model"], color="tab:orange", lw=2.0, label="final calc profile mapped")
    if "z_parametric_target_on_real" in proj:
        ax2.plot(
            proj["d_real"],
            proj["z_parametric_target_on_real"],
            color="tab:green",
            lw=1.5,
            ls=":",
            label="parametric target",
        )
    ax2.plot(proj["d_real"], proj["z_merged"], color="tab:blue", lw=2.2, label="merged profile")

    ax2.axvspan(proj["landward_limit_d"], proj["seaward_limit_d"], alpha=0.08, color="tab:blue", label="merge domain")
    ax2.scatter([geometry.d_toe0], [geometry.z_toe0], s=60, zorder=6, label="toe used")
    ax2.scatter([geometry.d_crest0], [geometry.z_crest0], s=60, zorder=6, label="crest used")
    ax2.scatter([geometry.d_landward0], [geometry.z_landward0], s=60, zorder=6, label="landward boundary")

    if plot_proxy:
        proxy_d = _get_first_numeric((profile_row, proxy_col))
        poly_seaward_d = _get_first_numeric((profile_row, proxy_seaward_col))
        poly_landward_d = _get_first_numeric((profile_row, proxy_landward_col))
        if np.isfinite(poly_landward_d):
            ax2.axvline(poly_landward_d, ls="-.", lw=1.1, color="0.35", alpha=0.85, label="polygon landward")
        if np.isfinite(poly_seaward_d):
            ax2.axvline(poly_seaward_d, ls="-.", lw=1.1, color="0.65", alpha=0.85, label="polygon seaward")
        if np.isfinite(proxy_d):
            proxy_z = float(np.interp(proxy_d, proj["d_real"], proj["z_real"]))
            ax2.scatter([proxy_d], [proxy_z], s=115, marker="P", color="tab:cyan", edgecolor="black", linewidth=0.6, zorder=7, label="proxy")

    ax2.scatter([feat["d_toe_model"]], [feat["z_toe_model"]], s=75, marker="*", zorder=7, label="modelled toe")
    ax2.scatter([feat["d_crest_model"]], [feat["z_crest_model"]], s=75, marker="^", zorder=7, label="modelled crest")
    ax2.axhline(0.0, ls=":", lw=1.0, color="0.5")
    ax2.set_xlabel("Real profile d (m, increasing seaward)")
    ax2.set_ylabel("z (m)")
    ax2.set_title(
        "Parametric real-profile transfer | "
        f"target ΔV={proj['calc_target_volume_change_m2']:.2f}, "
        f"merged ΔV={proj['merged_volume_change_m2']:.2f}, "
        f"error={proj['volume_balance_error_m2']:.2f} m²/m"
    )
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", ncol=2)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    return fig, {"projection": proj, "features": feat}


def summarize_simulation(
    geometry: TranslationGeometry,
    params: DuneToeStormParams,
    result: Mapping[str, np.ndarray],
    *,
    t_idx: int = -1,
) -> dict[str, float]:
    feat = translate_modeled_features_to_real(geometry, result, t_idx=t_idx)
    out = asdict(geometry)
    out.update(feat)
    out.update(
        {
            "Ds_final": float(np.asarray(result.get("Ds_ts", np.array([params.Ds])), dtype=float)[t_idx]),
            "z0_final": float(np.asarray(result["z0"], dtype=float)[t_idx]),
            "x0_final": float(np.asarray(result["x0"], dtype=float)[t_idx]),
        }
    )
    return out
