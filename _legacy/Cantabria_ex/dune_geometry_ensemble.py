from __future__ import annotations

"""Ensemble dune-geometry detector for cross-shore profiles.

The detector is designed for operational dune-erosion workflows where the goal is
not to find an isolated morphological point, but to build a coherent initial
geometry for the storm-impact model.  It combines several independent cues
(curvature, beach/dune breakpoints, relative relief and perpendicular distance),
optionally augmented with pybeach predictions, then scores complete
heel--crest--toe geometries under simple physical constraints.

The public functions intentionally match the previous constrained detector API:
``apply_to_dataset()``, ``plot_profile_diagnostics()`` and
``smooth_longitudinal_feature()``.  This keeps ``run_dune_workflow.py`` small and
makes the detector swappable without changing the rest of the pipeline.
"""

from dataclasses import dataclass
import ast
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:  # optional, only used for nicer progress bars
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _progress(iterable, *, show: bool = True, desc: str = "", total: int | None = None, unit: str = "it"):
    if show and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)
    return iterable


def _load_pybeach_profile_class():
    """Return pybeach.Profile when pybeach is installed, otherwise None.

    pybeach has exposed ``Profile`` from slightly different import paths across
    releases/examples. Keeping this import lazy lets the workflow run with the
    internal ensemble unchanged in environments where pybeach is not installed.
    """
    try:
        from pybeach.beach import Profile  # type: ignore
        return Profile
    except Exception:
        try:
            from pybeach import Profile  # type: ignore
            return Profile
        except Exception:
            return None


def _scalar_prediction(value: Any) -> tuple[float, float | None]:
    """Extract a coordinate and optional probability from pybeach outputs.

    Depending on pybeach version/method, predictions may be returned as a
    scalar, an array, or a ``(prediction, probability)`` tuple. The workflow only
    needs the coordinate as an extra candidate; probabilities are used as a soft
    bonus when available.
    """
    prob: float | None = None
    pred = value
    if isinstance(value, tuple) and value:
        pred = value[0]
        if len(value) > 1:
            try:
                parr = np.asarray(value[1], dtype=float).ravel()
                if parr.size:
                    prob = float(np.nanmax(parr))
            except Exception:
                prob = None
    try:
        arr = np.asarray(pred, dtype=float).ravel()
        if arr.size:
            x = float(arr[0])
            return (x if np.isfinite(x) else np.nan), prob
    except Exception:
        pass
    return np.nan, prob


def _pybeach_profile(d: np.ndarray, z: np.ndarray, lo: float, hi: float, *, min_points: int = 20):
    """Create a pybeach Profile over a clean search segment.

    Returns ``None`` when pybeach is unavailable or the segment is too short. The
    coordinates are passed in the same cross-shore convention as the workflow
    (``d`` increasing seaward), which matches pybeach's example convention.
    """
    Profile = _load_pybeach_profile_class()
    if Profile is None:
        return None
    dw, zw = _window_points(d, z, lo, hi)
    if dw.size < int(min_points):
        return None
    try:
        order = np.argsort(dw)
        dw, zw = dw[order], zw[order]
        dw, keep = np.unique(dw, return_index=True)
        zw = zw[keep]
        if dw.size < int(min_points):
            return None
        return Profile(dw, zw)
    except Exception:
        return None




def _pybeach_input_z(d: np.ndarray, z: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Return the elevation series passed to pybeach.

    The checked profile can be relatively coarse/noisy at metre scale.  Pybeach
    methods are sensitive to local curvature and relief, so by default we pass a
    lightly smoothed profile to pybeach while keeping all reported elevations
    interpolated from the original input profile.  This gives pybeach a cleaner
    morphology without overwriting the observed topography in the final output.
    """
    if not bool(kwargs.get("pybeach_use_smoothed_profile", True)):
        return z

    median_window = float(
        kwargs.get(
            "pybeach_smooth_median_window_m",
            kwargs.get("median_window_m", 5.0),
        )
    )
    mean_window = float(
        kwargs.get(
            "pybeach_smooth_mean_window_m",
            kwargs.get("sg_window_m", kwargs.get("mean_window_m", 11.0)),
        )
    )
    return _smooth_profile(z, d, median_window_m=median_window, mean_window_m=mean_window)


def _pybeach_toe_candidates(
    d: np.ndarray,
    z: np.ndarray,
    lo: float,
    hi: float,
    *,
    use_pybeach: bool = False,
    pybeach_methods: Sequence[str] = ("ml", "mc", "rr", "pd"),
    pybeach_ml_models: Sequence[str] = ("mixed_clf",),
    pybeach_min_points: int = 20,
    **kwargs: Any,
) -> list[Candidate]:
    """Optional pybeach dune-toe candidates.

    These candidates are added as another vote in the ensemble. They never
    replace the internal detector and are ignored silently if pybeach is missing
    or a given pybeach method fails on a profile.
    """
    if not use_pybeach:
        return []
    z_pybeach = _pybeach_input_z(d, z, **kwargs)
    p = _pybeach_profile(d, z_pybeach, lo, hi, min_points=pybeach_min_points)
    if p is None:
        return []

    candidates: list[Candidate] = []
    methods = {str(m).lower() for m in pybeach_methods}
    if "ml" in methods:
        for model_name in pybeach_ml_models:
            try:
                x, prob = _scalar_prediction(p.predict_dunetoe_ml(str(model_name)))
                if np.isfinite(x) and lo <= x <= hi:
                    score = -float(prob) if prob is not None and np.isfinite(prob) else -0.35
                    candidates.append(Candidate(float(x), _interp(d, z, float(x)), f"pybeach_ml_{model_name}", score=score))
            except Exception:
                continue
    if "mc" in methods:
        try:
            x, _ = _scalar_prediction(p.predict_dunetoe_mc())
            if np.isfinite(x) and lo <= x <= hi:
                candidates.append(Candidate(float(x), _interp(d, z, float(x)), "pybeach_mc", score=-0.20))
        except Exception:
            pass
    if "rr" in methods:
        try:
            x, _ = _scalar_prediction(p.predict_dunetoe_rr())
            if np.isfinite(x) and lo <= x <= hi:
                candidates.append(Candidate(float(x), _interp(d, z, float(x)), "pybeach_rr", score=-0.15))
        except Exception:
            pass
    if "pd" in methods:
        try:
            x, _ = _scalar_prediction(p.predict_dunetoe_pd())
            if np.isfinite(x) and lo <= x <= hi:
                candidates.append(Candidate(float(x), _interp(d, z, float(x)), "pybeach_pd", score=-0.10))
        except Exception:
            pass

    return _deduplicate_candidates(candidates, min_distance_m=2.0)


def _pybeach_crest_candidates(
    d: np.ndarray,
    z: np.ndarray,
    lo: float,
    hi: float,
    *,
    use_pybeach: bool = False,
    pybeach_min_points: int = 20,
    **kwargs: Any,
) -> list[Candidate]:
    """Optional pybeach dune-crest candidate."""
    if not use_pybeach:
        return []
    z_pybeach = _pybeach_input_z(d, z, **kwargs)
    p = _pybeach_profile(d, z_pybeach, lo, hi, min_points=pybeach_min_points)
    if p is None:
        return []
    try:
        x, _ = _scalar_prediction(p.predict_dunecrest())
    except Exception:
        return []
    if not np.isfinite(x) or not (lo <= x <= hi):
        return []
    return [Candidate(float(x), _interp(d, z, float(x)), "pybeach_crest", score=-0.12)]




def _preferred_pybeach_toe(
    candidates: list[Candidate],
    *,
    priority: Sequence[str] = ("ml", "mc", "rr", "pd"),
) -> Candidate | None:
    """Pick one toe from pybeach candidates using method priority.

    The candidate list can contain multiple pybeach methods.  In pybeach-only
    mode we keep the choice explicit and deterministic: first choose the best
    candidate from the highest-priority method that returned a valid point, then
    fall back to the lowest-score pybeach candidate.
    """
    if not candidates:
        return None
    order = [str(x).lower() for x in priority]
    for method in order:
        if method == "ml":
            subset = [c for c in candidates if c.source.startswith("pybeach_ml")]
        else:
            subset = [c for c in candidates if c.source == f"pybeach_{method}"]
        if subset:
            return sorted(subset, key=lambda c: (c.score, c.d))[0]
    return sorted(candidates, key=lambda c: (c.score, c.d))[0]


def _detect_profile_geometry_pybeach_only(row: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Detect toe/crest from pybeach only, with a minimal internal heel fallback.

    pybeach provides dune-toe and dune-crest methods, but no operational dune-heel
    detector.  The erosion workflow still needs a landward mesh boundary, so this
    mode keeps toe and crest exclusive to pybeach and estimates only the heel with
    the same conservative fallback used by the internal detector.
    """
    d_col = kwargs.get("d_col", "d")
    z_col = kwargs.get("z_col", "z")
    d, z = _profile_arrays(row, d_col=d_col, z_col=z_col)
    profile_idx = row.get("profile_idx", getattr(row, "name", np.nan))
    if d.size < 12:
        raise ValueError("Profile has too few finite points for pybeach-only detection.")

    search_lo, search_hi, search_source = _search_window(row, d, **kwargs)
    dw, zw = _window_points(d, z, search_lo, search_hi)
    if dw.size < int(kwargs.get("pybeach_min_points", 20)):
        raise ValueError(f"Search segment too short for pybeach within [{search_lo:.3f}, {search_hi:.3f}].")

    if _load_pybeach_profile_class() is None:
        raise ImportError("DETECTION_MODE='pybeach_only' requires pybeach to be installed in the active environment.")

    pb_kwargs = dict(kwargs)
    pb_kwargs["use_pybeach"] = True
    toe_candidates = _pybeach_toe_candidates(d, z, search_lo, search_hi, **pb_kwargs)
    toe = _preferred_pybeach_toe(
        toe_candidates,
        priority=kwargs.get("pybeach_only_toe_method_priority", ("ml", "mc", "rr", "pd")),
    )
    if toe is None:
        raise ValueError("pybeach did not return a valid dune-toe candidate.")

    crest_candidates = _pybeach_crest_candidates(d, z, search_lo, search_hi, **pb_kwargs)
    if not crest_candidates and bool(kwargs.get("pybeach_only_allow_internal_crest_fallback", False)):
        z_smooth_tmp = _smooth_profile(
            z,
            d,
            median_window_m=kwargs.get("median_window_m", 5.0),
            mean_window_m=kwargs.get("sg_window_m", kwargs.get("mean_window_m", 11.0)),
        )
        crest_candidates = _crest_candidates(d, z_smooth_tmp, search_lo, min(search_hi, toe.d - 2.0), **kwargs)
        crest_candidates = [Candidate(c.d, c.z, "internal_crest_fallback", c.score) for c in crest_candidates]

    if not crest_candidates:
        raise ValueError("pybeach did not return a valid dune-crest candidate.")

    # The crest must lie landward of the toe for the convention used here
    # (d increases seaward). Prefer the highest compatible pybeach crest.
    crest_pool = [c for c in crest_candidates if c.d < toe.d - float(kwargs.get("min_toe_crest_gap_m", 4.0))]
    if not crest_pool:
        raise ValueError("pybeach crest/toe order is not physically valid for this profile.")
    crest = sorted(crest_pool, key=lambda c: (-c.z, c.score, c.d))[0]

    if bool(kwargs.get("pybeach_only_refine_crest_freeboard", False)):
        refine_window = float(kwargs.get("crest_refine_window_m", 12.0))
        md = (d >= crest.d - refine_window) & (d <= crest.d + refine_window) & (d >= search_lo) & (d <= min(search_hi, toe.d))
        if md.sum() >= 3:
            ii = np.where(md)[0][int(np.nanargmax(z[md]))]
            crest = Candidate(float(d[ii]), float(z[ii]), crest.source + "+local_freeboard", crest.score)

    z_smooth = _smooth_profile(
        z,
        d,
        median_window_m=kwargs.get("median_window_m", 5.0),
        mean_window_m=kwargs.get("sg_window_m", kwargs.get("mean_window_m", 11.0)),
    )
    heel_candidates = _heel_candidates_for_crest(d, z, z_smooth, crest, search_lo, row, **kwargs)
    if not heel_candidates:
        raise ValueError("No valid heel fallback found for pybeach-only geometry.")

    scored: list[GeometryCandidate] = []
    for heel in heel_candidates:
        geom = _score_geometry(d, z, heel, crest, toe, row, **kwargs)
        if geom is not None:
            scored.append(geom)

    if scored:
        best = min(scored, key=lambda g: g.score)
    else:
        # Very conservative last resort: keep the best landward heel candidate
        # and compute a lightweight diagnostic score.  This is rare, but keeps
        # pybeach-only extraction usable for toe/crest shapefiles even when the
        # backdune side is poorly defined.
        heel = sorted(heel_candidates, key=lambda c: ("fallback" in c.source, abs(c.d - crest.d), c.score))[0]
        z_toe = _interp(d, z, toe.d)
        z_crest = _interp(d, z, crest.d)
        z_heel = _interp(d, z, heel.d)
        face_d, face_z = _window_points(d, z, crest.d, toe.d)
        beach_d, beach_z = _window_points(d, z, toe.d, min(toe.d + float(kwargs.get("beach_slope_window_m", 50.0)), d[-1]))
        face_slope, _, face_rmse, _ = _robust_line_fit(face_d, face_z)
        beach_slope, _, beach_rmse, _ = _robust_line_fit(beach_d, beach_z)
        tan_face = abs(face_slope) if np.isfinite(face_slope) else np.nan
        tan_beach = abs(beach_slope) if np.isfinite(beach_slope) else np.nan
        relief = z_crest - z_toe if np.isfinite(z_crest) and np.isfinite(z_toe) else np.nan
        score = float((face_rmse if np.isfinite(face_rmse) else 1.0) + (beach_rmse if np.isfinite(beach_rmse) else 1.0) + 0.4)
        confidence = float(1.0 / (1.0 + np.exp(2.8 * (score - 0.75))))
        confidence = max(0.0, min(1.0, confidence))
        best = GeometryCandidate(
            heel=Candidate(heel.d, z_heel, heel.source, heel.score),
            crest=Candidate(crest.d, z_crest, crest.source, crest.score),
            toe=Candidate(toe.d, z_toe, toe.source, toe.score),
            score=score,
            confidence=confidence,
            components={
                "fit_score": score,
                "face_rmse": float(face_rmse) if np.isfinite(face_rmse) else np.nan,
                "beach_rmse": float(beach_rmse) if np.isfinite(beach_rmse) else np.nan,
                "back_rmse": np.nan,
                "relief": float(relief) if np.isfinite(relief) else np.nan,
                "relief_penalty": np.nan,
                "slope_separation": float(tan_face - tan_beach) if np.isfinite(tan_face) and np.isfinite(tan_beach) else np.nan,
                "slope_penalty": np.nan,
                "polygon_penalty": np.nan,
                "source_penalty": np.nan,
                "tan_beta_f": float(tan_beach) if np.isfinite(tan_beach) else np.nan,
                "tan_beta_D": float(tan_face) if np.isfinite(tan_face) else np.nan,
                "tan_beta_back": np.nan,
            },
            sources={"heel": heel.source, "crest": crest.source, "toe": toe.source},
        )

    comp = best.components
    status = "ok" if best.confidence >= float(kwargs.get("ok_confidence_threshold", 0.45)) else "low_confidence"
    return {
        "profile_idx": profile_idx,
        "detection_status": status,
        "detection_method": "pybeach_only_toe_crest_internal_heel",
        "pybeach_enabled": True,
        "pybeach_available": True,
        "pybeach_used": True,
        "pybeach_only": True,
        "detection_confidence": best.confidence,
        "geometry_score": best.score,
        "geometry_score_components": dict(comp),
        "detection_search_source": search_source,
        "d_detection_search_min": float(search_lo),
        "d_detection_search_max": float(search_hi),
        "d_toe_final": float(best.toe.d),
        "z_toe_final": float(best.toe.z),
        "d_crest": float(best.crest.d),
        "z_crest": float(best.crest.z),
        "d_heel_final": float(best.heel.d),
        "z_heel_final": float(best.heel.z),
        "toe_source": best.sources["toe"],
        "crest_source": best.sources["crest"],
        "heel_source": best.sources["heel"],
        "toe_confidence": _feature_confidence(best, "toe"),
        "crest_confidence": _feature_confidence(best, "crest"),
        "heel_confidence": _feature_confidence(best, "heel"),
        "tan_beta_f_detection": comp.get("tan_beta_f", np.nan),
        "tan_beta_D_detection": comp.get("tan_beta_D", np.nan),
        "tan_beta_back_detection": comp.get("tan_beta_back", np.nan),
        "face_rmse": comp.get("face_rmse", np.nan),
        "beach_rmse": comp.get("beach_rmse", np.nan),
        "back_rmse": comp.get("back_rmse", np.nan),
        "dune_relief_m": comp.get("relief", np.nan),
        "slope_separation": comp.get("slope_separation", np.nan),
        "n_geometry_candidates": len(toe_candidates) + len(crest_candidates) + len(heel_candidates),
        "error": np.nan,
    }

def _finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(pd.to_numeric(value, errors="coerce"))
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _as_array(value: Any) -> np.ndarray:
    """Return a 1-D float array from arrays, lists or stringified lists.

    Several profile sources store arrays as strings containing ``np.float64``
    wrappers.  This parser keeps the detector independent of the exact pickle
    representation used upstream.
    """
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).ravel()
    if isinstance(value, (list, tuple, pd.Series)):
        try:
            return np.asarray(value, dtype=float).ravel()
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.asarray([], dtype=float)
        text = text.replace("np.float64", "")
        text = text.replace("array", "")
        try:
            parsed = ast.literal_eval(text)
            return np.asarray(parsed, dtype=float).ravel()
        except Exception:
            cleaned = text.replace("[", " ").replace("]", " ").replace(",", " ")
            return np.fromstring(cleaned, sep=" ", dtype=float)
    try:
        return np.asarray(value, dtype=float).ravel()
    except Exception:
        return np.asarray([], dtype=float)


def _profile_arrays(row: Mapping[str, Any], *, d_col: str = "d", z_col: str = "z") -> tuple[np.ndarray, np.ndarray]:
    d = _as_array(row.get(d_col))
    z = _as_array(row.get(z_col))
    n = min(d.size, z.size)
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    d = d[:n].astype(float)
    z = z[:n].astype(float)
    mask = np.isfinite(d) & np.isfinite(z)
    d, z = d[mask], z[mask]
    if d.size < 2:
        return d, z
    order = np.argsort(d)
    d, z = d[order], z[order]
    d_unique, idx = np.unique(d, return_index=True)
    return d_unique, z[idx]


def _interp(d: np.ndarray, z: np.ndarray, x: float) -> float:
    if d.size < 2 or not np.isfinite(x) or x < d[0] or x > d[-1]:
        return np.nan
    return float(np.interp(x, d, z))


def _spacing(d: np.ndarray) -> float:
    if d.size < 2:
        return 1.0
    dx = np.diff(d)
    dx = dx[np.isfinite(dx) & (dx > 0)]
    if dx.size == 0:
        return 1.0
    return float(np.nanmedian(dx))


def _window_points(d: np.ndarray, z: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if hi < lo:
        lo, hi = hi, lo
    m = (d >= lo) & (d <= hi) & np.isfinite(z)
    return d[m], z[m]


def _rolling_median(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    n = int(max(1, n))
    if n % 2 == 0:
        n += 1
    if n <= 1 or x.size < 3:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(n, center=True, min_periods=1).median().to_numpy(dtype=float)


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    n = int(max(1, n))
    if n % 2 == 0:
        n += 1
    if n <= 1 or x.size < 3:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(n, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _smooth_profile(z: np.ndarray, d: np.ndarray, *, median_window_m: float = 5.0, mean_window_m: float = 9.0) -> np.ndarray:
    dx = _spacing(d)
    n_med = max(3, int(round(median_window_m / max(dx, 1e-6))))
    n_mean = max(3, int(round(mean_window_m / max(dx, 1e-6))))
    return _rolling_mean(_rolling_median(z, n_med), n_mean)


def _robust_line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    """Return slope, intercept, robust RMSE and number of points."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return np.nan, np.nan, np.inf, int(x.size)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return np.nan, np.nan, np.inf, int(x.size)
    residual = y - (slope * x + intercept)
    med = np.nanmedian(residual)
    mad = np.nanmedian(np.abs(residual - med))
    sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(residual)
    if np.isfinite(sigma) and sigma > 0:
        keep = np.abs(residual - med) <= 3.5 * sigma
        if keep.sum() >= 2 and keep.sum() < x.size:
            try:
                slope, intercept = np.polyfit(x[keep], y[keep], 1)
                residual = y[keep] - (slope * x[keep] + intercept)
            except Exception:
                pass
    rmse = float(np.sqrt(np.nanmean(residual**2))) if residual.size else np.inf
    return float(slope), float(intercept), rmse, int(x.size)


def _local_extrema_indices(y: np.ndarray, *, mode: str = "max") -> np.ndarray:
    if y.size < 3:
        return np.asarray([], dtype=int)
    if mode == "max":
        m = (y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]) & ((y[1:-1] > y[:-2]) | (y[1:-1] > y[2:]))
    else:
        m = (y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:]) & ((y[1:-1] < y[:-2]) | (y[1:-1] < y[2:]))
    return np.where(m)[0] + 1


def _percentile_scale(values: np.ndarray, default: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    scale = np.nanpercentile(np.abs(values), 75)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(values)
    return float(scale) if np.isfinite(scale) and scale > 0 else default


def _relative_relief(d: np.ndarray, z: np.ndarray, half_window_m: float) -> np.ndarray:
    rr = np.full_like(z, np.nan, dtype=float)
    for i, x in enumerate(d):
        m = (d >= x - half_window_m) & (d <= x + half_window_m)
        zz = z[m]
        zz = zz[np.isfinite(zz)]
        if zz.size < 3:
            continue
        zmin, zmax = float(np.nanmin(zz)), float(np.nanmax(zz))
        if zmax <= zmin:
            continue
        rr[i] = (z[i] - zmin) / (zmax - zmin)
    return rr


def _perpendicular_distance_candidates(d: np.ndarray, z: np.ndarray, crest_d: float, search_hi: float, *, max_candidates: int = 4) -> list[float]:
    """Candidates from maximum distance to the crest--seaward-end line."""
    c_z = _interp(d, z, crest_d)
    e_z = _interp(d, z, search_hi)
    if not np.isfinite(c_z) or not np.isfinite(e_z) or search_hi <= crest_d:
        return []
    m = (d > crest_d) & (d <= search_hi)
    x = d[m]
    y = z[m]
    if x.size < 4:
        return []
    x1, y1 = crest_d, c_z
    x2, y2 = search_hi, e_z
    denom = math.hypot(y2 - y1, x2 - x1)
    if denom <= 0:
        return []
    dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
    order = np.argsort(dist)[::-1][:max_candidates]
    return [float(x[i]) for i in order if np.isfinite(dist[i])]


@dataclass
class Candidate:
    d: float
    z: float
    source: str
    score: float = 0.0


@dataclass
class GeometryCandidate:
    heel: Candidate
    crest: Candidate
    toe: Candidate
    score: float
    confidence: float
    components: dict[str, float]
    sources: dict[str, str]


# -----------------------------------------------------------------------------
# Search windows and candidates
# -----------------------------------------------------------------------------
def _search_window(row: Mapping[str, Any], d: np.ndarray, *, search_landward_col: str = "d_dune_landward_polygon", search_seaward_col: str = "d_dune_seaward_polygon", search_landward_buffer_m: float = 100.0, search_seaward_buffer_m: float = 100.0, **_: Any) -> tuple[float, float, str]:
    dmin = float(np.nanmin(d)) if d.size else np.nan
    dmax = float(np.nanmax(d)) if d.size else np.nan
    land = _finite_float(row.get(search_landward_col), np.nan)
    sea = _finite_float(row.get(search_seaward_col), np.nan)
    if np.isfinite(land) and np.isfinite(sea) and sea > land:
        lo = max(dmin, land - float(search_landward_buffer_m))
        hi = min(dmax, sea + float(search_seaward_buffer_m))
        return float(lo), float(hi), "polygon_buffer"
    return dmin, dmax, "profile_extent"


def _crest_candidates(d: np.ndarray, z: np.ndarray, search_lo: float, search_hi: float, *, min_prominence_m: float = 0.35, max_candidates: int = 12, **_: Any) -> list[Candidate]:
    m = (d >= search_lo) & (d <= search_hi)
    dw, zw = d[m], z[m]
    if dw.size < 5:
        return []
    peaks_local = _local_extrema_indices(zw, mode="max")
    if peaks_local.size == 0:
        peaks_local = np.asarray([int(np.nanargmax(zw))])
    z10 = float(np.nanpercentile(zw, 10))
    z90 = float(np.nanpercentile(zw, 90))
    relief = max(z90 - z10, 0.1)
    candidates: list[Candidate] = []
    for i in peaks_local:
        left = zw[max(0, i - 8):i + 1]
        right = zw[i:min(zw.size, i + 9)]
        neigh_low = float(np.nanmin(np.r_[left, right])) if left.size + right.size > 0 else np.nan
        prom = float(zw[i] - neigh_low) if np.isfinite(neigh_low) else 0.0
        # Also keep high, broad maxima even if local prominence is small.
        highness = (zw[i] - z10) / relief
        if prom >= min_prominence_m or highness > 0.65:
            candidates.append(Candidate(float(dw[i]), float(zw[i]), "local_max", score=-highness))
    # Add high quantile points as a safety net for flat/wide crests.
    order = np.argsort(zw)[::-1]
    for i in order[:max_candidates]:
        if all(abs(float(dw[i]) - c.d) > 3.0 for c in candidates):
            highness = (zw[i] - z10) / relief
            candidates.append(Candidate(float(dw[i]), float(zw[i]), "high_point", score=-highness))

    # Optional pybeach crest prediction. It is treated as one more candidate,
    # not as a replacement for the internal morphology-based candidates.
    candidates.extend(_pybeach_crest_candidates(d, z, search_lo, search_hi, **_))

    candidates = _deduplicate_candidates(candidates, min_distance_m=2.0)
    candidates = sorted(candidates, key=lambda c: (c.score, -c.z))[:max_candidates]
    return candidates


def _toe_candidates_for_crest(
    d: np.ndarray,
    z: np.ndarray,
    z_smooth: np.ndarray,
    crest: Candidate,
    search_hi: float,
    *,
    min_toe_crest_gap_m: float = 8.0,
    max_toe_from_crest_m: float = 160.0,
    beach_window_m: float = 50.0,
    max_candidates_per_method: int = 5,
    **kwargs: Any,
) -> list[Candidate]:
    lo = crest.d + float(min_toe_crest_gap_m)
    hi = min(search_hi, crest.d + float(max_toe_from_crest_m))
    m = (d >= lo) & (d <= hi)
    if m.sum() < 4:
        return []
    dw, zw = d[m], z_smooth[m]
    candidates: list[Candidate] = []

    # 1) Curvature: large curvature seaward of the crest, especially where the
    # profile starts to flatten into the beach.
    slope = np.gradient(z_smooth, d, edge_order=1)
    curv = np.gradient(slope, d, edge_order=1)
    cm = curv[m]
    if cm.size:
        scale = _percentile_scale(cm, 1.0)
        order = np.argsort(np.abs(cm) / scale)[::-1][:max_candidates_per_method]
        for ii in order:
            x = float(dw[ii])
            candidates.append(Candidate(x, _interp(d, z, x), "curvature", score=-float(abs(cm[ii]) / scale)))

    # 2) Relative relief: toe commonly sits at the lower part of the local dune
    # relief where the profile begins its sustained climb to the crest.
    for hw in (15.0, 25.0, 40.0):
        rr = _relative_relief(d, z_smooth, hw)
        rm = rr[m]
        if rm.size:
            target = 0.25
            good = np.where(np.isfinite(rm))[0]
            if good.size:
                order = good[np.argsort(np.abs(rm[good] - target))[:max_candidates_per_method]]
                for ii in order:
                    x = float(dw[ii])
                    candidates.append(Candidate(x, _interp(d, z, x), f"relative_relief_{int(hw)}m", score=float(abs(rm[ii] - target))))

    # 3) Perpendicular distance relative to crest--seaward-end line.
    for x in _perpendicular_distance_candidates(d, z_smooth, crest.d, hi, max_candidates=max_candidates_per_method):
        candidates.append(Candidate(x, _interp(d, z, x), "perpendicular_distance", score=0.0))

    # 4) Breakpoint scan: evaluates how well a candidate separates the dune face
    # from the fronting beach used later by the erosion model.
    stride = max(1, int(round(3.0 / max(_spacing(d), 1e-6))))
    scan_idx = np.where(m)[0][::stride]
    bp: list[tuple[float, float, float]] = []
    for idx in scan_idx:
        x = float(d[idx])
        face_d, face_z = _window_points(d, z, crest.d, x)
        beach_d, beach_z = _window_points(d, z, x, min(x + float(beach_window_m), d[-1]))
        if face_d.size < 4 or beach_d.size < 4:
            continue
        face_slope, _, face_rmse, _ = _robust_line_fit(face_d, face_z)
        beach_slope, _, beach_rmse, _ = _robust_line_fit(beach_d, beach_z)
        if not np.isfinite(face_slope) or not np.isfinite(beach_slope):
            continue
        z_toe = _interp(d, z, x)
        relief = crest.z - z_toe
        # With d increasing seaward, the dune face usually slopes down seaward:
        # slope = dz/dd < 0.  Use magnitudes for separation.
        slope_sep = abs(face_slope) - abs(beach_slope)
        penalty = 0.0
        if relief < 0.5:
            penalty += (0.5 - relief) * 2.0
        if slope_sep < 0.005:
            penalty += (0.005 - slope_sep) * 50.0
        score = face_rmse + beach_rmse + penalty
        bp.append((score, x, z_toe))
    for score, x, z_toe in sorted(bp, key=lambda t: t[0])[:max_candidates_per_method]:
        candidates.append(Candidate(float(x), float(z_toe), "breakpoint", score=float(score)))

    # Optional pybeach candidates. Use a window that includes the current crest
    # and the seaward beach, then filter the result below so it remains valid for
    # this specific crest candidate.
    pb_lo = max(float(d[0]), crest.d - 40.0)
    candidates.extend(_pybeach_toe_candidates(d, z, pb_lo, search_hi, **kwargs))

    # Keep only candidates compatible with the current crest.
    candidates = [c for c in candidates if np.isfinite(c.d) and c.d >= lo and c.d <= hi]

    # Deduplicate while keeping source information compact.
    candidates = _deduplicate_candidates(candidates, min_distance_m=2.0)
    return candidates[: max_candidates_per_method * 4]


def _heel_candidates_for_crest(
    d: np.ndarray,
    z: np.ndarray,
    z_smooth: np.ndarray,
    crest: Candidate,
    search_lo: float,
    row: Mapping[str, Any],
    *,
    search_landward_col: str = "d_dune_landward_polygon",
    min_heel_crest_gap_m: float = 5.0,
    max_heel_from_crest_m: float = 180.0,
    max_candidates_per_method: int = 5,
    **_: Any,
) -> list[Candidate]:
    hi = crest.d - float(min_heel_crest_gap_m)
    lo = max(search_lo, crest.d - float(max_heel_from_crest_m))
    m = (d >= lo) & (d <= hi)
    candidates: list[Candidate] = []

    if m.sum() >= 3:
        dw, zw = d[m], z_smooth[m]
        mins_local = _local_extrema_indices(zw, mode="min")
        if mins_local.size:
            order = mins_local[np.argsort(zw[mins_local])[:max_candidates_per_method]]
            for ii in order:
                x = float(dw[ii])
                candidates.append(Candidate(x, _interp(d, z, x), "local_low", score=float(zw[ii])))

        # Relative relief on the landward side: low relative relief often marks
        # the transition from the back of the dune to the adjacent terrain.
        for hw in (15.0, 25.0, 40.0):
            rr = _relative_relief(d, z_smooth, hw)
            rm = rr[m]
            good = np.where(np.isfinite(rm))[0]
            if good.size:
                order = good[np.argsort(np.abs(rm[good] - 0.25))[:max_candidates_per_method]]
                for ii in order:
                    x = float(dw[ii])
                    candidates.append(Candidate(x, _interp(d, z, x), f"relative_relief_{int(hw)}m", score=float(abs(rm[ii] - 0.25))))

        # Back breakpoint: split the back of the dune from the crest ramp.
        stride = max(1, int(round(3.0 / max(_spacing(d), 1e-6))))
        scan_idx = np.where(m)[0][::stride]
        bp: list[tuple[float, float]] = []
        for idx in scan_idx:
            x = float(d[idx])
            back_d, back_z = _window_points(d, z, lo, x)
            ramp_d, ramp_z = _window_points(d, z, x, crest.d)
            if back_d.size < 3 or ramp_d.size < 4:
                continue
            _, _, back_rmse, _ = _robust_line_fit(back_d, back_z)
            ramp_slope, _, ramp_rmse, _ = _robust_line_fit(ramp_d, ramp_z)
            score = back_rmse + ramp_rmse
            if np.isfinite(ramp_slope):
                score -= min(abs(ramp_slope), 0.2) * 0.5
            bp.append((score, x))
        for score, x in sorted(bp, key=lambda t: t[0])[:max_candidates_per_method]:
            candidates.append(Candidate(float(x), _interp(d, z, x), "back_breakpoint", score=float(score)))

    # Polygon landward point is a prior/fallback, not a hard truth.
    poly_land = _finite_float(row.get(search_landward_col), np.nan)
    if np.isfinite(poly_land) and search_lo <= poly_land <= hi:
        candidates.append(Candidate(float(poly_land), _interp(d, z, poly_land), "polygon_fallback", score=0.5))

    # Conservative fallback: a fixed offset landward of the crest within the
    # available profile.  It keeps the model alive while reporting low heel
    # confidence.
    fallback = max(search_lo, crest.d - min(40.0, max(15.0, 0.35 * float(max_heel_from_crest_m))))
    if np.isfinite(fallback) and fallback < hi:
        candidates.append(Candidate(float(fallback), _interp(d, z, fallback), "width_fallback", score=1.0))

    candidates = [c for c in candidates if np.isfinite(c.d) and np.isfinite(c.z) and c.d < crest.d - min_heel_crest_gap_m]
    return _deduplicate_candidates(candidates, min_distance_m=2.0)[: max_candidates_per_method * 4]


def _deduplicate_candidates(candidates: list[Candidate], *, min_distance_m: float = 2.0) -> list[Candidate]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: (c.score, c.d))
    kept: list[Candidate] = []
    for cand in candidates:
        if not np.isfinite(cand.d) or not np.isfinite(cand.z):
            continue
        close = [k for k in kept if abs(k.d - cand.d) <= min_distance_m]
        if not close:
            kept.append(cand)
        else:
            k = close[0]
            # Keep the best location, but append source labels for diagnostics.
            if cand.source not in k.source:
                k.source = f"{k.source}+{cand.source}"
            k.score = min(k.score, cand.score)
    return sorted(kept, key=lambda c: (c.score, c.d))


# -----------------------------------------------------------------------------
# Geometry scoring
# -----------------------------------------------------------------------------
def _score_geometry(
    d: np.ndarray,
    z: np.ndarray,
    heel: Candidate,
    crest: Candidate,
    toe: Candidate,
    row: Mapping[str, Any],
    *,
    beach_window_m: float = 50.0,
    min_crest_toe_relief_m: float = 0.6,
    min_toe_crest_gap_m: float = 8.0,
    min_heel_crest_gap_m: float = 5.0,
    search_landward_col: str = "d_dune_landward_polygon",
    search_seaward_col: str = "d_dune_seaward_polygon",
    **_: Any,
) -> GeometryCandidate | None:
    if not (heel.d < crest.d < toe.d):
        return None
    if toe.d - crest.d < min_toe_crest_gap_m or crest.d - heel.d < min_heel_crest_gap_m:
        return None

    z_toe = _interp(d, z, toe.d)
    z_crest = _interp(d, z, crest.d)
    z_heel = _interp(d, z, heel.d)
    if not np.isfinite(z_toe) or not np.isfinite(z_crest) or not np.isfinite(z_heel):
        return None
    relief = z_crest - z_toe
    if relief < 0.15:
        return None

    face_d, face_z = _window_points(d, z, crest.d, toe.d)
    beach_d, beach_z = _window_points(d, z, toe.d, min(toe.d + beach_window_m, d[-1]))
    back_d, back_z = _window_points(d, z, heel.d, crest.d)
    face_slope, _, face_rmse, n_face = _robust_line_fit(face_d, face_z)
    beach_slope, _, beach_rmse, n_beach = _robust_line_fit(beach_d, beach_z)
    back_slope, _, back_rmse, n_back = _robust_line_fit(back_d, back_z)

    if n_face < 3 or n_beach < 3:
        return None

    tan_face = abs(face_slope) if np.isfinite(face_slope) else np.nan
    tan_beach = abs(beach_slope) if np.isfinite(beach_slope) else np.nan
    if not np.isfinite(tan_face) or not np.isfinite(tan_beach):
        return None

    # Physical penalties.  Keep them soft: real profiles are messy, but clear
    # violations should lose against coherent alternatives.
    relief_pen = max(0.0, min_crest_toe_relief_m - relief) * 2.5
    slope_sep = tan_face - tan_beach
    slope_pen = max(0.0, 0.010 - slope_sep) * 45.0
    beach_slope_pen = max(0.0, tan_beach - 0.18) * 4.0 + max(0.0, 0.003 - tan_beach) * 4.0
    face_slope_pen = max(0.0, tan_face - 0.75) * 2.0

    # Priors from polygon intersections.  They act as weak anchors only.
    poly_land = _finite_float(row.get(search_landward_col), np.nan)
    poly_sea = _finite_float(row.get(search_seaward_col), np.nan)
    poly_pen = 0.0
    if np.isfinite(poly_sea):
        poly_pen += min(abs(toe.d - poly_sea) / 75.0, 2.0) * 0.15
    if np.isfinite(poly_land):
        poly_pen += min(abs(heel.d - poly_land) / 75.0, 2.0) * 0.10

    # Candidate-source prior: breakpoint agreement is valuable for toe; local
    # maxima are valuable for crests; polygon/width heel fallbacks are less safe.
    source_pen = 0.0
    if "breakpoint" not in toe.source:
        source_pen += 0.12
    if "curvature" in toe.source:
        source_pen -= 0.05
    if "pybeach" in toe.source:
        source_pen -= 0.08
    if "pybeach_ml" in toe.source:
        source_pen -= 0.05
    if "local_max" not in crest.source:
        source_pen += 0.08
    if "pybeach_crest" in crest.source:
        source_pen -= 0.03
    if "fallback" in heel.source:
        source_pen += 0.20

    # Normalize RMSE by profile relief to avoid over-penalising energetic dunes.
    local = z[(d >= heel.d) & (d <= min(toe.d + beach_window_m, d[-1]))]
    relief_scale = max(float(np.nanpercentile(local, 95) - np.nanpercentile(local, 5)), 0.5) if local.size else 1.0
    fit_score = (face_rmse + beach_rmse + 0.55 * back_rmse) / relief_scale

    score = fit_score + relief_pen + slope_pen + beach_slope_pen + face_slope_pen + poly_pen + source_pen
    if not np.isfinite(score):
        return None
    # Confidence is intentionally conservative.  High confidence means the
    # geometry is good enough to drive a model, not merely that a point was found.
    confidence = float(1.0 / (1.0 + np.exp(2.8 * (score - 0.75))))
    confidence = max(0.0, min(1.0, confidence))
    components = {
        "fit_score": float(fit_score),
        "face_rmse": float(face_rmse),
        "beach_rmse": float(beach_rmse),
        "back_rmse": float(back_rmse),
        "relief": float(relief),
        "relief_penalty": float(relief_pen),
        "slope_separation": float(slope_sep),
        "slope_penalty": float(slope_pen),
        "polygon_penalty": float(poly_pen),
        "source_penalty": float(source_pen),
        "tan_beta_f": float(tan_beach),
        "tan_beta_D": float(tan_face),
        "tan_beta_back": float(abs(back_slope)) if np.isfinite(back_slope) else np.nan,
    }
    return GeometryCandidate(
        heel=Candidate(heel.d, z_heel, heel.source, heel.score),
        crest=Candidate(crest.d, z_crest, crest.source, crest.score),
        toe=Candidate(toe.d, z_toe, toe.source, toe.score),
        score=float(score),
        confidence=confidence,
        components=components,
        sources={"heel": heel.source, "crest": crest.source, "toe": toe.source},
    )


def detect_profile_geometry(row: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Detect heel, crest and toe for a single profile.

    ``detection_mode`` controls the backend:

    - ``ensemble``: internal geometry ensemble only.
    - ``ensemble_with_pybeach``: internal ensemble plus pybeach candidates.
    - ``pybeach_only``: pybeach toe/crest only, with internal heel fallback.
    """
    mode = str(kwargs.get("detection_mode", "ensemble_with_pybeach" if kwargs.get("use_pybeach", False) else "ensemble")).lower()
    if mode in {"pybeach", "pybeach_only", "pybeach-only"}:
        return _detect_profile_geometry_pybeach_only(row, **kwargs)
    if mode in {"ensemble", "internal", "internal_ensemble"}:
        kwargs = dict(kwargs)
        kwargs["use_pybeach"] = False
    elif mode in {"ensemble_with_pybeach", "ensemble+pybeach", "hybrid"}:
        kwargs = dict(kwargs)
        kwargs["use_pybeach"] = True

    d_col = kwargs.get("d_col", "d")
    z_col = kwargs.get("z_col", "z")
    d, z = _profile_arrays(row, d_col=d_col, z_col=z_col)
    profile_idx = row.get("profile_idx", getattr(row, "name", np.nan))
    if d.size < 12:
        raise ValueError("Profile has too few finite points for dune-geometry detection.")

    search_lo, search_hi, search_source = _search_window(row, d, **kwargs)
    dw, zw = _window_points(d, z, search_lo, search_hi)
    if dw.size < 12:
        raise ValueError(f"Search segment too short within [{search_lo:.3f}, {search_hi:.3f}].")

    z_smooth = _smooth_profile(z, d, median_window_m=kwargs.get("median_window_m", 5.0), mean_window_m=kwargs.get("sg_window_m", kwargs.get("mean_window_m", 11.0)))
    crests = _crest_candidates(d, z_smooth, search_lo, search_hi, **kwargs)
    if not crests:
        raise ValueError("No viable crest candidates found.")

    all_geoms: list[GeometryCandidate] = []
    for crest in crests:
        # Refine crest to the observed maximum in a local window.  This protects
        # freeboard estimates while preserving the candidate's source.
        refine_window = float(kwargs.get("crest_refine_window_m", 12.0))
        md = (d >= crest.d - refine_window) & (d <= crest.d + refine_window) & (d >= search_lo) & (d <= search_hi)
        if md.sum() >= 3:
            ii = np.where(md)[0][int(np.nanargmax(z[md]))]
            crest = Candidate(float(d[ii]), float(z[ii]), crest.source + "+local_freeboard", crest.score)

        toes = _toe_candidates_for_crest(d, z, z_smooth, crest, search_hi, **kwargs)
        heels = _heel_candidates_for_crest(d, z, z_smooth, crest, search_lo, row, **kwargs)
        if not toes or not heels:
            continue
        for toe in toes:
            for heel in heels:
                geom = _score_geometry(d, z, heel, crest, toe, row, **kwargs)
                if geom is not None:
                    all_geoms.append(geom)

    if not all_geoms:
        raise ValueError("No physically coherent heel-crest-toe geometry found.")

    best = min(all_geoms, key=lambda g: g.score)
    status = "ok" if best.confidence >= float(kwargs.get("ok_confidence_threshold", 0.45)) else "low_confidence"
    comp = best.components
    pybeach_enabled = bool(kwargs.get("use_pybeach", False))
    pybeach_available = _load_pybeach_profile_class() is not None
    pybeach_used = "pybeach" in best.toe.source or "pybeach" in best.crest.source
    return {
        "profile_idx": profile_idx,
        "detection_status": status,
        "detection_method": "ensemble_curvature_breakpoint_rr_pd_pybeach" if pybeach_enabled else "ensemble_curvature_breakpoint_rr_pd",
        "pybeach_enabled": pybeach_enabled,
        "pybeach_available": pybeach_available,
        "pybeach_used": pybeach_used,
        "detection_confidence": best.confidence,
        "geometry_score": best.score,
        "geometry_score_components": dict(comp),
        "detection_search_source": search_source,
        "d_detection_search_min": float(search_lo),
        "d_detection_search_max": float(search_hi),
        "d_toe_final": float(best.toe.d),
        "z_toe_final": float(best.toe.z),
        "d_crest": float(best.crest.d),
        "z_crest": float(best.crest.z),
        "d_heel_final": float(best.heel.d),
        "z_heel_final": float(best.heel.z),
        "toe_source": best.sources["toe"],
        "crest_source": best.sources["crest"],
        "heel_source": best.sources["heel"],
        "toe_confidence": _feature_confidence(best, "toe"),
        "crest_confidence": _feature_confidence(best, "crest"),
        "heel_confidence": _feature_confidence(best, "heel"),
        "tan_beta_f_detection": comp.get("tan_beta_f", np.nan),
        "tan_beta_D_detection": comp.get("tan_beta_D", np.nan),
        "tan_beta_back_detection": comp.get("tan_beta_back", np.nan),
        "face_rmse": comp.get("face_rmse", np.nan),
        "beach_rmse": comp.get("beach_rmse", np.nan),
        "back_rmse": comp.get("back_rmse", np.nan),
        "dune_relief_m": comp.get("relief", np.nan),
        "slope_separation": comp.get("slope_separation", np.nan),
        "n_geometry_candidates": len(all_geoms),
        "error": np.nan,
    }


def _feature_confidence(best: GeometryCandidate, feature: str) -> float:
    base = best.confidence
    src = best.sources.get(feature, "")
    bonus = 0.0
    if "+" in src:
        bonus += 0.08
    if feature == "toe" and "breakpoint" in src:
        bonus += 0.08
    if feature == "toe" and "pybeach" in src:
        bonus += 0.06
    if feature == "crest" and "local" in src:
        bonus += 0.08
    if feature == "crest" and "pybeach" in src:
        bonus += 0.04
    if feature == "heel" and "fallback" in src:
        bonus -= 0.18
    return float(max(0.0, min(1.0, base + bonus)))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def apply_to_dataset(
    gdf: pd.DataFrame,
    *,
    show_progress: bool = True,
    progress_desc: str = "Detecting dune geometry",
    **kwargs: Any,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    iterator = _progress(gdf.iterrows(), show=show_progress, total=len(gdf), desc=progress_desc, unit="profile")
    for profile_idx, row in iterator:
        try:
            rec = detect_profile_geometry(row, **kwargs)
            rec["profile_idx"] = profile_idx
        except Exception as exc:  # keep batch workflows alive
            rec = {
                "profile_idx": profile_idx,
                "detection_status": "failed",
                "detection_method": str(kwargs.get("detection_mode", "ensemble_with_pybeach" if bool(kwargs.get("use_pybeach", False)) else "ensemble")),
                "pybeach_enabled": bool(kwargs.get("use_pybeach", False)),
                "pybeach_available": _load_pybeach_profile_class() is not None,
                "pybeach_used": False,
                "pybeach_only": str(kwargs.get("detection_mode", "")).lower() in {"pybeach", "pybeach_only", "pybeach-only"},
                "detection_confidence": 0.0,
                "geometry_score": np.nan,
                "geometry_score_components": {},
                "detection_search_source": "",
                "d_detection_search_min": np.nan,
                "d_detection_search_max": np.nan,
                "d_toe_final": np.nan,
                "z_toe_final": np.nan,
                "d_crest": np.nan,
                "z_crest": np.nan,
                "d_heel_final": np.nan,
                "z_heel_final": np.nan,
                "toe_source": "",
                "crest_source": "",
                "heel_source": "",
                "toe_confidence": 0.0,
                "crest_confidence": 0.0,
                "heel_confidence": 0.0,
                "tan_beta_f_detection": np.nan,
                "tan_beta_D_detection": np.nan,
                "tan_beta_back_detection": np.nan,
                "face_rmse": np.nan,
                "beach_rmse": np.nan,
                "back_rmse": np.nan,
                "dune_relief_m": np.nan,
                "slope_separation": np.nan,
                "n_geometry_candidates": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            # Best-effort search-window diagnostics even on failure.
            try:
                d, _ = _profile_arrays(row, d_col=kwargs.get("d_col", "d"), z_col=kwargs.get("z_col", "z"))
                if d.size:
                    lo, hi, src = _search_window(row, d, **kwargs)
                    rec.update({"d_detection_search_min": lo, "d_detection_search_max": hi, "detection_search_source": src})
            except Exception:
                pass
        records.append(rec)
    return pd.DataFrame.from_records(records)


def smooth_longitudinal_feature(
    results: pd.DataFrame,
    *,
    gdf: pd.DataFrame | None = None,
    feature_col: str,
    profile_col: str = "profile_idx",
    out_col: str | None = None,
    beach_col: str = "Playa",
    outlier_window: int = 9,
    xy_k: float = 2.5,
    xy_abs_dev_m: float = 12.0,
    smooth_window: int = 9,
    smooth_polyorder: int | None = None,
    min_valid_per_beach: int = 4,
    show_progress: bool = False,
    progress_desc: str = "Smoothing feature by beach",
    **_: Any,
) -> pd.DataFrame:
    """Robust alongshore smoothing by beach.

    The smoother is deliberately conservative: if a beach has too few finite
    detections, the raw values are retained.  Outliers are replaced by local
    median values before a rolling median/mean pass.
    """
    out_col = out_col or f"{feature_col}_smooth"
    if results.empty:
        results[out_col] = np.nan
        return results
    out = results.copy()
    out[out_col] = pd.to_numeric(out.get(feature_col, np.nan), errors="coerce")

    if gdf is None or beach_col not in getattr(gdf, "columns", []):
        groups = [("all", out.index.to_numpy())]
    else:
        meta = gdf[[beach_col]].copy()
        meta[profile_col] = meta.index
        beach_map = meta.drop_duplicates(profile_col).set_index(profile_col)[beach_col].to_dict()
        out["__smooth_beach__"] = out[profile_col].map(beach_map).fillna("unknown")
        groups = [(name, idx.to_numpy()) for name, idx in out.groupby("__smooth_beach__", dropna=False).groups.items()]

    for _, idx in _progress(groups, show=show_progress, total=len(groups), desc=progress_desc, unit="beach"):
        vals = pd.to_numeric(out.loc[idx, feature_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).sum() < min_valid_per_beach:
            continue
        # Keep the input order as alongshore order.  In this workflow profiles
        # are already ordered consistently by id/profile index within each beach.
        ser = pd.Series(vals)
        local_med = ser.rolling(max(3, int(outlier_window) | 1), center=True, min_periods=1).median().to_numpy(dtype=float)
        resid = vals - local_med
        mad = np.nanmedian(np.abs(resid[np.isfinite(resid)] - np.nanmedian(resid[np.isfinite(resid)]))) if np.isfinite(resid).any() else np.nan
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(resid)
        cleaned = vals.copy()
        if np.isfinite(sigma) and sigma > 0:
            bad = np.abs(resid) > max(float(xy_abs_dev_m), float(xy_k) * sigma)
            cleaned[bad] = local_med[bad]
        else:
            bad = np.abs(resid) > float(xy_abs_dev_m)
            cleaned[bad] = local_med[bad]
        # Fill short gaps before smoothing.  Long all-NaN sections remain NaN.
        cleaned_ser = pd.Series(cleaned).interpolate(limit_direction="both")
        smoothed = cleaned_ser.rolling(max(3, int(smooth_window) | 1), center=True, min_periods=1).median()
        smoothed = smoothed.rolling(max(3, int(smooth_window) | 1), center=True, min_periods=1).mean().to_numpy(dtype=float)
        smoothed[~np.isfinite(vals)] = np.nan
        out.loc[idx, out_col] = smoothed

    if "__smooth_beach__" in out.columns:
        out = out.drop(columns="__smooth_beach__")
    return out


def plot_profile_diagnostics(
    gdf: pd.DataFrame,
    profile_idx: Any,
    *,
    d_col: str = "d",
    z_col: str = "z",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    **kwargs: Any,
):
    """Plot the detected geometry and main search window for one profile."""
    row = gdf.loc[profile_idx]
    d, z = _profile_arrays(row, d_col=d_col, z_col=z_col)
    rec: dict[str, Any]
    try:
        rec = detect_profile_geometry(row, d_col=d_col, z_col=z_col, **kwargs)
    except Exception as exc:
        rec = {"error": f"{type(exc).__name__}: {exc}"}
        if d.size:
            lo, hi, src = _search_window(row, d, **kwargs)
            rec.update({"d_detection_search_min": lo, "d_detection_search_max": hi, "detection_search_source": src})

    z_smooth = _smooth_profile(z, d, median_window_m=kwargs.get("median_window_m", 5.0), mean_window_m=kwargs.get("sg_window_m", kwargs.get("mean_window_m", 11.0))) if d.size else z
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (11, 5)))
    ax.plot(d, z, lw=1.0, alpha=0.65, label="profile")
    if d.size:
        ax.plot(d, z_smooth, lw=1.5, label="smoothed")
    lo = rec.get("d_detection_search_min", np.nan)
    hi = rec.get("d_detection_search_max", np.nan)
    if np.isfinite(lo) and np.isfinite(hi):
        ax.axvspan(lo, hi, color="0.8", alpha=0.25, label="search window")
    features = [
        ("d_heel_final", "z_heel_final", "heel", "s"),
        ("d_crest", "z_crest", "crest", "^"),
        ("d_toe_final", "z_toe_final", "toe", "o"),
    ]
    for dc, zc, label, marker in features:
        x = _finite_float(rec.get(dc), np.nan)
        y = _finite_float(rec.get(zc), np.nan)
        if np.isfinite(x):
            if not np.isfinite(y):
                y = _interp(d, z, x)
            ax.scatter([x], [y], s=70, marker=marker, label=f"{label}: {rec.get(label + '_source', '')}")
            ax.axvline(x, lw=0.8, alpha=0.4)
    title = f"Profile {profile_idx} | {rec.get('detection_status', 'failed')} | confidence={_finite_float(rec.get('detection_confidence'), 0):.2f}"
    if rec.get("error") and not pd.isna(rec.get("error")):
        title += f" | {rec.get('error')}"
    ax.set_title(title)
    ax.set_xlabel("d [m]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.25)
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif np.isfinite(lo) and np.isfinite(hi):
        pad = max((hi - lo) * 0.10, 10.0)
        ax.set_xlim(lo - pad, hi + pad)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig, ax, rec
