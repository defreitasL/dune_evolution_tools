"""Diagnostics utilities.

Currently includes a mass-closure check for the 1D dune-volume balance:

    dV/dt = - qD

where:
- V is dune volume per unit alongshore width [m^3/m]
- qD is total erosion/transport rate leaving the dune cross-section [m^3/m/s]

This helper is meant for *sanity checks* in examples and notebooks.
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np


def _trapz_cum(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration.

    Parameters
    ----------
    y : (n,) array
    x : (n,) array

    Returns
    -------
    cum : (n,) array
        cum[i] = integral_{x[0]}^{x[i]} y dx
    """
    n = x.size
    out = np.zeros(n, dtype=float)
    for i in range(1, n):
        dx = x[i] - x[i-1]
        out[i] = out[i-1] + 0.5 * (y[i] + y[i-1]) * dx
    return out


def check_mass_closure(
    result: Dict[str, np.ndarray],
    *,
    volume_key: str = "V",
    flux_key: str = "qD",
    time_key: str = "time_s",
    baseline: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Check mass closure for the dune-volume balance.

    The model core is designed around:
        dV/dt = -qD

    This function computes:
        V_pred(t) = V0 - ∫ qD dt
        residual(t) = V(t) - V_pred(t)

    Parameters
    ----------
    result : dict
        Output dict from the model.
    volume_key : str
        Key for volume time series [m^3/m].
    flux_key : str
        Key for erosion flux time series [m^3/m/s].
    time_key : str
        Key for time vector [s].
    baseline : float, optional
        If provided, uses this as V0 instead of V[0]. Useful for comparing against
        an external initial volume definition.

    Returns
    -------
    out : dict
        Contains arrays and summary metrics:
        - time_s, time_h
        - V, V_pred, residual
        - int_qD (cumulative integral of qD)
        - max_abs_residual, rms_residual, rel_max_abs_residual
    """
    t = np.asarray(result[time_key], dtype=float)
    V = np.asarray(result[volume_key], dtype=float)
    qD = np.asarray(result[flux_key], dtype=float)

    if t.ndim != 1:
        raise ValueError(f"{time_key} must be 1D")
    if V.shape != t.shape or qD.shape != t.shape:
        raise ValueError("time, V and qD must have the same shape")

    # integrate qD over time
    int_qD = _trapz_cum(qD, t)  # [m^3/m]
    V0 = float(V[0] if baseline is None else baseline)
    V_pred = V0 - int_qD
    residual = V - V_pred

    max_abs = float(np.nanmax(np.abs(residual)))
    rms = float(np.sqrt(np.nanmean(residual * residual)))
    denom = max(1e-12, float(np.nanmax(np.abs(V))))
    rel_max = max_abs / denom

    return {
        "time_s": t,
        "time_h": t / 3600.0,
        "V": V,
        "qD": qD,
        "int_qD": int_qD,
        "V_pred": V_pred,
        "residual": residual,
        "max_abs_residual": np.array([max_abs], dtype=float),
        "rms_residual": np.array([rms], dtype=float),
        "rel_max_abs_residual": np.array([rel_max], dtype=float),
    }
