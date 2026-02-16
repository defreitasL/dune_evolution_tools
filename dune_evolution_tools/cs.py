"""Empirical options for the transport coefficient Cs.

Larson et al. (2004) discuss variability of Cs and propose an empirical trend
as a function of deep-water RMS wave height and grain size (Eq. 37).

Eq. (37) (Larson et al., 2004):
    Cs = A * exp( -b * (Hrms0 / D50) )

with A = 1.34e-3 and b = 3.19e-4, and validity (as stated) roughly for:
    0.15 mm < D50 < 0.50 mm

Important:
- Use consistent units: Hrms0 [m] and D50 [m] so the ratio is dimensionless.
"""
from __future__ import annotations
import numpy as np

def cs_larson2004_eq37(Hrms0: np.ndarray, D50_m: float, A: float = 1.34e-3, b: float = 3.19e-4) -> np.ndarray:
    Hrms0 = np.asarray(Hrms0, dtype=float)
    D50_m = float(D50_m)
    if D50_m <= 0.0:
        raise ValueError("D50 must be > 0 for Eq. (37)")
    ratio = Hrms0 / D50_m
    ratio = np.maximum(0.0, ratio)
    return A * np.exp(-b * ratio)
