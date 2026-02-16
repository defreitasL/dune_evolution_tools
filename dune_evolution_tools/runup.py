"""Runup utilities (Stockdon et al. 2006 R2%)."""
from __future__ import annotations
import numpy as np

G = 9.81
TWO_PI = 2.0 * np.pi

def deepwater_wavelength(T: np.ndarray, g: float = G) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    return g * T * T / TWO_PI

def runup_stockdon_r2(H0: np.ndarray, T: np.ndarray, tan_beta_f: float) -> np.ndarray:
    H0 = np.asarray(H0, dtype=float)
    T = np.asarray(T, dtype=float)
    beta = float(tan_beta_f)

    L0 = deepwater_wavelength(T)
    HL = np.maximum(0.0, H0) * np.maximum(0.0, L0)
    sqrtHL = np.sqrt(HL)

    term1 = 0.35 * beta * sqrtHL
    term2 = 0.5 * np.sqrt(HL * (0.563 * beta * beta + 0.004))
    return 1.1 * (term1 + term2)
