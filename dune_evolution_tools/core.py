"""Numba-accelerated core with RK4 integration.

Core equations:
- Larson (2004) Eqs. 13–15 + u_s^2 = 2 g Ru (Eq. 18)
- Larson (2016) overwash-modified qD and partitioning with alpha (Eqs. 3–4)
- Larson (2004) slope substitution (Eq. 21) to impose dune-face slope geometrically (repose angle wrt horizontal)
"""
from __future__ import annotations
import numpy as np

try:
    from numba import njit as _njit
    def njit(*args, **kwargs):
        return _njit(*args, **kwargs)
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco
    NUMBA_AVAILABLE = False

G = 9.81
DEG2RAD = np.pi / 180.0

@njit(cache=True, fastmath=True)
def dune_face_tan_from_repose(alpha_rep_deg: float) -> float:
    # alpha_rep_deg is the repose angle w.r.t. the horizontal (DRT convention)
    beta_D = alpha_rep_deg * DEG2RAD
    if beta_D > (0.5*np.pi - 1e-6):
        beta_D = 0.5*np.pi - 1e-6
    if beta_D < 1e-6:
        beta_D = 1e-6
    return np.tan(beta_D)

@njit(cache=True, fastmath=True)
def tan_beta_effective(tan_beta_R: float, tan_beta_D: float) -> float:
    eps = 1e-12
    if tan_beta_R <= eps:
        return tan_beta_R
    if tan_beta_D <= tan_beta_R + eps:
        tan_beta_D = tan_beta_R + 10.0*eps
    return 1.0 / (1.0/tan_beta_R - 1.0/tan_beta_D)

@njit(cache=True, fastmath=True)
def qd_impact(Ru: float, T: float, Ds: float, z0: float, Cs: float) -> float:
    if T <= 0.0:
        return 0.0
    if Ru <= z0:
        return 0.0
    s = Ds - z0
    if s <= 0.0:
        return 0.0
    if Ru <= Ds:
        dz = Ru - z0
        return 4.0 * Cs * dz * dz / T
    else:
        dz = Ru - z0
        return 4.0 * Cs * dz * s / T

@njit(cache=True, fastmath=True)
def alpha_overwash(Ru: float, Ds: float, z0: float, A: float) -> float:
    if A <= 0.0:
        return 0.0
    if Ru <= Ds:
        return 0.0
    s = Ds - z0
    if s <= 0.0:
        return 0.0
    a = (1.0 / A) * ((Ru - z0) / s - 1.0)
    if a < 0.0:
        a = 0.0
    return a

@njit(cache=True, fastmath=True)
def dune_volume(Ds: float, z0: float, tan_beta_eff: float) -> float:
    s = Ds - z0
    if s <= 0.0 or tan_beta_eff <= 0.0:
        return 0.0
    return 0.5 * s * s / tan_beta_eff

@njit(cache=True, fastmath=True)
def dz0dt(Ru: float, T: float, Ds: float, z0: float, Cs: float, tan_beta_eff: float, s_min: float) -> float:
    s = Ds - z0
    if s <= s_min:
        return 0.0
    qD = qd_impact(Ru, T, Ds, z0, Cs)
    return qD * tan_beta_eff / s

@njit(cache=True, fastmath=True)
def rk4_step(z: float, Ru0: float, T0: float, Cs0: float, Ru1: float, T1: float, Cs1: float, dt: float,
             Ds: float, tan_beta_eff: float, s_min: float) -> float:
    Rum = 0.5 * (Ru0 + Ru1)
    Tm  = 0.5 * (T0 + T1)
    Csm = 0.5 * (Cs0 + Cs1)

    k1 = dz0dt(Ru0, T0, Ds, z, Cs0, tan_beta_eff, s_min)
    k2 = dz0dt(Rum, Tm, Ds, z + 0.5*dt*k1, Csm, tan_beta_eff, s_min)
    k3 = dz0dt(Rum, Tm, Ds, z + 0.5*dt*k2, Csm, tan_beta_eff, s_min)
    k4 = dz0dt(Ru1, T1, Ds, z + dt*k3, Cs1, tan_beta_eff, s_min)

    z_new = z + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
    if z_new > Ds - s_min:
        z_new = Ds - s_min
    return z_new


@njit(cache=True, fastmath=True)
def _state_derivs_with_crest(Ru: float, T: float, Cs: float, Ds: float, z0: float,
                            tan_beta_eff: float, s_min: float,
                            A_overwash: float, k_crest: float, crest_width_m: float):
    """Compute coupled derivatives for storm-only crest lowering.

    We enforce mass consistency for the geometric volume:
        V = 0.5 * (Ds - z0)^2 / tan_beta_eff
    so that:
        dV/dt = -qD
    while allowing Ds(t) to decrease due to overwash:
        dDs/dt = -(k_crest/crest_width_m) * qL   if Ru > Ds
    where qL is the landward/overwash fraction of the total erosion flux qD.

    Returns
    -------
    dz0dt, dDsdt, qD, qS, qL
    """
    qk = qd_impact(Ru, T, Ds, z0, Cs)
    ak = alpha_overwash(Ru, Ds, z0, A_overwash)

    qsk = qk / (1.0 + ak) if (1.0 + ak) > 0.0 else qk
    qlk = qk - qsk

    dDsdt = 0.0
    if crest_width_m > 0.0 and k_crest > 0.0 and ak > 0.0:
        dDsdt = -(k_crest / crest_width_m) * qlk

    s = Ds - z0
    if s < s_min:
        s = s_min

    # Mass-consistent relation for the geometric wedge:
    #   (s/tan_beta_eff) * (dDsdt - dz0dt) = -qD
    # => dz0dt = dDsdt + qD * tan_beta_eff / s
    dz0 = dDsdt + (qk * tan_beta_eff / s)

    return dz0, dDsdt, qk, qsk, qlk


@njit(cache=True, fastmath=True)
def rk4_step_coupled(z0: float, Ds: float,
                     Ru0: float, T0: float, Cs0: float,
                     Ru1: float, T1: float, Cs1: float,
                     dt: float,
                     tan_beta_eff: float, s_min: float,
                     A_overwash: float, k_crest: float, crest_width_m: float):
    """RK4 step for the coupled system (z0, Ds)."""
    Rum = 0.5 * (Ru0 + Ru1)
    Tm  = 0.5 * (T0 + T1)
    Csm = 0.5 * (Cs0 + Cs1)

    k1_z, k1_Ds, _, _, _ = _state_derivs_with_crest(Ru0, T0, Cs0, Ds, z0, tan_beta_eff, s_min, A_overwash, k_crest, crest_width_m)
    k2_z, k2_Ds, _, _, _ = _state_derivs_with_crest(Rum, Tm, Csm, Ds + 0.5*dt*k1_Ds, z0 + 0.5*dt*k1_z, tan_beta_eff, s_min, A_overwash, k_crest, crest_width_m)
    k3_z, k3_Ds, _, _, _ = _state_derivs_with_crest(Rum, Tm, Csm, Ds + 0.5*dt*k2_Ds, z0 + 0.5*dt*k2_z, tan_beta_eff, s_min, A_overwash, k_crest, crest_width_m)
    k4_z, k4_Ds, _, _, _ = _state_derivs_with_crest(Ru1, T1, Cs1, Ds + dt*k3_Ds, z0 + dt*k3_z, tan_beta_eff, s_min, A_overwash, k_crest, crest_width_m)

    z0_new = z0 + (dt/6.0)*(k1_z + 2.0*k2_z + 2.0*k3_z + k4_z)
    Ds_new = Ds + (dt/6.0)*(k1_Ds + 2.0*k2_Ds + 2.0*k3_Ds + k4_Ds)

    # enforce physical bounds
    if Ds_new < 0.0:
        Ds_new = 0.0
    if z0_new < 0.0:
        z0_new = 0.0

    # enforce minimum dune height s_min
    if Ds_new < z0_new + s_min:
        Ds_new = z0_new + s_min
    if z0_new > Ds_new - s_min:
        z0_new = Ds_new - s_min

    return z0_new, Ds_new


@njit(cache=True, fastmath=True)
def simulate_core_rk4_crest(time_s: np.ndarray, Ru: np.ndarray, T: np.ndarray, Cs_t: np.ndarray,
                            Ds_init: float, z0_init: float, tan_beta_f: float,
                            A_overwash: float, alpha_rep_deg: float, s_min: float,
                            k_crest: float, crest_width_m: float):
    """Core simulation with storm-only crest lowering Ds(t).

    This keeps the same erosion/partitioning structure as `simulate_core_rk4`,
    but integrates both z0(t) and Ds(t) consistently with the geometric wedge volume.
    """
    n = time_s.size
    z0 = np.empty(n, dtype=np.float64)
    Ds_ts = np.empty(n, dtype=np.float64)
    x0 = np.empty(n, dtype=np.float64)
    zd = np.empty(n, dtype=np.float64)
    xd = np.empty(n, dtype=np.float64)
    V  = np.empty(n, dtype=np.float64)
    dVdt = np.empty(n, dtype=np.float64)
    qD = np.empty(n, dtype=np.float64)
    qS = np.empty(n, dtype=np.float64)
    qL = np.empty(n, dtype=np.float64)
    dDsdt = np.empty(n, dtype=np.float64)

    tan_beta_D = dune_face_tan_from_repose(alpha_rep_deg)
    tan_beta_eff = tan_beta_effective(tan_beta_f, tan_beta_D)

    z0[0] = z0_init
    Ds_ts[0] = Ds_init
    z0_ref = z0_init

    for k in range(n):
        zk = z0[k]
        Dsk = Ds_ts[k]

        dzk, dDsk, qk, qsk, qlk = _state_derivs_with_crest(
            Ru[k], T[k], Cs_t[k], Dsk, zk, tan_beta_eff, s_min, A_overwash, k_crest, crest_width_m
        )

        qD[k] = qk
        qS[k] = qsk
        qL[k] = qlk
        dDsdt[k] = dDsk

        V[k] = dune_volume(Dsk, zk, tan_beta_eff)
        dVdt[k] = -qk

        zd[k] = Dsk

        if tan_beta_f > 0.0:
            x0[k] = (zk - z0_ref) / tan_beta_f
        else:
            x0[k] = np.nan

        if tan_beta_D > 0.0:
            xd[k] = x0[k] + (Dsk - zk) / tan_beta_D
        else:
            xd[k] = np.nan

        if k < n - 1:
            dt = time_s[k+1] - time_s[k]
            if dt < 0.0:
                dt = 0.0
            z0[k+1], Ds_ts[k+1] = rk4_step_coupled(
                zk, Dsk,
                Ru[k], T[k], Cs_t[k],
                Ru[k+1], T[k+1], Cs_t[k+1],
                dt,
                tan_beta_eff, s_min,
                A_overwash, k_crest, crest_width_m
            )

    return z0, Ds_ts, x0, zd, xd, V, dVdt, dDsdt, qD, qS, qL, tan_beta_D, tan_beta_eff


@njit(cache=True, fastmath=True)
def simulate_core_rk4(time_s: np.ndarray, Ru: np.ndarray, T: np.ndarray, Cs_t: np.ndarray,
                      Ds: float, z0_init: float, tan_beta_f: float,
                      A_overwash: float, alpha_rep_deg: float, s_min: float):
    n = time_s.size
    z0 = np.empty(n, dtype=np.float64)
    x0 = np.empty(n, dtype=np.float64)
    zd = np.empty(n, dtype=np.float64)
    xd = np.empty(n, dtype=np.float64)
    V  = np.empty(n, dtype=np.float64)
    dVdt = np.empty(n, dtype=np.float64)
    qD = np.empty(n, dtype=np.float64)
    qS = np.empty(n, dtype=np.float64)
    qL = np.empty(n, dtype=np.float64)

    tan_beta_D = dune_face_tan_from_repose(alpha_rep_deg)
    tan_beta_eff = tan_beta_effective(tan_beta_f, tan_beta_D)

    z0[0] = z0_init
    z0_ref = z0_init

    for k in range(n):
        zk = z0[k]
        qk = qd_impact(Ru[k], T[k], Ds, zk, Cs_t[k])
        ak = alpha_overwash(Ru[k], Ds, zk, A_overwash)

        qsk = qk / (1.0 + ak) if (1.0 + ak) > 0.0 else qk
        qlk = qk - qsk

        qD[k] = qk
        qS[k] = qsk
        qL[k] = qlk

        V[k] = dune_volume(Ds, zk, tan_beta_eff)
        dVdt[k] = -qk

        zd[k] = Ds

        if tan_beta_f > 0.0:
            x0[k] = (zk - z0_ref) / tan_beta_f
        else:
            x0[k] = np.nan

        if tan_beta_D > 0.0:
            xd[k] = x0[k] + (Ds - zk) / tan_beta_D
        else:
            xd[k] = np.nan

        if k < n - 1:
            dt = time_s[k+1] - time_s[k]
            if dt < 0.0:
                dt = 0.0
            z0[k+1] = rk4_step(zk, Ru[k], T[k], Cs_t[k], Ru[k+1], T[k+1], Cs_t[k+1], dt, Ds, tan_beta_eff, s_min)

    return z0, x0, zd, xd, V, dVdt, qD, qS, qL, tan_beta_D, tan_beta_eff
