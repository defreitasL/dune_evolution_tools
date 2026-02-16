"""Profile mesh + avalanching (instantaneous slope-limiter) utilities.

This module adds an optional 1D cross-shore grid representation of the dune profile
and applies an "instantaneous avalanching" relaxation identical in spirit to the
DRT MATLAB implementation (Cohn & Anderson 2025):

If local slope exceeds tan(alpha_rep), redistribute elevations between adjacent
nodes until all slopes satisfy |dz/dx| <= tan(alpha_rep).

Design goals
------------
- Keep the fast RK4+Numba toe evolution core unchanged.
- Add a mesh-based profile reconstruction ONLY for morphology/diagnostics/plots.
- Allow two crest options (only when mesh is enabled):
    * crest_mode="fixed"  : crest x-position is fixed at its initial value.
    * crest_mode="moving" : crest x-position follows the geometric relation:
          xd(t) = x0(t) + (Ds - z0(t)) / tan(alpha_rep)
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


@njit(cache=True, fastmath=True)
def _enforce_beachface(z: np.ndarray, x: np.ndarray, x0: float, z0: float, tan_beta_f: float):
    """Enforce a beachface line for x <= x0, passing through (x0,z0) with slope tan_beta_f."""
    if tan_beta_f <= 0.0:
        return
    x_swl = x0 - z0 / tan_beta_f  # location where z=0 along the beachface line
    for i in range(x.size):
        if x[i] <= x0:
            z[i] = tan_beta_f * (x[i] - x_swl)


@njit(cache=True, fastmath=True)
def _enforce_landward(
    z: np.ndarray,
    x: np.ndarray,
    xd: float,
    Ds: float,
    landward_crest_width_m: float,
    tan_beta_back: float,
    z_back: float,
    landward_back_slope_m: float,
):
    """Enforce a simple landward boundary profile: plateau + backdune slope + back-barrier flat.

    - Plateau: z = Ds for xd <= x < xd + landward_crest_width_m
    - Backdune slope: z decreases landward with slope tan_beta_back, limited to z_back
    - Flat: z = z_back once reached

    Notes
    -----
    If landward_back_slope_m <= 0, we revert to the previous behavior: z=Ds for x>=xd.
    """
    if landward_back_slope_m <= 0.0:
        for i in range(x.size):
            if x[i] >= xd:
                z[i] = Ds
        return

    xp = xd + landward_crest_width_m
    for i in range(x.size):
        if x[i] < xd:
            continue
        if x[i] < xp:
            z[i] = Ds
        else:
            zb = Ds - tan_beta_back * (x[i] - xp)
            if zb < z_back:
                zb = z_back
            z[i] = zb


@njit(cache=True, fastmath=True)
def _avalanche_sweep(z: np.ndarray, dx: float, tan_alpha: float, i0: int, i1: int) -> float:
    """One left-to-right sweep that relaxes slope exceedances.

    Parameters
    ----------
    i0, i1 : int
        Sweep over pairs (i, i+1) for i in [i0, i1).
        Typically, i0 is 0 or near the toe index; i1 is n-1.

    Returns
    -------
    max_excess : float
        Maximum slope excess (in tan-units) encountered during this sweep.
    """
    max_excess = 0.0
    if dx <= 0.0:
        return 0.0
    lim = tan_alpha * dx
    n = z.size
    if i0 < 0:
        i0 = 0
    if i1 > n - 1:
        i1 = n - 1

    for i in range(i0, i1):
        dz = z[i+1] - z[i]
        adz = dz if dz >= 0.0 else -dz
        if adz > lim:
            excess = adz - lim
            if excess > max_excess:
                max_excess = excess / dx
            # Move half the excess from the higher node to the lower node.
            # This conserves mean elevation of the pair.
            ddz = 0.5 * excess
            if dz > 0.0:
                z[i] += ddz
                z[i+1] -= ddz
            else:
                z[i] -= ddz
                z[i+1] += ddz
    return max_excess


@njit(cache=True, fastmath=True)
def avalanche_relax_instantaneous(
    z: np.ndarray,
    x: np.ndarray,
    x0: float,
    z0: float,
    xd: float,
    Ds: float,
    tan_beta_f: float,
    tan_alpha: float,
    landward_crest_width_m: float,
    tan_beta_back: float,
    z_back: float,
    landward_back_slope_m: float,
    dx: float,
    max_iters: int,
) -> float:
    """Apply instantaneous avalanching until slopes satisfy the repose criterion.

    We repeatedly sweep and re-enforce the beachface (seaward) and landward boundary (landward)
    boundary conditions. This matches the typical DRT approach: erosion updates first,
    then avalanching relaxation.

    Returns
    -------
    slope_excess_max : float
        Max slope exceedance (in tan-units, i.e., max(|dz/dx| - tan_alpha, 0)).
    """
    slope_excess_max = 0.0
    # Enforce boundaries before relax
    _enforce_beachface(z, x, x0, z0, tan_beta_f)
    _enforce_landward(z, x, xd, Ds, landward_crest_width_m, tan_beta_back, z_back, landward_back_slope_m)

    for _ in range(max_iters):
        # one sweep across full domain
        excess = _avalanche_sweep(z, dx, tan_alpha, 0, z.size - 1)
        if excess > slope_excess_max:
            slope_excess_max = excess
        # Re-enforce boundaries after sweep
        _enforce_beachface(z, x, x0, z0, tan_beta_f)
        _enforce_landward(z, x, xd, Ds, landward_crest_width_m, tan_beta_back, z_back, landward_back_slope_m)

        # early exit when stable
        if excess <= 1e-12:
            break

    # final slope excess computation
    slope_excess_max = 0.0
    for i in range(z.size - 1):
        s = (z[i+1] - z[i]) / dx
        if s < 0.0:
            s = -s
        excess = s - tan_alpha
        if excess > slope_excess_max:
            slope_excess_max = excess
    if slope_excess_max < 0.0:
        slope_excess_max = 0.0
    return slope_excess_max


def build_mesh_grid(
    Ds: float,
    z0_init: float,
    tan_beta_f: float,
    tan_alpha: float,
    seaward_buffer_m: float,
    landward_crest_width_m: float,
    landward_back_slope_m: float,
    landward_back_buffer_m: float,
    dx_mesh: float,
    crest_mode_int: int,
) -> tuple[np.ndarray, float]:
    """Build a fixed cross-shore grid that covers the foreshore + dune + plateau.

    Returns
    -------
    x : array
        Cross-shore coordinates [m]
    xd0 : float
        Initial crest x-position [m] (used if crest_mode="fixed")
    """
    # toe initial at x0=0 by convention
    x0 = 0.0
    if tan_beta_f <= 0.0:
        raise ValueError("tan_beta_f must be > 0")

    # dune face slope limit (repose)
    if tan_alpha <= 0.0:
        raise ValueError("tan(alpha_rep) must be > 0")

    # Initial crest position (geometric)
    xd0 = x0 + (Ds - z0_init) / tan_alpha

    # seaward extent: reach SWL (z=0) then buffer
    x_swl0 = x0 - z0_init / tan_beta_f
    x_min = x_swl0 - float(seaward_buffer_m)

    # landward extent: plateau after crest + optional backdune slope + buffer
    base_land = float(landward_crest_width_m) + float(landward_back_slope_m) + float(landward_back_buffer_m)
    if crest_mode_int == 0:  # fixed
        x_max = xd0 + base_land
    else:  # moving: allow a bit extra to accommodate retreat
        x_max = xd0 + base_land + 20.0

    n = int(np.ceil((x_max - x_min) / float(dx_mesh))) + 1
    x = x_min + np.arange(n, dtype=float) * float(dx_mesh)
    return x, xd0


@njit(cache=True, fastmath=True)
def _profile_init(z: np.ndarray, x: np.ndarray, x0: float, z0: float, xd: float, Ds: float, tan_beta_f: float, tan_alpha: float, landward_crest_width_m: float, tan_beta_back: float, z_back: float, landward_back_slope_m: float):
    """Initialize a piecewise-linear profile: beachface + dune face (repose) + plateau."""
    # beachface line passes through (x0,z0)
    x_swl = x0 - z0 / tan_beta_f
    for i in range(x.size):
        if x[i] <= x0:
            z[i] = tan_beta_f * (x[i] - x_swl)
        elif x[i] < xd:
            z[i] = z0 + tan_alpha * (x[i] - x0)
        else:
            # landward boundary: plateau + backdune slope + flat
            xp = xd + landward_crest_width_m
            if landward_back_slope_m <= 0.0:
                z[i] = Ds
            elif x[i] < xp:
                z[i] = Ds
            else:
                zb = Ds - tan_beta_back * (x[i] - xp)
                if zb < z_back:
                    zb = z_back
                z[i] = zb


@njit(cache=True, fastmath=True)
def simulate_profile_mesh(
    time_s: np.ndarray,
    x: np.ndarray,
    z0: np.ndarray,
    x0: np.ndarray,
    Ds_ts: np.ndarray,
    tan_beta_f: float,
    tan_alpha: float,
    landward_crest_width_m: float,
    tan_beta_back: float,
    z_back: float,
    landward_back_slope_m: float,
    crest_mode_int: int,
    xd0_fixed: float,
    dx_mesh: float,
    max_avalanche_iters: int,
):
    """Build a time-evolving profile on a fixed mesh and apply avalanching.

    Returns
    -------
    z_prof : (nt, nx) array
    V_mesh : (nt,) array
        Volume above the beachface baseline between toe and crest (m3/m).
    slope_excess_max : (nt,) array
        Max slope exceedance (tan-units) after relaxation.
    xd_ts : (nt,) array
        Crest x-position time series (fixed or moving).
    """
    nt = time_s.size
    nx = x.size
    z_prof = np.empty((nt, nx), dtype=np.float64)
    V_mesh = np.empty(nt, dtype=np.float64)
    slope_excess_max = np.empty(nt, dtype=np.float64)
    xd_ts = np.empty(nt, dtype=np.float64)

    # initial crest position
    Ds0 = Ds_ts[0]
    if crest_mode_int == 0:
        xd_ts[0] = xd0_fixed
    else:
        s0 = Ds0 - z0[0]
        xd_ts[0] = x0[0] + (s0 / tan_alpha) if s0 > 0.0 else x0[0]

    _profile_init(z_prof[0], x, x0[0], z0[0], xd_ts[0], Ds0, tan_beta_f, tan_alpha, landward_crest_width_m, tan_beta_back, z_back, landward_back_slope_m)

    dx = float(dx_mesh)
    if dx <= 0.0:
        dx = x[1] - x[0]

    # For each time step, start from previous profile, enforce toe/crest, avalanche
    for k in range(1, nt):
        Dsk = Ds_ts[k]
        # crest position
        if crest_mode_int == 0:
            xd = xd0_fixed
        else:
            s = Dsk - z0[k]
            xd = x0[k] + (s / tan_alpha) if s > 0.0 else x0[k]
        xd_ts[k] = xd

        # start from previous profile
        for i in range(nx):
            z_prof[k, i] = z_prof[k-1, i]

        # enforce current toe beachface and plateau
        _enforce_beachface(z_prof[k], x, x0[k], z0[k], tan_beta_f)
        _enforce_landward(z_prof[k], x, xd, Dsk, landward_crest_width_m, tan_beta_back, z_back, landward_back_slope_m)

        # relax
        slope_excess = avalanche_relax_instantaneous(
            z_prof[k], x,
            x0[k], z0[k],
            xd, Dsk,
            tan_beta_f,
            tan_alpha,
            landward_crest_width_m,
            tan_beta_back,
            z_back,
            landward_back_slope_m,
            dx,
            max_avalanche_iters,
        )
        slope_excess_max[k] = slope_excess

        # compute volume above beachface baseline between toe and crest (simple rectangle rule)
        # baseline beachface everywhere:
        x_swl = x0[k] - z0[k] / tan_beta_f
        vol = 0.0
        for i in range(nx):
            if x[i] < x0[k] or x[i] > xd:
                continue
            zb = tan_beta_f * (x[i] - x_swl)
            dz = z_prof[k, i] - zb
            if dz > 0.0:
                vol += dz * dx
        V_mesh[k] = vol

    # k=0 diagnostics
    slope_excess_max[0] = 0.0
    x_swl0 = x0[0] - z0[0] / tan_beta_f
    vol0 = 0.0
    for i in range(nx):
        if x[i] < x0[0] or x[i] > xd_ts[0]:
            continue
        zb = tan_beta_f * (x[i] - x_swl0)
        dz = z_prof[0, i] - zb
        if dz > 0.0:
            vol0 += dz * (x[1]-x[0])
    V_mesh[0] = vol0

    return z_prof, V_mesh, slope_excess_max, xd_ts
