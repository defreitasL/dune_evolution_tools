"""Parameter containers for the dune toe storm model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CsMode = Literal["constant", "larson2004_eq37"]
CrestMode = Literal["fixed", "moving"]


@dataclass(frozen=True)
class DuneToeStormParams:
    """Model parameters.

    Notes
    -----
        - alpha_rep_deg is the *repose angle with respect to the horizontal* (DRT convention).
      The limiting dune-face slope used by the mesh avalanching is tan(alpha_rep_deg).
    - If `Cs_mode="larson2004_eq37"`, you must provide a deep-water wave-height series:
      - via `simulate_from_waves(..., H0=...)`, or
      - via `simulate_from_runup(..., H0_for_Cs=...)`
      If `H0_is_Hs=True`, we interpret H0 as Hs and convert to Hrms by Hrms = Hs/sqrt(2).
      If `H0_is_Hs=False`, H0 is assumed to already be Hrms.
    - Mesh-based avalanching diagnostics are enabled with `use_profile_mesh=True`.
      In that case you can choose `crest_mode`:
        * "fixed"  : crest x-position is fixed at its initial value.
        * "moving" : crest x-position follows xd(t) = x0(t) + (Ds - z0(t))/tan(alpha_rep).
    - Optional storm-only crest lowering can be enabled with `crest_erosion=True`.
      In that case, the dune crest elevation Ds(t) is allowed to *decrease only* (no recovery),
      driven by the landward/overwash component qL:
          dDs/dt = -(k_crest / crest_width_m) * qL   if Ru > Ds
      This is a pragmatic storm-scale closure that preserves the event focus of the model.
    """

    # --- geometry / state ---
    Ds: float
    z0_init: float
    tan_beta_f: float

    # --- overwash partition (Larson 2016) ---
    A_overwash: float = 3.0

    # --- erosion intensity Cs ---
    Cs: float = 1.8e-3
    Cs_mode: CsMode = "constant"

    # Larson (2004) Eq. (37): Cs = A * exp(-b * Hrms0/D50)
    Cs_eq37_A: float = 1.34e-3
    Cs_eq37_b: float = 3.19e-4
    H0_is_Hs: bool = True  # if True, Hrms = Hs/sqrt(2); else Hrms = H0

    # --- repose angle (wrt horizontal) ---
    alpha_rep_deg: float = 32.0

    # --- sediment ---
    D50: float = 0.0002  # [m]

    # --- numerics (toe ODE) ---
    s_min: float = 1e-3

    # --- optional profile mesh + instantaneous avalanching (DRT-style) ---
    use_profile_mesh: bool = False
    dx_mesh: float = 0.25
    seaward_buffer_m: float = 5.0
    landward_crest_width_m: float = 3.0
    # --- optional landward/backdune representation (for plots/mesh boundaries) ---
    # After the crest/plateau section (width landward_crest_width_m), the profile can
    # descend landward with slope tan_beta_back over landward_back_slope_m, then remain
    # flat at z_back for landward_back_buffer_m.
    tan_beta_back: float = -1.0  # if <=0, uses tan_beta_f
    z_back: float = 0.0
    landward_back_slope_m: float = 4.0
    landward_back_buffer_m: float = 2.0

    avalanche_max_iters: int = 100
    crest_mode: CrestMode = "fixed"  # only used when use_profile_mesh=True

    # --- optional storm-only crest lowering (no recovery) ---
    crest_erosion: bool = False
    crest_width_m: float = 10.0  # [m] effective alongshore-averaged width of the active crest-lowering zone
    k_crest: float = 1.0         # [-] fraction of qL that produces crest lowering (0..1)
