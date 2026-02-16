"""Example: dune erosion simulation driven by Total Water Level (TWL).

This example is analogous to ``example_storm_run.py`` but uses a user-provided
Total Water Level series (TWL) as the driver.

For demonstration we *construct*:
    TWL(t) = SWL(t) + Ru_stockdon(t)
where SWL includes an idealized tide + surge. The model, however, receives only
TWL(t).

Important
---------
- TWL must be in the SAME vertical datum as z0_init and Ds.
- The model uses exceedances (TWL - z0) and the overwash threshold (TWL > Ds).

Outputs
-------
- profiles_over_time_twl.png
- positions_timeseries_twl.png
- transports_timeseries_twl.png
- volume_timeseries_twl.png
- dune_width_timeseries_twl.png
- avalanche_diagnostics_twl.png
- dune_width_twl.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from dune_evolution_tools import DuneToeStormParams, DuneToeStormModel
from dune_evolution_tools.runup import runup_stockdon_r2
from dune_evolution_tools.diagnostics import check_mass_closure
from dune_evolution_tools.plotting import (
    plot_profiles_over_time,
    plot_positions,
    plot_transports,
    plot_volume_timeseries,
    plot_dune_width,
    plot_avalanche_diagnostics,
    plot_dune_width
)


def synthetic_storm_waves(time_s: np.ndarray):
    """Simple synthetic storm wave time series."""
    t_h = time_s / 3600.0
    bump = np.sin(np.clip((t_h / t_h.max()) * np.pi, 0.0, np.pi)) ** 2
    H0 = 0.5 + 5.0 * bump
    Tp = 8.0 + 10.0 * bump
    return H0, Tp


def synthetic_swl(time_s: np.ndarray):
    """Idealized still-water level: tide + storm surge."""
    t_h = time_s / 3600.0
    # Semi-diurnal tide (~12.42 h) plus a small diurnal component
    tide = 1.2 * np.sin(2.0 * np.pi * t_h / 12.42) + 0.2 * np.sin(2.0 * np.pi * t_h / 24.0)
    # Surge scales with storm intensity (smooth bump over the event)
    bump = np.sin(np.clip((t_h / t_h.max()) * np.pi, 0.0, np.pi)) ** 2
    surge = 0.6 * bump
    return tide + surge


def main():
    out_dir = os.path.dirname(__file__) or "."
    out_dir = os.path.join(out_dir, "twl_out")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 3-day storm at 1-min resolution
    time_s = np.arange(0, 24 * 3600 * 3 + 1, 60.0)

    H0, Tp = synthetic_storm_waves(time_s)
    SWL = synthetic_swl(time_s)

    # Build TWL(t) for this example (the model will only receive TWL)
    Ru = runup_stockdon_r2(H0=H0, T=Tp, tan_beta_f=0.10)
    TWL = SWL + Ru

    params = DuneToeStormParams(
        Ds=3.0,
        z0_init=0.5,
        tan_beta_f=0.10,
        # --- Cs option ---
        Cs_mode="larson2004_eq37",
        Cs=1.8e-3,
        D50=0.00025,
        H0_is_Hs=True,
        # overwash partition
        A_overwash=3.0,
        # repose angle (relative to horizontal)
        alpha_rep_deg=37.0,
        # mesh + instantaneous avalanching
        use_profile_mesh=True,
        dx_mesh=0.25,
        seaward_buffer_m=2,
        landward_crest_width_m=10.0,

        z_back=1.5,
        landward_back_slope_m=5.0,
        landward_back_buffer_m=5.0,
        # storm-only crest lowering
        crest_mode="moving",
        crest_erosion=True,
        crest_width_m=10.0, # Controls the intensity of storm-only crest lowering
    )

    model = DuneToeStormModel(params)

    # TWL-driven simulation (T is still required by the erosion-rate formulation)
    res = model.simulate_from_twl(time_s=time_s, TWL=TWL, T=Tp, H0_for_Cs=H0)

    # Plots
    plot_profiles_over_time(model, res, n_profiles=8, savepath=os.path.join(out_dir, "profiles_over_time_twl.png"))
    plot_positions(res, savepath=os.path.join(out_dir, "positions_timeseries_twl.png"))
    plot_transports(res, savepath=os.path.join(out_dir, "transports_timeseries_twl.png"))
    plot_volume_timeseries(res, savepath=os.path.join(out_dir, "volume_timeseries_twl.png"))
    plot_dune_width(res, savepath=os.path.join(out_dir, "dune_width_timeseries_twl.png"))
    plot_avalanche_diagnostics(res, savepath=os.path.join(out_dir, "avalanche_diagnostics_twl.png"))
    plot_dune_width(res, show_crest_width=False, savepath=os.path.join(out_dir, "dune_width_twl.png"))

    # Optional: plot TWL input for reference
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_s / 3600.0, TWL, label="TWL", linewidth=1.6)
    ax.plot(time_s / 3600.0, SWL, label="SWL", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Water level (m)")
    ax.set_title("Forcing (example): TWL = SWL + Ru")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "twl_timeseries.png"), dpi=160)

    # Mass-closure sanity check: V(t) ?= V0 - ∫ qD dt
    clo = check_mass_closure(res)
    print("Mass closure:")
    print(f" - max_abs_residual = {float(clo['max_abs_residual'][0]):.6g} m^3/m")
    print(f" - rms_residual     = {float(clo['rms_residual'][0]):.6g} m^3/m")
    print(f" - rel_max_abs      = {float(clo['rel_max_abs_residual'][0])*100.0:.4g} % of max|V|")

    print("Saved figures in examples/:")
    print(" - profiles_over_time_twl.png")
    print(" - positions_timeseries_twl.png")
    print(" - transports_timeseries_twl.png")
    print(" - volume_timeseries_twl.png")
    print(" - dune_width_timeseries_twl.png")
    print(" - avalanche_diagnostics_twl.png")
    print(" - twl_timeseries.png")


if __name__ == "__main__":
    main()
