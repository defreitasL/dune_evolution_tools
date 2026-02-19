"""Example: dune erosion simulation.

Demonstrates:
- Stockdon runup from H0,T
- Cs computed from Larson (2004) Eq. (37) (optional)

Outputs:
- profiles_over_time.png
- positions_timeseries.png
- transports_timeseries.png
- cs_timeseries.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from dune_evolution_tools import DuneToeStormParams, DuneToeStormModel
from dune_evolution_tools.diagnostics import check_mass_closure
from dune_evolution_tools.plotting import plot_profiles_over_time, plot_positions, plot_transports, plot_volume_timeseries, save_profile_evolution_gif

def synthetic_storm(time_s: np.ndarray):
    t_h = time_s / 3600.0
    bump = np.sin(np.clip((t_h / t_h.max()) * np.pi, 0.0, np.pi)) ** 2
    H0 = 0.5 + 4.0 * bump
    Tp = 8.0 + 10.0 * bump
    return H0, Tp

def main():
    out_dir = os.path.dirname(__file__) or "."
    out_dir = os.path.join(out_dir, "waves_out")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    time_s = np.arange(0, 24*3600 * 3 + 1, 60.0)  # 1-min

    H0, Tp = synthetic_storm(time_s)

    params = DuneToeStormParams(
        Ds=3.0,
        z0_init=0.5,
        tan_beta_f=0.10,
        # --- Cs option ---
        Cs_mode="larson2004_eq37",   # try "constant" if you want a fixed Cs
        Cs=1.8e-3,                   # used only if Cs_mode="constant"
        D50=0.00025,                 # [m] used by Eq. (37)
        H0_is_Hs=True,               # interpret H0 as Hs (convert to Hrms)
        # overwash partition
        A_overwash=3.0,
        # repose angle
        alpha_rep_deg=37.0,
            # mesh + instantaneous avalanching
        use_profile_mesh=True,
        dx_mesh=0.25,
        seaward_buffer_m=2,
        landward_crest_width_m=10.0,

        z_back=1.5,
        landward_back_slope_m=5.0,
        landward_back_buffer_m=2.0,
        # storm-only crest lowering
        crest_mode="moving",
        crest_erosion=True,
        crest_width_m=7.0, # Controls the intensity of storm-only crest lowering
    )
    model = DuneToeStormModel(params)

    res = model.simulate_from_waves(time_s=time_s, H0=H0, T=Tp, runup_mode="stockdon")

    plot_profiles_over_time(model, res, n_profiles=8, savepath=os.path.join(out_dir, "profiles_over_time.png"))
    plot_positions(res, savepath=os.path.join(out_dir, "positions_timeseries.png"))
    plot_transports(res, savepath=os.path.join(out_dir, "transports_timeseries.png"))
    plot_volume_timeseries(res, savepath=os.path.join(out_dir, "volume_timeseries.png"))

    # Save profile evolution as a GIF
    save_profile_evolution_gif(
        model,
        res, 
        out_gif=os.path.join(out_dir, "profile_evolution.gif"),
        water_level="auto",   # o "Ru_used" / "TWL_used"
        every=15,             # 1 = todos los pasos; 3 = 1 frame cada 3 timesteps
        fps=30,
        dpi=200,
        )

    # Cs time series
    fig, ax = plt.subplots()
    ax.plot(res["time_s"]/3600.0, res["Cs_used"])
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cs (-)")
    ax.set_title("Cs(t) used by the model")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cs_timeseries.png"), dpi=160)

    print("Saved figures in examples/:")
    print(" - profiles_over_time.png")
    print(" - positions_timeseries.png")
    print(" - transports_timeseries.png")
    print(" - cs_timeseries.png")
    print(" - volume_timeseries.png")

    # Mass-closure sanity check: V(t) ?= V0 - ∫ qD dt
    clo = check_mass_closure(res)
    print("Mass closure:")
    print(f" - max_abs_residual = {float(clo['max_abs_residual'][0]):.6g} m^3/m")
    print(f" - rms_residual     = {float(clo['rms_residual'][0]):.6g} m^3/m")
    print(f" - rel_max_abs      = {float(clo['rel_max_abs_residual'][0])*100.0:.4g} % of max|V|")

if __name__ == "__main__":
    main()
