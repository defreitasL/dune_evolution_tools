"""Example: dune erosion simulation (runup-driven).

Demonstrates:
- User-provided runup Ru(t) forcing (model uses Ru directly for dz0/dt)
- Cs computed from Larson (2004) Eq. (37) using an *input* Hrms(t) series

Inputs in this example:
- Ru(t): runup time series [m]
- Tp(t): peak period time series [s]
- Hrms(t): deep-water RMS wave height series [m] (for Cs Eq. 37)

Outputs:
- profiles_over_time.png
- positions_timeseries.png
- transports_timeseries.png
- cs_timeseries.png
- avalanche_diagnostics.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from dune_evolution_tools import DuneToeStormParams, DuneToeStormModel
from dune_evolution_tools.plotting import (
    plot_profiles_over_time,
    plot_positions,
    plot_transports,
    plot_avalanche_diagnostics,
)


def synthetic_storm(time_s: np.ndarray, seed: int = 0):
    """
    Build a simple synthetic storm with:
    - Ru(t): runup [m]
    - Tp(t): peak period [s]
    - Hrms(t): deep-water RMS wave height [m] used ONLY for Cs Eq.37

    Notes
    -----
    This is just a demo generator. In real applications you would feed:
    - Ru from a runup proxy (Stockdon, SWASH, observations, etc.)
    - Hrms from deep-water waves (or your preferred source) already as RMS.
    """
    rng = np.random.default_rng(seed)
    t_h = time_s / 3600.0
    bump = np.sin(np.clip((t_h / t_h.max()) * np.pi, 0.0, np.pi)) ** 2

    # Runup forcing (independent input)
    Ru = 0.3 + 2.2 * bump  # [m]
    Ru += 0.03 * rng.normal(size=Ru.size)  # small noise

    # Peak period
    Tp = 8.0 + 10.0 * bump  # [s]
    Tp += 0.10 * rng.normal(size=Tp.size)

    # Deep-water RMS wave height series for Cs Eq. 37 (independent input)
    Hrms = 0.4 + 1.8 * bump  # [m]
    Hrms += 0.02 * rng.normal(size=Hrms.size)

    # Guard negatives
    Ru = np.maximum(Ru, 0.0)
    Tp = np.maximum(Tp, 1.0)
    Hrms = np.maximum(Hrms, 0.0)

    return Ru, Tp, Hrms


def main():
    out_dir = os.path.dirname(__file__) or "."
    out_dir = os.path.join(out_dir, "runnup_out")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 72 h at 1-min resolution
    time_s = np.arange(0, 24 * 3600 * 3 + 1, 300.0)  # 5-min (keeps the demo fast)

    Ru, Tp, Hrms = synthetic_storm(time_s)

    params = DuneToeStormParams(
        Ds=3.0,
        z0_init=0.5,
        tan_beta_f=0.10,
        # --- Cs option ---
        Cs_mode="larson2004_eq37",   # try "constant" if you want fixed Cs
        Cs=1.8e-3,                   # used only if Cs_mode="constant"
        D50=0.00025,                 # [m] used by Eq. (37)
        H0_is_Hs=False,              # IMPORTANT: our provided H0_for_Cs is Hrms already
        # overwash partition
        A_overwash=3.0,
        # repose angle
        alpha_rep_deg=37.0,
        # --- optional profile mesh + instantaneous avalanching ---
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

    # Run the model using *Ru(t) directly*
    # Hrms is passed only for Cs(t) Eq.37
    res = model.simulate_from_runup(time_s=time_s, Ru=Ru, T=Tp, H0_for_Cs=Hrms)

    # Plots
    plot_profiles_over_time(model, res, n_profiles=8, savepath=os.path.join(out_dir, "profiles_over_time.png"))
    plot_positions(res, savepath=os.path.join(out_dir, "positions_timeseries.png"))
    plot_transports(res, savepath=os.path.join(out_dir, "transports_timeseries.png"))
    plot_avalanche_diagnostics(res, savepath=os.path.join(out_dir, "avalanche_diagnostics.png"))

    # Cs time series
    fig, ax = plt.subplots()
    ax.plot(res["time_s"] / 3600.0, res["Cs_used"])
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cs (-)")
    ax.set_title("Cs(t) used by the model (Larson 2004 Eq. 37 with input Hrms)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cs_timeseries.png"), dpi=160)
    plt.close(fig)

    print("Saved figures in examples/:")
    print(" - profiles_over_time.png")
    print(" - positions_timeseries.png")
    print(" - transports_timeseries.png")
    print(" - cs_timeseries.png")
    print(" - avalanche_diagnostics.png")

    # quick diagnostics
    x_face = res["xd"] - res["x0"]
    dt = np.diff(res["time_s"])
    IA = np.abs(np.diff(x_face) / dt)

    print("Ru range:", float(Ru.min()), float(Ru.max()))
    print("Hrms range:", float(Hrms.min()), float(Hrms.max()))
    print("z0 range:", float(res["z0"].min()), float(res["z0"].max()))
    print("Ds range:", float(res.get("Ds_ts", np.array([model.params.Ds])).min()), float(res.get("Ds_ts", np.array([model.params.Ds])).max()))
    print("x_face range:", float(x_face.min()), float(x_face.max()))
    print("IA_proxy max:", float(IA.max()), "m/s")


if __name__ == "__main__":
    main()
