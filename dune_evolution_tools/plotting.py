"""Lightweight plotting helpers (matplotlib only)."""
from __future__ import annotations
from typing import Dict, Optional, Union, Literal, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .model import DuneToeStormModel
from .diagnostics import check_mass_closure

def plot_profiles_over_time(model: DuneToeStormModel, result: Dict[str, np.ndarray], n_profiles: int = 8,
                            seaward_buffer_m: float = 50.0, landward_crest_width_m: float = 30.0,
                            savepath: Optional[str] = None):
    """Plot dune profiles at selected times.

    If the simulation produced a profile mesh (keys "x_prof" and "z_prof"), those are used.
    Otherwise we fall back to the simple geometric reconstruction.
    """
    t = result["time_s"]
    idx = np.linspace(0, len(t) - 1, n_profiles).astype(int)

    fig, ax = plt.subplots(figsize=(10, 4))

    if "x_prof" in result and "z_prof" in result:
        x = result["x_prof"]
        z_prof = result["z_prof"]
        for i in idx:
            ax.plot(x, z_prof[i], label=f"t={t[i]/3600:.1f} h")
        ax.set_title("Dune profile evolution (mesh + instantaneous avalanching)")
    else:
        z0 = result["z0"]
        x0 = result["x0"]
        tan_beta_D = float(result["tan_beta_D"][0])
        for i in idx:
            Ds_i = float(result.get("Ds_ts", np.array([model.params.Ds]))[i])
            x, z = model.build_profile_xy(
                float(z0[i]), float(x0[i]), Ds_i,
                float(model.params.tan_beta_f), tan_beta_D,
                seaward_buffer_m=seaward_buffer_m,
                landward_crest_width_m=landward_crest_width_m,
                landward_back_slope_m=float(getattr(model.params, "landward_back_slope_m", 40.0)),
                landward_back_buffer_m=float(getattr(model.params, "landward_back_buffer_m", 20.0)),
                z_back=float(getattr(model.params, "z_back", 0.0)),
                tan_beta_back=float(getattr(model.params, "tan_beta_back", -1.0)),
            )
            ax.plot(x, z, label=f"t={t[i]/3600:.1f} h")
        ax.set_title("Dune profile evolution (geometric repose slope)")

    ax.set_xlabel("Cross-shore x (m)")
    ax.set_ylabel("Elevation z (m)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160)
    return fig



def save_profile_evolution_gif(
    model: DuneToeStormModel,
    result: Dict[str, np.ndarray],
    out_gif: str,
    water_level: Union[Literal["auto"], str] = "auto",
    every: int = 1,
    fps: int = 12,
    dpi: int = 140,
    seaward_buffer_m: float = 50.0,
    landward_crest_width_m: float = 30.0,
    landward_back_slope_m: float = 40.0,
    landward_back_buffer_m: float = 20.0,
    z_back: float = 0.0,
    tan_beta_back: float = -1.0,
    fill_base: Union[Literal["auto"], float] = "auto",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Dune profile evolution",
):
    """Create and save a GIF of the evolving dune/beach profile."""
    if every < 1:
        raise ValueError("every must be >= 1")
    time_s = np.asarray(result["time_s"], dtype=float)

    # --- pick water-level series ---
    if water_level == "auto":
        wl_key = "TWL_used" if "TWL_used" in result else "Ru_used"
    else:
        wl_key = str(water_level)
    if wl_key not in result:
        raise KeyError(f"water_level={water_level!r} -> key {wl_key!r} not found in result")
    wl = np.asarray(result[wl_key], dtype=float)

    # --- profile source ---
    use_mesh = ("x_prof" in result) and ("z_prof" in result)
    if use_mesh:
        x_mesh = np.asarray(result["x_prof"], dtype=float)
        z_mesh = np.asarray(result["z_prof"], dtype=float)  # (time, x)
        if z_mesh.shape[0] != time_s.size:
            raise ValueError("z_prof must have shape (time, x)")
    else:
        z0 = np.asarray(result["z0"], dtype=float)
        x0 = np.asarray(result["x0"], dtype=float)
        Ds_ts = np.asarray(result.get("Ds_ts", np.full_like(z0, float(model.params.Ds))), dtype=float)
        tan_beta_D = float(np.asarray(result.get("tan_beta_D", np.array([np.tan(np.deg2rad(34.0))])))[0])
        tan_beta_f = float(model.params.tan_beta_f)

    # --- axis limits (GLOBAL, no white space in x) ---
    if xlim is None or ylim is None:
        if use_mesh:
            xmin = float(np.nanmin(x_mesh))
            xmax = float(np.nanmax(x_mesh))
            zmin = float(np.nanmin(z_mesh))
            zmax = float(np.nanmax(z_mesh))
        else:
            tan_beta_f = float(model.params.tan_beta_f)
            xmin = float(np.nanmin(x0 - z0 / tan_beta_f - seaward_buffer_m))
            face_w = (Ds_ts - z0) / tan_beta_D
            xmax = float(np.nanmax(x0 + face_w + landward_crest_width_m + landward_back_slope_m + landward_back_buffer_m))
            zmin = float(-tan_beta_f * seaward_buffer_m)
            zmin = min(zmin, float(z_back))
            zmax = float(np.nanmax(Ds_ts))

        zmin = min(zmin, float(np.nanmin(wl)))
        zmax = max(zmax, float(np.nanmax(wl)))

        # >>> CHANGE: no x padding (to avoid white space)
        if xlim is None:
            xlim = (xmin, xmax)

        if ylim is None:
            ypad = 0.08 * (zmax - zmin) if zmax > zmin else 1.0
            ylim = (zmin - ypad, zmax + ypad)

    # --- fill baseline ---
    fill_base_val = float(ylim[0]) if fill_base == "auto" else float(fill_base)

    # --- frame indices ---
    idx = np.arange(0, time_s.size, every, dtype=int)
    if idx.size < 2:
        raise ValueError("Not enough frames after applying 'every'")

    # Ensure output dir exists
    out_gif = str(out_gif)
    out_dir = os.path.dirname(out_gif)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --- build figure ---
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Cross-shore x (m)")
    ax.set_ylabel("Elevation z (m)")
    ax.set_title(title)

    # Build initial profile (pre-storm) once
    if use_mesh:
        x_init = x_mesh
        z_init = z_mesh[idx[0]]
    else:
        Ds0 = float(Ds_ts[idx[0]])
        x_init, z_init = model.build_profile_xy(
            float(z0[idx[0]]), float(x0[idx[0]]), Ds0,
            float(model.params.tan_beta_f), tan_beta_D,
            seaward_buffer_m=seaward_buffer_m,
            landward_crest_width_m=landward_crest_width_m,
            landward_back_slope_m=landward_back_slope_m,
            landward_back_buffer_m=landward_back_buffer_m,
            z_back=z_back,
            tan_beta_back=tan_beta_back,
        )

    # >>> CHANGE: pre-storm profile persistent dashed light-gray line
    pre_line, = ax.plot(
        x_init, z_init,
        linestyle="--",
        color="lightgray",
        linewidth=2.0,
        zorder=2,
        label="Pre-storm"
    )

    # Sand fill + evolving profile + water level
    sand = ax.fill_between(x_init, z_init, fill_base_val, color="wheat", alpha=1, zorder=1)
    prof_line, = ax.plot(x_init, z_init, color="black", linewidth=2.2, zorder=3, label="Profile", zorder=2)

    # >>> CHANGE: legend label for blue line as TWL
    wl_line, = ax.plot(
        [xlim[0], xlim[1]],
        [wl[idx[0]], wl[idx[0]]],
        color="tab:blue",
        linewidth=2.2,
        zorder=0,
        label="TWL"
    )

    # Legend (includes TWL + others; if you want ONLY TWL, I can simplify)
    ax.legend(loc="upper right", frameon=True)

    # Time label on the top (axes coordinates)
    time_text = ax.text(
        0.01, 1.02, "",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=11,
    )

    def _frame_profile(k: int):
        ii = idx[k]
        if use_mesh:
            return x_mesh, z_mesh[ii]
        Ds_i = float(Ds_ts[ii])
        x_i, z_i = model.build_profile_xy(
            float(z0[ii]), float(x0[ii]), Ds_i,
            float(model.params.tan_beta_f), tan_beta_D,
            seaward_buffer_m=seaward_buffer_m,
            landward_crest_width_m=landward_crest_width_m,
            landward_back_slope_m=landward_back_slope_m,
            landward_back_buffer_m=landward_back_buffer_m,
            z_back=z_back,
            tan_beta_back=tan_beta_back,
        )
        return x_i, z_i

    def update(k: int):
        nonlocal sand
        ii = idx[k]
        x_i, z_i = _frame_profile(k)

        # update evolving profile line
        prof_line.set_data(x_i, z_i)

        # update water line
        ywl = float(wl[ii])
        wl_line.set_data([xlim[0], xlim[1]], [ywl, ywl])

        # update sand fill (re-create, robust and simple)
        sand.remove()
        sand = ax.fill_between(x_i, z_i, fill_base_val, color="gold", alpha=0.35, zorder=1)

        # update time text
        time_text.set_text(f"t = {time_s[ii]/3600.0:.2f} h")

        return prof_line, wl_line, sand, time_text, pre_line

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=idx.size,
        interval=1000.0 / float(fps),
        blit=False,
    )

    writer = animation.PillowWriter(fps=int(fps))
    ani.save(out_gif, writer=writer, dpi=int(dpi))
    plt.close(fig)
    return out_gif

def plot_positions(result: Dict[str, np.ndarray], savepath: Optional[str] = None):
    th = result["time_s"] / 3600.0
    fig, ax = plt.subplots()
    ax.plot(th, result["z0"], label="z0 (toe elev)", color="black", linewidth=1.8)
    ax.plot(th, result["zd"], label="zd (crest elev)", color="tab:gray", linewidth=1.0)
    ax.plot(th, result["x0"], label="x0 (toe x)")
    ax.plot(th, result["xd"], label="xd (crest x)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Value (m)")
    ax.set_title("Positions")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160)
    return fig


def plot_dune_width(result: Dict[str, np.ndarray], savepath: Optional[str] = None):
    """Plot dune width time series.

    The model defines the cross-shore dune width as the distance between the dune toe
    and the dune crest:

        W(t) = xd(t) - x0(t)

    When a profile mesh is enabled, this width is consistent with the chosen crest_mode
    (fixed or moving). Otherwise it corresponds to the geometric reconstruction.
    """
    th = np.asarray(result["time_s"], dtype=float) / 3600.0
    if "x_face" in result:
        W = np.asarray(result["x_face"], dtype=float)
    else:
        W = np.asarray(result["xd"], dtype=float) - np.asarray(result["x0"], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(th, W, color="black", linewidth=1.8)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Dune width W = xd - x0 (m)")
    ax.set_title("Dune width time series")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160)
    return fig

# def plot_transports(result: Dict[str, np.ndarray], savepath: Optional[str] = None):
#     th = result["time_s"] / 3600.0
#     fig, host_ax = plt.subplots(figsize=(10, 5))

#     components = [
#         ("qD", "qD (erosion total)", "tab:blue", "-"),
#         ("qS", "qS (offshore)", "tab:orange", "--"),
#         ("qL", "qL (landward/overwash)", "tab:green", "--"),
#     ]

#     # Create one right-side y-axis per transport component.
#     axes = [host_ax] + [host_ax.twinx() for _ in range(len(components) - 1)]
#     # Position extra axes in axes coordinates so they stay to the right and
#     # do not collapse the data area.
#     offsets = [1.00, 1.12, 1.24, 1.36]

#     for ax, (_, label, color, _), offset in zip(axes, components, offsets):
#         ax.patch.set_visible(False)
#         ax.spines["top"].set_visible(False)
#         ax.spines["left"].set_visible(False)
#         ax.spines["bottom"].set_visible(False)
#         ax.yaxis.set_label_position("right")
#         ax.yaxis.tick_right()
#         ax.spines["right"].set_position(("axes", offset))
#         ax.tick_params(axis="y", colors=color)
#         ax.set_ylabel(f"{label}\n(m$^3$/m/s)", color=color, rotation=90, va="center")

#     lines = []
#     for ax, (key, label, color, ls) in zip(axes, components):
#         line, = ax.plot(th, result[key], color=color, linewidth=1.8, label=label, linestyle=ls)
#         lines.append(line)

#     host_ax.set_xlabel("Time (hours)")
#     host_ax.set_title("Transports")
#     host_ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", frameon=False)

#     # Reserve room on the right for 4 y-axes labels/ticks.
#     fig.subplots_adjust(left=0.08, right=0.68, bottom=0.14, top=0.90)
#     if savepath:
#         fig.savefig(savepath, dpi=160)
#     return fig


def plot_transports(result: Dict[str, np.ndarray], savepath: Optional[str] = None):
    th = result["time_s"] / 3600.0
    fig, ax = plt.subplots(figsize=(10, 5))

    components = [
        ("qD", "qD (erosion total)", "tab:blue", "-"),
        ("qS", "qS (offshore)", "tab:orange", "--"),
        ("qL", "qL (landward/overwash)", "tab:green", "--"),
    ]


    ax.set_ylabel(f"Transport components\n(m$^3$/m/s)", rotation=90, va="center")

    lines = []
    for (key, label, color, ls) in components:
        line, = ax.plot(th, result[key], color=color, linewidth=1.8, label=label, linestyle=ls)
        lines.append(line)

    ax.set_xlabel("Time (hours)")
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper center", frameon=False, ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.15))

    if savepath:
        fig.savefig(savepath, dpi=160)
    return fig

def plot_avalanche_diagnostics(result: Dict[str, np.ndarray], savepath: Optional[str] = None):
    """Plot diagnostics to quantify the effect of repose enforcement.

    This is diagnostic-only and does not represent an external sediment transport term.

    Panels
    ------
    (1) Volume difference + face-width change rate
        - black: V - V_vert  (uses DeltaV_mesh if available, else DeltaV_geom)
        - red  : IA_proxy = |d(x_face)/dt| in mm/h (right axis)

    (2) Mesh avalanching diagnostic (only if mesh was enabled)
        - gray: mesh_slope_excess_max converted to degrees above repose
    """
    th = result["time_s"] / 3600.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0]})

    ax = axes[0]

    # Prefer mesh-based delta volume if available
    if "DeltaV_mesh" in result:
        ax.plot(th, result["DeltaV_mesh"], color="black", linewidth=1.8, label="V_mesh - V_vert")
        ax.set_ylabel(r"$V_{mesh} - V_{vert}$ (m$^3$/m)", color="black")
    elif "DeltaV_geom" in result:
        ax.plot(th, result["DeltaV_geom"], color="black", linewidth=1.8, label="V - V_vert")
        ax.set_ylabel(r"$V - V_{vert}$ (m$^3$/m)", color="black")

    if "IA_proxy" in result:
        ax2 = ax.twinx()
        ax2.plot(th, result["IA_proxy"] * 3600.0 * 1000.0, color="red", linewidth=1.8,
                 label="IA_proxy (mm/h)")
        ax2.tick_params(axis="y", colors="red")
        ax2.set_ylabel("IA_proxy (mm/h)", color="red")

    ax.set_title("Geometry / repose diagnostics")
    ax.grid(False)

    # --- slope excess (mesh only) ---
    axb = axes[1]
    if "mesh_slope_excess_max" in result:
        # stored in tan-units: max(|dz/dx| - tan(alpha), 0)
        excess_tan = np.asarray(result["mesh_slope_excess_max"], dtype=float)
        alpha = float(result.get("alpha_rep_deg", np.array([np.nan]))[0])
        if np.isfinite(alpha):
            tan_alpha = np.tan(np.deg2rad(alpha))
            # Convert to degrees above repose
            deg_excess = np.rad2deg(np.arctan(tan_alpha + excess_tan)) - alpha
            deg_excess = np.maximum(deg_excess, 0.0)
            axb.plot(th, deg_excess, color="tab:gray", linewidth=1.6)
            axb.set_ylabel(r"Max slope excess (deg)")
        else:
            axb.plot(th, excess_tan, color="tab:gray", linewidth=1.6)
            axb.set_ylabel(r"Max slope excess (tan)")
        axb.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
        axb.set_title("Mesh avalanching diagnostic")
    else:
        axb.text(0.02, 0.6, "No mesh avalanching output\n(use_profile_mesh=False)",
                 transform=axb.transAxes)

    axb.set_xlabel("Time (hours)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160)
    return fig


def plot_volume_timeseries(
    result: Dict[str, np.ndarray],
    *,
    volume_key: str = "V",
    show_mass_closure: bool = True,
    savepath: Optional[str] = None,
):
    """Plot dune volume time series V(t).

    Optionally overlays the predicted volume from mass closure:
        V_pred(t) = V0 - ∫ qD dt

    Parameters
    ----------
    result : dict
        Model output dict.
    volume_key : str
        Key for volume time series.
    show_mass_closure : bool
        If True, also plots V_pred(t) computed from qD.
    savepath : str, optional
        If provided, saves the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    th = np.asarray(result["time_s"], dtype=float) / 3600.0
    V = np.asarray(result[volume_key], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(th, V, label=f"{volume_key}(t)", color="black", linewidth=1.8)

    if show_mass_closure:
        clo = check_mass_closure(result, volume_key=volume_key)
        ax.plot(th, clo["V_pred"], label="V_pred (V0 - ∫qD dt)", linestyle="--", linewidth=1.4)

        # small annotation (no clutter)
        rel = float(clo["rel_max_abs_residual"][0]) * 100.0
        ax.text(
            0.02, 0.02,
            f"max |V - V_pred| / max|V| = {rel:.3g} %",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
        )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Volume (m$^3$/m)")
    ax.set_title("Dune volume time series")
    ax.legend(frameon=False)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160)
    return fig


def plot_dune_width(
    res: dict,
    savepath: str | None = None,
    # --- crest-width detection (mesh mode) ---
    crest_tol_m: float = 0.02,          # z >= Ds - crest_tol_m defines "crest plateau"
    # --- if no mesh is available ---
    crest_width_fallback_m: float | None = None,
    show_crest_width: bool = True,
):
    """
    Plot:
      - Dune width: W_dune(t) = xd - x0  (or x_face if available)
      - Crest width: W_crest(t) estimated from mesh profile plateau, if available

    Parameters
    ----------
    res : dict
        Simulation result dictionary.
        Expected keys for dune width:
            - "time_s", "x0", "xd" (and optionally "x_face")
        For crest width from mesh (recommended):
            - "x_prof" (nx,), "z_prof" (nt,nx), and "Ds_ts" (nt,) or "Ds" scalar.
    crest_tol_m : float
        Tolerance below Ds to classify nodes as belonging to the crest plateau.
    crest_width_fallback_m : float | None
        Used only if mesh is not present. If None and no mesh, crest width is not plotted.
    show_crest_width : bool
        Toggle crest width plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    t_h = np.asarray(res["time_s"], dtype=float) / 3600.0

    # --- dune width ---
    if "x_face" in res:
        W_dune = np.asarray(res["x_face"], dtype=float)
    else:
        W_dune = np.asarray(res["xd"], dtype=float) - np.asarray(res["x0"], dtype=float)

    fig, ax1 = plt.subplots()
    ax1.plot(t_h, W_dune, label="Dune width  W = xd - x0")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Dune width (m)")

    # --- crest width ---
    if show_crest_width:
        W_crest = None

        has_mesh = ("x_prof" in res) and ("z_prof" in res)
        if has_mesh:
            x = np.asarray(res["x_prof"], dtype=float)           # (nx,)
            z = np.asarray(res["z_prof"], dtype=float)           # (nt, nx)

            if "Ds_ts" in res:
                Ds_ts = np.asarray(res["Ds_ts"], dtype=float)    # (nt,)
            elif "Ds" in res:
                Ds_ts = np.full(z.shape[0], float(res["Ds"]), dtype=float)
            else:
                Ds_ts = np.full(z.shape[0], np.nan, dtype=float)

            W_crest = np.full(z.shape[0], np.nan, dtype=float)

            for i in range(z.shape[0]):
                Ds_i = float(Ds_ts[i])
                if not np.isfinite(Ds_i):
                    continue

                # nodes "near crest elevation"
                mask = z[i, :] >= (Ds_i - crest_tol_m)
                if not np.any(mask):
                    continue

                # select the *contiguous* segment that contains the max elevation (crest)
                imax = int(np.argmax(z[i, :]))

                # expand left
                l = imax
                while l > 0 and mask[l - 1]:
                    l -= 1

                # expand right
                r = imax
                while r < mask.size - 1 and mask[r + 1]:
                    r += 1

                W_crest[i] = x[r] - x[l]

        elif crest_width_fallback_m is not None:
            W_crest = np.full_like(t_h, float(crest_width_fallback_m), dtype=float)

        # plot crest width on secondary axis if available
        if W_crest is not None:
            ax2 = ax1.twinx()
            ax2.plot(t_h, W_crest, linestyle="--", label="Crest width (plateau)", alpha=0.9)
            ax2.set_ylabel("Crest width (m)")

            # combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            ax1.legend(loc="best")
    else:
        ax1.legend(loc="best")

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=160)
    return fig