"""High-level handler for the dune toe storm model."""
from __future__ import annotations
from typing import Dict, Literal, Optional
import numpy as np

from .params import DuneToeStormParams
from .core import simulate_core_rk4, simulate_core_rk4_crest
from .runup import runup_stockdon_r2
from .cs import cs_larson2004_eq37
from .mesh import build_mesh_grid, simulate_profile_mesh

RunupMode = Literal["stockdon"]

class DuneToeStormModel:
    """Event-scale dune toe model (Larson 2004 + Larson 2016) with a fast RK4+Numba core."""

    name = "DuneToeStormModel"

    def __init__(self, params: DuneToeStormParams):
        self.params = params

    def _build_Cs_series(self, time_s: np.ndarray, H0: Optional[np.ndarray]) -> np.ndarray:
        """Build a Cs(t) series based on params.

        - If Cs_mode="constant": returns a constant series equal to params.Cs
        - If Cs_mode="larson2004_eq37": uses Larson (2004) Eq. (37) as a function of Hrms0/D50

        For Eq. (37) we need a wave height time series (deep-water). If `H0_is_Hs=True`,
        the input H0 is interpreted as Hs and converted to Hrms via Hrms = Hs/sqrt(2).
        """
        n = time_s.size
        if self.params.Cs_mode == "constant":
            return np.full(n, float(self.params.Cs), dtype=float)

        if self.params.Cs_mode == "larson2004_eq37":
            if H0 is None:
                raise ValueError(
                    "Cs_mode='larson2004_eq37' requires a wave-height series H0 "
                    "(provide it in simulate_from_waves, or via H0_for_Cs in simulate_from_runup)."
                )
            H0 = np.asarray(H0, dtype=float)
            if H0.shape != time_s.shape:
                raise ValueError("H0 must have the same shape as time_s when used to compute Cs")
            Hrms = H0 / np.sqrt(2.0) if bool(self.params.H0_is_Hs) else H0
            return cs_larson2004_eq37(
                Hrms0=Hrms,
                D50_m=float(self.params.D50),
                A=float(self.params.Cs_eq37_A),
                b=float(self.params.Cs_eq37_b),
            ).astype(float)

        raise ValueError(f"Unknown Cs_mode: {self.params.Cs_mode!r}")

    def simulate_from_runup(
        self,
        time_s: np.ndarray,
        Ru: np.ndarray,
        T: np.ndarray,
        H0_for_Cs: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate using user-provided runup Ru(t) and wave period T(t).

        If params.Cs_mode='larson2004_eq37', you must also provide H0_for_Cs (deep-water wave height series)
        to compute Cs(t) via Larson (2004) Eq. (37).
        """
        time_s = np.asarray(time_s, dtype=float)
        Ru = np.asarray(Ru, dtype=float)
        T = np.asarray(T, dtype=float)

        if time_s.ndim != 1:
            raise ValueError("time_s must be 1D")
        if Ru.shape != time_s.shape or T.shape != time_s.shape:
            raise ValueError("Ru and T must match time_s shape")

        Cs_t = self._build_Cs_series(time_s=time_s, H0=H0_for_Cs)

        # --- core integration ---
        if bool(getattr(self.params, "crest_erosion", False)):
            out = simulate_core_rk4_crest(
                time_s=time_s,
                Ru=Ru,
                T=T,
                Cs_t=Cs_t,
                Ds_init=float(self.params.Ds),
                z0_init=float(self.params.z0_init),
                tan_beta_f=float(self.params.tan_beta_f),
                A_overwash=float(self.params.A_overwash),
                alpha_rep_deg=float(self.params.alpha_rep_deg),
                s_min=float(self.params.s_min),
                k_crest=float(getattr(self.params, "k_crest", 1.0)),
                crest_width_m=float(getattr(self.params, "crest_width_m", 10.0)),
            )
            (z0, Ds_ts, x0, zd, xd, V, dVdt, dDsdt, qD, qS, qL, tan_beta_D, tan_beta_eff) = out
        else:
            out = simulate_core_rk4(
                time_s=time_s,
                Ru=Ru,
                T=T,
                Cs_t=Cs_t,
                Ds=float(self.params.Ds),
                z0_init=float(self.params.z0_init),
                tan_beta_f=float(self.params.tan_beta_f),
                A_overwash=float(self.params.A_overwash),
                alpha_rep_deg=float(self.params.alpha_rep_deg),
                s_min=float(self.params.s_min),
            )
            (z0, x0, zd, xd, V, dVdt, qD, qS, qL, tan_beta_D, tan_beta_eff) = out
            Ds_ts = np.full_like(z0, float(self.params.Ds), dtype=float)
            dDsdt = np.zeros_like(z0, dtype=float)


        # --- Geometry / repose diagnostics (diagnostic only; not an external transport term) ---
        # These diagnostics help quantify how imposing a dune-face repose slope affects:
        # - volume V(z0) and its sensitivity dV/dz0
        # - reconstructed profile geometry (dune-face width)
        # - (optional) mesh-based instantaneous avalanching (DRT-style)
        tan_beta_f = float(self.params.tan_beta_f)
        Ds_series = np.asarray(Ds_ts, dtype=float)
        tan_beta_D_val = float(tan_beta_D)
        tan_beta_eff_val = float(tan_beta_eff)

        s = Ds_series - z0  # dune height above the toe [m]

        # "Vertical-face" proxy volume uses tan(beta_f) (limit of very steep dune face).
        if tan_beta_f > 0.0:
            V_vert = np.where(s > 0.0, 0.5 * s * s / tan_beta_f, 0.0)
            dVdz0_vert = np.where(s > 0.0, -s / tan_beta_f, 0.0)
        else:
            V_vert = np.full_like(V, np.nan, dtype=float)
            dVdz0_vert = np.full_like(V, np.nan, dtype=float)

        DeltaV_geom = V - V_vert
        with np.errstate(divide="ignore", invalid="ignore"):
            deltaV = np.where(V_vert > 0.0, V / V_vert - 1.0, np.nan)

        if tan_beta_eff_val > 0.0:
            dVdz0_eff = np.where(s > 0.0, -s / tan_beta_eff_val, 0.0)
        else:
            dVdz0_eff = np.full_like(V, np.nan, dtype=float)

        # Geometric dune-face width implied by repose slope (moving-crest geometry).
        if tan_beta_D_val > 0.0:
            x_face_geom = np.where(s > 0.0, s / tan_beta_D_val, 0.0)
        else:
            x_face_geom = np.full_like(V, np.nan, dtype=float)

        # Optional: build a mesh profile and apply instantaneous avalanching (DRT-style slope limiter)
        mesh_outputs = {}
        if bool(getattr(self.params, "use_profile_mesh", False)):
            crest_mode = getattr(self.params, "crest_mode", "fixed")
            if crest_mode not in ("fixed", "moving"):
                raise ValueError("crest_mode must be 'fixed' or 'moving' when use_profile_mesh=True")
            crest_mode_int = 0 if crest_mode == "fixed" else 1

            # Landward/backdune representation (for mesh boundaries/plots)
            landward_crest_width_m = float(getattr(self.params, "landward_crest_width_m", 30.0))
            landward_back_slope_m = float(getattr(self.params, "landward_back_slope_m", 40.0))
            landward_back_buffer_m = float(getattr(self.params, "landward_back_buffer_m", 20.0))
            z_back = float(getattr(self.params, "z_back", 0.0))
            tan_beta_back = float(getattr(self.params, "tan_beta_back", -1.0))
            if tan_beta_back <= 0.0:
                tan_beta_back = tan_beta_f

            x_prof, xd0 = build_mesh_grid(
                Ds=float(Ds_series[0]),
                z0_init=float(self.params.z0_init),
                tan_beta_f=tan_beta_f,
                tan_alpha=tan_beta_D_val,
                seaward_buffer_m=float(getattr(self.params, "seaward_buffer_m", 50.0)),
                landward_crest_width_m=landward_crest_width_m,
                landward_back_slope_m=landward_back_slope_m,
                landward_back_buffer_m=landward_back_buffer_m,
                dx_mesh=float(getattr(self.params, "dx_mesh", 0.25)),
                crest_mode_int=crest_mode_int,
            )

            z_prof, V_mesh, slope_excess_max, xd_ts = simulate_profile_mesh(
                time_s=time_s,
                x=x_prof,
                z0=z0,
                x0=x0,
                Ds_ts=Ds_series,
                tan_beta_f=tan_beta_f,
                tan_alpha=tan_beta_D_val,
                landward_crest_width_m=landward_crest_width_m,
                tan_beta_back=tan_beta_back,
                z_back=z_back,
                landward_back_slope_m=landward_back_slope_m,
                crest_mode_int=crest_mode_int,
                xd0_fixed=xd0,
                dx_mesh=float(getattr(self.params, "dx_mesh", 0.25)),
                max_avalanche_iters=int(getattr(self.params, "avalanche_max_iters", 60)),
            )

            # Override crest series with mesh crest series (fixed or moving)
            xd = xd_ts.astype(float)

            # Mesh-based dune-face width and its change rate
            x_face = xd - x0
            with np.errstate(invalid="ignore"):
                IA_proxy = np.abs(np.gradient(x_face, time_s))

            DeltaV_mesh = V_mesh - V_vert

            mesh_outputs = {
                "x_prof": x_prof,
                "z_prof": z_prof,
                "V_mesh": V_mesh,
                "DeltaV_mesh": DeltaV_mesh,
                "mesh_slope_excess_max": slope_excess_max,
                "crest_mode": np.array([crest_mode_int], dtype=np.int32),
            }
        else:
            # No mesh: keep moving-crest geometry from the ODE
            x_face = x_face_geom
            with np.errstate(invalid="ignore"):
                IA_proxy = np.abs(np.gradient(x_face, time_s))

        Gamma_geom = (tan_beta_eff_val / tan_beta_f) if tan_beta_f > 0.0 else np.nan
        Gamma_geom_ts = np.full_like(time_s, Gamma_geom, dtype=float)

        beta_D_deg = float(np.degrees(np.arctan(tan_beta_D_val)))
        beta_eff_deg = float(np.degrees(np.arctan(tan_beta_eff_val)))

        return {
            "time_s": time_s,
            "Ru_used": Ru,
            # Note: if you call simulate_from_twl(), Ru_used will store the *effective*
            # water-level series used internally (equal to TWL). In that case, TWL_used
            # will also be provided for clarity.
            "T_used": T,
            "Cs_used": Cs_t,
            "alpha_rep_deg": np.array([float(self.params.alpha_rep_deg)], dtype=float),
            "use_profile_mesh": np.array([1 if bool(getattr(self.params, "use_profile_mesh", False)) else 0], dtype=np.int32),
            "Ds_ts": Ds_series,
            "dDsdt": dDsdt,
            "z0": z0,
            "x0": x0,
            "zd": zd,
            "xd": xd,
            "x_face": x_face,
            "x_face_geom": x_face_geom,
            "IA_proxy": IA_proxy,
            "V": V,
            "V_vert": V_vert,
            "DeltaV_geom": DeltaV_geom,
            "deltaV": deltaV,
            "dVdt": dVdt,
            "dVdz0_eff": dVdz0_eff,
            "dVdz0_vert": dVdz0_vert,
            "qD": qD,
            "qS": qS,
            "qL": qL,
            "tan_beta_D": np.array([tan_beta_D], dtype=float),
            "tan_beta_eff": np.array([tan_beta_eff], dtype=float),
            "Gamma_geom": np.array([Gamma_geom], dtype=float),
            "Gamma_geom_ts": Gamma_geom_ts,
            "beta_D_deg": np.array([beta_D_deg], dtype=float),
            "beta_eff_deg": np.array([beta_eff_deg], dtype=float),
            **mesh_outputs,
        }

    def simulate_from_twl(
        self,
        time_s: np.ndarray,
        TWL: np.ndarray,
        T: np.ndarray,
        H0_for_Cs: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate using a *total water level* (TWL) time series.

        This is a convenience wrapper around :meth:`simulate_from_runup` that treats the
        provided TWL(t) as the effective water-level forcing used by the core.

        Interpretation
        --------------
        - TWL(t) must be in the **same vertical datum** as the model elevations z0 and Ds.
        - The core equations only use exceedances like (water_level - z0) and the
          overwash threshold (water_level > Ds).
        - Therefore, internally we simply pass Ru := TWL.

        Notes
        -----
        If params.Cs_mode='larson2004_eq37', you must provide H0_for_Cs (deep-water wave height
        series, either Hrms0 directly or Hs depending on params.H0_is_Hs) to compute Cs(t).
        """
        time_s = np.asarray(time_s, dtype=float)
        TWL = np.asarray(TWL, dtype=float)
        T = np.asarray(T, dtype=float)

        if time_s.ndim != 1:
            raise ValueError("time_s must be 1D")
        if TWL.shape != time_s.shape or T.shape != time_s.shape:
            raise ValueError("TWL and T must match time_s shape")

        res = self.simulate_from_runup(time_s=time_s, Ru=TWL, T=T, H0_for_Cs=H0_for_Cs)
        # Add TWL explicitly for clarity.
        res["TWL_used"] = TWL
        res["water_level_mode"] = np.array(["twl"], dtype="<U8")
        return res

    def simulate_from_waves(
        self,
        time_s: np.ndarray,
        H0: np.ndarray,
        T: np.ndarray,
        runup_mode: RunupMode = "stockdon",
    ) -> Dict[str, np.ndarray]:
        """Simulate using wave inputs, computing Ru(t) internally.

        Parameters
        ----------
        H0 : array
            Deep-water wave height series [m] (often Hs).
        T : array
            Peak wave period series [s] (often Tp).
        runup_mode : {"stockdon"}
            Currently only Stockdon R2% is implemented.

        Notes
        -----
        If params.Cs_mode='larson2004_eq37', this method will compute Cs(t) from H0 and D50.
        """
        time_s = np.asarray(time_s, dtype=float)
        H0 = np.asarray(H0, dtype=float)
        T = np.asarray(T, dtype=float)

        if time_s.ndim != 1:
            raise ValueError("time_s must be 1D")
        if H0.shape != time_s.shape or T.shape != time_s.shape:
            raise ValueError("H0 and T must match time_s shape")

        if runup_mode != "stockdon":
            raise ValueError("Only runup_mode='stockdon' supported")

        Ru = runup_stockdon_r2(H0=H0, T=T, tan_beta_f=float(self.params.tan_beta_f))
        res = self.simulate_from_runup(time_s=time_s, Ru=Ru, T=T, H0_for_Cs=H0)
        res["H0_used"] = H0
        return res

    @staticmethod
    def build_profile_xy(
        z0: float,
        x0: float,
        Ds: float,
        tan_beta_f: float,
        tan_beta_D: float,
        seaward_buffer_m: float = 50.0,
        landward_crest_width_m: float = 30.0,
        landward_back_slope_m: float = 40.0,
        landward_back_buffer_m: float = 20.0,
        z_back: float = 0.0,
        tan_beta_back: float = -1.0,
        n_foreshore: int = 200,
    ):
        """Build a simple polyline (x,z) profile for plotting."""
        if tan_beta_f <= 0.0:
            raise ValueError("tan_beta_f must be > 0")
        if tan_beta_D <= 0.0:
            raise ValueError("tan_beta_D must be > 0")

        x_swl = x0 - z0 / tan_beta_f
        x_sea = x_swl - seaward_buffer_m

        # foreshore (line passing through toe and z=0 at x_swl)
        x_fs = np.linspace(x_sea, x0, max(2, int(n_foreshore)))
        z_fs = tan_beta_f * (x_fs - x_swl)

        # dune face
        xd = x0 + (Ds - z0) / tan_beta_D

        # landward representation: plateau + backdune slope + optional flat buffer
        xp = xd + float(landward_crest_width_m)

        tanb = float(tan_beta_back)
        if tanb <= 0.0:
            tanb = float(tan_beta_f)

        x_pts = [x0, xd, xp]
        z_pts = [z0, Ds, Ds]

        # Backdune slope (if requested)
        Ls = float(landward_back_slope_m)
        Lbuf = float(landward_back_buffer_m)
        zmin = float(z_back)

        if Ls > 0.0 and tanb > 0.0:
            # distance needed to reach z_back
            dist_to_zmin = (Ds - zmin) / tanb if Ds > zmin else 0.0
            if dist_to_zmin > 0.0 and dist_to_zmin < Ls:
                # reach z_back before the slope length ends
                x_hit = xp + dist_to_zmin
                x_end = xp + Ls
                x_pts.extend([x_hit, x_end])
                z_pts.extend([zmin, zmin])
            else:
                # do not reach z_back within Ls (or Ds<=z_back), keep linear to the end
                x_end = xp + Ls
                z_end = Ds - tanb * Ls
                if z_end < zmin:
                    z_end = zmin
                x_pts.append(x_end)
                z_pts.append(z_end)

            # optional flat buffer beyond the back-slope endpoint
            if Lbuf > 0.0:
                x_pts.append(x_pts[-1] + Lbuf)
                z_pts.append(z_pts[-1])
        else:
            # No back-slope: just extend a finite plateau for visualization
            if Lbuf > 0.0:
                x_pts.append(xp + Lbuf)
                z_pts.append(Ds)

        x = np.concatenate([x_fs, np.array(x_pts, dtype=float)])
        z = np.concatenate([z_fs, np.array(z_pts, dtype=float)])
        return x, z
