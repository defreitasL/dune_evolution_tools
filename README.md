# 🏖️ dune_evolution_tools
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Numba](https://img.shields.io/badge/numba-accelerated-orange)
![Scope](https://img.shields.io/badge/scope-storm--scale-9cf)
![Status](https://img.shields.io/badge/status-alpha-yellow)
![Build](https://img.shields.io/badge/build-setuptools-informational)

---

## 📝 Description
`dune_evolution_tools` is a **fast, storm-event (hours–days)** dune erosion package that evolves the **dune toe elevation** (and optionally the **crest elevation**) using a reduced-complexity, alongshore-averaged **cross-shore** framework.

### ✨ Key features
- ⚡ **Numba-accelerated RK4 core** for rapid event simulations and ensembles
- 📉 **Dune toe evolution** from a geometric mass-balance closure (Larson et al., 2004)
- 🌊 **Overwash partitioning** into seaward/landward fluxes (Larson et al., 2016)
- 🌡️ Flexible forcing:
  - provide **runup** series `Ru(t)`
  - provide **Total Water Level** series `TWL(t)`
  - provide **waves** `H0(t), T(t)` and compute runup with **Stockdon** internally
- 🧩 Optional **profile mesh** reconstruction + **instantaneous avalanching** (DRT-style slope limiter; inspired by Cohn & Anderson, 2025)
- ✅ Built-in **mass-closure diagnostic**: check that `V(t) ≈ V0 − ∫qD dt`

### 🎯 Intended use
- Storm-scale evolution (≈ **6–72 h**, and generally **hours to a few days**)
- Rapid scenario testing, sensitivity analyses, Monte Carlo ensembles
- Not designed for long-term recovery, vegetation feedbacks, or detailed 2D/3D morphodynamics

---

## 🧰 Installation
From the repository root (recommended for development):

```bash
pip install -e .
```

Or standard installation:

```bash
pip install https://github.com/defreitasL/dune_evolution_tools.git
```

### Dependencies
- `numpy`
- `matplotlib`
- `numba` *(recommended; the core will run without it but slower)*

---

## 📐 Model
This section documents the equations **as implemented in the package**, and how they relate to the reference papers.

### 1) Geometry & state variables (Larson 2004)
The model evolves the **toe elevation** $(z_0(t))$ and uses a dune “height above toe”:

$$
s(t)=D_s(t)-z_0(t)
$$

A wedge-type dune volume per unit alongshore length (m³/m = m²) is:

$$
V(t)=\frac{1}{2}\frac{s(t)^2}{\tan(\beta_{\mathrm{eff}})}
$$
This is the same wedge-volume idea used by **Larson et al. (2004)** (Eq. 13), but in this package we allow both **$z_0(t)$** and (optionally) **$D_s(t)$** to evolve.

### 2) Effective slope / dune-face repose (Larson 2004 + DRT convention)
The dune-face slope is constrained by an **angle of repose** provided by the user:
- `alpha_rep_deg` is the repose angle **with respect to the horizontal** (DRT convention)

The corresponding tangent is:

$$
	\tan(\beta_D)=\tan(\alpha_{\mathrm{rep}})
$$

We then compute an **effective slope** $(\tan(\beta_{\mathrm{eff}}))$ using the **slope substitution** idea (Larson et al., 2004; Eq. 21), which prevents unrealistically steep faces while preserving the wedge-volume closure.

### 3) Mass balance closure (core)
The model is built around:

$$
\frac{dV}{dt} = -q_D
$$
where $q_D$ is the total erosion flux per unit alongshore length (units $m^3/(m\,s)=m^2/s$).

Since:

$$
V=\frac{1}{2}\frac{(D_s-z_0)^2}{\tan(\beta_{\mathrm{eff}})}
$$
then:

$$
\frac{dV}{dt}=\frac{s}{\tan(\beta_{\mathrm{eff}})}\left(\frac{dD_s}{dt}-\frac{dz_0}{dt}\right)
$$

#### Toe ODE (fixed crest)
If the crest is held constant ($dD_s/dt=0$):

$$
\boxed{
\frac{dz_0}{dt} = q_D\frac{\tan(\beta_{\mathrm{eff}})}{\max(s,s_{\min})}
}
$$
where `&s_min$` is a small numerical safeguard.

#### Toe + crest (optional coupled system)
If crest lowering is enabled, the model keeps the wedge mass balance consistent:

$$
\boxed{
\frac{dz_0}{dt} = \frac{dD_s}{dt} + q_D\frac{\tan(\beta_{\mathrm{eff}})}{\max(s,s_{\min})}
}
$$

### 4) Forcing: Runup vs TWL vs waves (Stockdon)
Internally the core uses a single “effective water-level” time series $\eta(t)$:

- **Runup mode**: $\eta(t)=Ru(t)$ (you provide runup)
- **TWL mode**: $\eta(t)=TWL(t)$ (you provide total water level)
- **Waves mode**: you provide `$H0(t)$, $T(t)$` and the package computes runup using **Stockdon et al. (2006)**

> ⚠️ Datum consistency: if you use `TWL(t)`, then `z0_init` and `Ds` must be in the **same vertical datum** as TWL.

### 5) Erosion flux $q_D$ (Larson-family impact formulation)
The implemented flux is **piecewise** to represent collision vs overwash behavior:

- If $\eta \le z_0$: no collision → $q_D=0$
- If $z_0 < \eta \le D_s$ (collision):

$$
q_D = 4\,C_s\,\frac{(\eta-z_0)^2}{T}
$$
- If $\eta > D_s$ (overwash regime):

$$
q_D = 4\,C_s\,\frac{(\eta-z_0)\,(D_s-z_0)}{T}
$$

This is consistent with the Larson-type impact framework (Larson et al., 2004) and is used together with the overwash partitioning described below (Larson et al., 2016).

### 6) Overwash partitioning (Larson 2016)
When $\eta > D_s$, the model partitions the total flux $q_D$ into:
- $q_S$: seaward component
- $q_L$: landward (overwash) component

Using the Larson (2016) CS-model style partitioning:

$$
q_S=\frac{q_D}{1+\alpha},\qquad q_L=q_D-q_S
$$
with $\alpha$ increasing with exceedance above the crest; in the code:

$$
\alpha=\frac{1}{A}\left(\frac{\eta-z_0}{D_s-z_0}-1\right),\quad \alpha\ge 0
$$
where `A_overwash` is the scaling parameter $A$.

### 7) Transport coefficient $C_s$ (Larson 2004, Eq. 37)
Two options are available:
- **Constant**: `Cs_mode="constant"` → uses `Cs`
- **Larson (2004) Eq. 37**: `Cs_mode="larson2004_eq37"`:

$$
C_s = A\,\exp\left(-b\,\frac{Hrms_0}{D_{50}}\right)
$$
Default coefficients are provided in `params.py`.  
When using Eq. 37, you must supply a wave height series for `Hrms0` (either directly as RMS, or as `Hs` with `H0_is_Hs=True` so the model converts `Hrms = Hs/√2`).

### 8) Optional storm-only crest lowering (no recovery)
If enabled (`crest_erosion=True`), the crest can lower during overwash:

$$
\boxed{
\frac{dD_s}{dt}=-\frac{k_{\mathrm{crest}}}{W_{\mathrm{crest}}}\;q_L
}
$$
- `crest_width_m` = $W_{\mathrm{crest}}$: **effective cross-shore width** of the active crest-lowering zone
- `k_crest` = fraction of $q_L$ that produces crest lowering (0–1)

This is a **storm-only** mechanism and does not include post-storm recovery.

### 9) Optional profile mesh + instantaneous avalanching (Cohn 2025 inspiration)
If `use_profile_mesh=True`, the model reconstructs $z(x,t)$ on a 1D grid and applies an **instantaneous slope limiter** so that:

$$
|dz/dx|\le \tan(\alpha_{\mathrm{rep}})
$$
This is conceptually similar to the DRT-style approach (Cohn & Anderson, 2025), but implemented as a fast “instantaneous adjustment” step after each time update.

---

## 🚀 How to use the package

### ✅ Quick start (Python API)
```python
import numpy as np
from dune_evolution_tools import DuneToeStormParams, DuneToeStormModel

params = DuneToeStormParams(
    Ds=3.0,
    z0_init=0.5,
    tan_beta_f=0.10,
    Cs_mode="larson2004_eq37",
    D50=0.00025,
    A_overwash=3.0,
    alpha_rep_deg=37.0,
)

model = DuneToeStormModel(params)
```

### 1) Waves → Stockdon runup (internal)
```python
res = model.simulate_from_waves(time_s=time_s, H0=H0, T=Tp, runup_mode="stockdon")
```

### 2) User-provided runup series
```python
# If Cs_mode="larson2004_eq37", also provide H0_for_Cs (Hrms or Hs depending on H0_is_Hs)
res = model.simulate_from_runup(time_s=time_s, Ru=Ru, T=Tp, H0_for_Cs=Hrms)
```

### 3) User-provided Total Water Level (TWL)
```python
res = model.simulate_from_twl(time_s=time_s, TWL=TWL, T=Tp, H0_for_Cs=H0)
```

### 📂 Run the bundled examples
From the repo root:

```bash
python examples/example_Stockdon.py
python examples/example_runnup_series.py
python examples/example_twl_storm_run.py
python examples/example_storm_run.py
```

What each example demonstrates:

- `examples/example_Stockdon.py`  
  🌊 Waves → Stockdon runup, **mesh + avalanching**, optional **crest lowering**

- `examples/example_runnup_series.py`  
  📈 Runup-driven simulation with **user-provided** `Ru(t)` and **input** `Hrms(t)` for `Cs(t)` (Eq. 37)

- `examples/example_twl_storm_run.py`  
  🌡️ **TWL-driven** simulation (the model receives only `TWL(t)`), plus dune-width plots

- `examples/example_storm_run.py`  
  🧪 Minimal handler example + **mass-closure check** + volume plot

> 💡 Tip: some examples use fine temporal resolution; if you want faster runs, edit the `time_s` definition inside the example.

### 📦 What you get in `res`
The model returns a dict of NumPy arrays, including (most common keys):

- `time_s`, `T_used`, `Cs_used`
- `z0`, `Ds_ts`, `dDsdt`
- `x0`, `xd`, `x_face`
- `V`, `dVdt`, `qD`, `qS`, `qL`
- Geometry diagnostics: `tan_beta_D`, `tan_beta_eff`, `IA_proxy`, `DeltaV_geom`, …
- If `use_profile_mesh=True`: `x_prof`, `z_prof`, `V_mesh`, `mesh_slope_excess_max`, …

### 📊 Plot helpers
You can generate publication-ready quicklooks with:
- `plot_profiles_over_time(...)`
- `plot_positions(res)`
- `plot_transports(res)`
- `plot_volume_timeseries(res)`
- `plot_dune_width(res)` *(dune width; optional crest width if mesh outputs exist)*
- `plot_avalanche_diagnostics(res)`

### ✅ Mass closure diagnostic
```python
from dune_evolution_tools.diagnostics import check_mass_closure
clo = check_mass_closure(res)
print(clo["max_abs_residual"], clo["rel_max_abs_residual"])
```

---

## 📚 References
- **Larson, M., Erikson, L., & Hanson, H. (2004)**. *An analytical model to predict dune erosion due to wave impact.* Coastal Engineering, 51, 675–696. https://doi.org/10.1016/j.coastaleng.2004.07.003  
- **Larson, M., Palalane, J., Fredriksson, C., & Hanson, H. (2016)**. *Simulating cross-shore material exchange at decadal scale. Theory and model component validation.* Coastal Engineering, 116, 57–66. https://doi.org/10.1016/j.coastaleng.2016.05.009  
- **Cohn, N., & Anderson, D. (2025)**. *Projecting the Longevity of Coastal Foredunes Under Stochastic Meteorological and Oceanographic Forcing.* Earth’s Future, 13, e2024EF005335. https://doi.org/10.1029/2024EF005335  
- **Stockdon, H. F., Holman, R. A., Howd, P. A., & Sallenger, A. H. (2006)**. *Empirical parameterization of setup, swash, and runup.* Coastal Engineering, 53, 573–588.

---

## 👤 Author
**Lucas de Freitas Pereira**  
IHCantabria — Environmental Hydraulics Institute of Cantabria (Spain)  
📧 lucas.defreitas@unican.es
