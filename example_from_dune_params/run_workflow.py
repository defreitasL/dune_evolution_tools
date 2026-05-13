import numpy as np
from dune_evolution_tools import DuneToeStormParams
from dune_evolution_tools.dune_params_workflow import (
    simulate_dune_profile_from_dune_params_parquet,
    run_dune_model_from_dune_params_parquet,
)
import pandas as pd

# FORCING 
TIME_S = np.arange(0.0, 48.0 * 3600.0 + 3600.0, 10.0*60.0)  # 48 hours at 10-minute intervals
Hs0 = np.full_like(TIME_S, 2.0)
T = np.full_like(TIME_S, 11.0)
TWL = np.full_like(TIME_S, 4.2)

base_params = DuneToeStormParams(
    Ds=5.0,                 # NOT USED: placeholder
    z0_init=3.0,            # NOT USED: placeholder
    tan_beta_f=0.05,        # NOT USED: placeholder
    Cs=1.8e-3,
    A_overwash=3.0,
    crest_erosion=True,
    k_crest=0.7,
    crest_width_m=10.0,
    use_profile_mesh=False,
)

# Single profile
d_final, z_final, V_to_beach = simulate_dune_profile_from_dune_params_parquet(
    "./example_from_dune_params/data/cantabria_dune_parameters.parquet",
    profile_id=2100,
    time_s=TIME_S,
    TWL=TWL,
    Hs0=Hs0,
    T=T,
    base_params=base_params,
)

import matplotlib.pyplot as plt
gdf = pd.read_parquet("./example_from_dune_params/data/cantabria_dune_parameters.parquet")
profile = gdf[gdf["id"] == 2100].iloc[0]
d_initial = profile["d"]
z_initial = profile["z_corregido"]
plt.figure(figsize=(10, 6))
plt.plot(d_initial, z_initial, label="Initial profile", color="blue")
plt.plot(d_final, z_final, label="Final profile", color="red")
plt.xlabel("Distance from shoreline (m)")
plt.ylabel("Elevation (m)")
plt.title("Dune profile evolution")
plt.legend()
plt.grid()
plt.show()

