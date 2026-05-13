# Cantabria dune-erosion workflow

This folder contains a compact workflow for detecting dune geometry from beach profiles, running the dune erosion model, and translating the model result back to the original profile geometry.

The workflow is controlled from `dune_workflow_config.py` and executed with:

```bash
python run_dune_workflow.py
```

The current design intentionally separates three decisions that were mixed in earlier versions:

1. **Dune crossing**: the profile intersects a dune polygon and should enter the geometry-detection stage.
2. **Geometry detection**: the profile receives toe, crest and heel diagnostics, even if it will not be simulated.
3. **Simulation gate**: the profile is allowed to run through the erosion model only if `simulate_dune == True`.

This makes the outputs easier to audit: every dune crossing can be checked, but protected/non-erodible cases are copied unchanged during the model stage.

---

## Main files

| File | Purpose |
|---|---|
| `dune_workflow_config.py` | User-editable paths, column names, detector settings, storm forcing, plotting controls and simulation gate. |
| `run_dune_workflow.py` | Main three-step workflow: polygon intersection, geometry detection, model + translation. |
| `dune_proxy_helpers.py` | Reads profile/polygon layers and computes dune-polygon intersections along each profile. |
| `dune_geometry_ensemble.py` | Ensemble detector for dune toe, crest and heel. It can optionally use `pybeach` as an extra candidate source. |
| `dune_real_profile_bridge.py` | Builds the calculation geometry, runs the model bridge and merges model results back to the real profile. |

Legacy detector scripts from previous iterations are no longer used by this workflow and were removed from this clean version.

---

## Outputs

The workflow writes three pickle files in `OUTPUT_DIR`:

1. `01_profiles_with_dune_proxies.pkl`
   - Original profiles plus dune-polygon intersection attributes.
   - Adds `is_dune` and `simulate_dune`.

2. `02_profiles_with_dune_geometry.pkl`
   - Product 01 plus detected/smoothed dune toe, crest and heel.
   - Detection is applied to all profiles with `is_dune == True`, regardless of `simulate_dune`.

3. `03_profiles_with_eroded_dunes.pkl`
   - Product 02 plus model/translation diagnostics and final arrays:
     - `d_dune_eroded`
     - `z_dune_eroded`
     - `x_dune_eroded`
     - `y_dune_eroded`

Skipped profiles keep the original profile arrays when `COPY_ORIGINAL_PROFILE_WHEN_SKIPPED = True`.

Diagnostic plots are written under:

```text
outputs_dune_workflow/plots/detection/
outputs_dune_workflow/plots/translation/
outputs_dune_workflow/plots/planview_by_playa/
```

---

## Simulation gate: `simulate_dune`

`is_dune` means that a profile intersects a dune polygon. It is used to decide whether the detector runs.

`simulate_dune` is stricter. It decides whether the erosion model runs.

The current rule uses the non-erodible line distance from the original profile table:

```python
NON_ERODIBLE_DISTANCE_COL = "dist_lnero"
SIMULATE_DUNE_REFERENCE_COL = "d_dune_seaward_polygon"
```

The workflow convention is that distance `d` increases seaward. Therefore:

```text
simulate_dune = True
    if the seaward edge of the selected dune polygon is seaward of, or equal to, dist_lnero

simulate_dune = False
    if the selected dune polygon lies behind/landward of dist_lnero
```

In code terms, for profiles that cross a dune polygon:

```python
simulate_dune = d_dune_seaward_polygon >= dist_lnero - SIMULATE_DUNE_TOLERANCE_M
```

The workflow also stores diagnostic columns:

| Column | Meaning |
|---|---|
| `simulate_dune` | Boolean gate used by the model stage. |
| `simulate_dune_reason` | Reason for the decision. |
| `simulate_dune_reference_d` | Along-profile distance used as the dune reference, usually `d_dune_seaward_polygon`. |
| `simulate_dune_lnero_d` | Along-profile distance of the non-erodible line. |

Typical `simulate_dune_reason` values:

```text
simulate_dune
no_dune_polygon
dune_behind_non_erodible_line
missing_d_dune_seaward_polygon
simulate_missing_dist_lnero
missing_dist_lnero
```

If `dist_lnero` is missing and you still want to simulate dune crossings, keep:

```python
SIMULATE_DUNE_IF_NON_ERODIBLE_MISSING = True
```

If missing `dist_lnero` should block simulation, set it to `False`.

---

## Recommended workflow for a new run

1. Edit `dune_workflow_config.py`:
   - input paths;
   - `PROFILE_ID_COL`;
   - profile columns (`D_COL`, `Z_COL`, `X_COL`, `Y_COL`);
   - polygon class column and dune class values;
   - simulation gate columns;
   - storm forcing.

2. Run:

```bash
python run_dune_workflow.py
```

3. Check Product 01:
   - number of profiles with `is_dune == True`;
   - number of profiles with `simulate_dune == True`;
   - `simulate_dune_reason` counts.

4. Check Product 02:
   - `detection_status`;
   - `detection_confidence`;
   - `d_toe_final_smooth`;
   - `d_crest_smooth`;
   - `d_heel_final_smooth`.

5. Check Product 03:
   - `dune_model_status`;
   - `dune_model_message`;
   - `d_toe_model`, `d_crest_model`;
   - translation plots.

---

## Optional pybeach support

The internal ensemble detector works without `pybeach`.

If `pybeach` is installed in the current environment, it can be used as an additional candidate source by setting:

```python
USE_PYBEACH = True
```

If `pybeach` is unavailable, the workflow continues with the internal detector. The columns:

```text
pybeach_enabled
pybeach_available
pybeach_used
```

help diagnose whether it was actually used.

Because `pybeach` currently depends on older pandas versions, it is better to use it from a dedicated Python 3.9/3.10 environment rather than from the main model environment.

By default, the profile passed to pybeach is lightly smoothed first:

```python
PYBEACH_USE_SMOOTHED_PROFILE = True
PYBEACH_SMOOTH_MEDIAN_WINDOW_M = 7.0
PYBEACH_SMOOTH_MEAN_WINDOW_M = 17.0
```

The smoothing is used only to help pybeach identify toe/crest positions on a cleaner morphology. Reported elevations and final diagnostics are still interpolated from the original/corrected profile used by the workflow.

---

## Notes on profile distance convention

The scripts assume:

```text
d small  = landward
d large  = seaward
```

Therefore a physically valid detected dune geometry should generally satisfy:

```text
d_heel < d_crest < d_toe
```

and the dune-polygon interval should satisfy:

```text
d_dune_landward_polygon < d_dune_seaward_polygon
```

If a new dataset uses the opposite profile orientation, fix the profile orientation before running the workflow.

---

## Detection backends

The detector is configured with `DETECTION_MODE` in `dune_workflow_config.py`.

```python
DETECTION_MODE = "ensemble"              # internal methods only
DETECTION_MODE = "ensemble_with_pybeach" # internal methods + pybeach candidates
DETECTION_MODE = "pybeach_only"          # pybeach toe/crest only
```

### `ensemble`

Uses only the internal detector:

```text
local maxima for crest candidates
curvature candidates for toe
beach/dune breakpoint candidates
relative-relief candidates
perpendicular-distance candidates
heel candidates from backdune/topography/polygon fallback
physical scoring of complete heel-crest-toe geometries
```

This mode does not require `pybeach`.

### `ensemble_with_pybeach`

Adds pybeach toe/crest predictions to the internal candidate pool. The final geometry is still chosen by the internal physical scorer. This is usually the safest operational mode because pybeach can help propose the toe/crest, but physically inconsistent combinations can still be rejected.

### `pybeach_only`

Uses pybeach as the only source for toe and crest:

```text
pybeach -> dune toe
pybeach -> dune crest
internal fallback -> dune heel
```

The internal detector is not allowed to choose toe or crest in this mode. The only internal part retained is the heel fallback, because pybeach does not provide a dune-heel detector and the erosion-model mesh still needs a landward boundary.

Useful options:

```python
PYBEACH_METHODS = ("ml", "mc", "rr", "pd")
PYBEACH_ML_MODELS = ("mixed_clf",)
PYBEACH_USE_SMOOTHED_PROFILE = True
PYBEACH_SMOOTH_MEDIAN_WINDOW_M = 7.0
PYBEACH_SMOOTH_MEAN_WINDOW_M = 17.0
PYBEACH_ONLY_TOE_METHOD_PRIORITY = ("ml", "mc", "rr", "pd")
PYBEACH_ONLY_REFINE_CREST_FREEBOARD = False
PYBEACH_ONLY_ALLOW_INTERNAL_CREST_FALLBACK = False
```

If `PYBEACH_ONLY_REFINE_CREST_FREEBOARD=True`, the crest distance returned by pybeach is locally snapped to the highest topographic point in a small window. This improves freeboard estimates but is no longer a strictly pybeach-only crest position.

If `PYBEACH_ONLY_ALLOW_INTERNAL_CREST_FALLBACK=True`, the workflow can fall back to the internal crest detector when pybeach fails. Keep this `False` when you want a strict pybeach-only comparison.

---

## Full workflow map

### Step 0 — Configuration

`dune_workflow_config.py` defines:

```text
input profile and polygon paths
profile columns: d, z, X, Y, id
polygon class column and dune class values
detection backend and detector settings
simulation gate using dist_lnero
storm-model parameters
plot/output controls
```

### Step 1 — Dune-polygon intersection

`build_dune_proxies()` loads profiles and polygons, converts polygons to the profile CRS, keeps only dune polygons, and intersects each profile with the dune polygons.

Main outputs:

```text
is_dune
d_dune_landward_polygon
d_dune_seaward_polygon
x/y dune proxy points
polygon width diagnostics
simulate_dune
simulate_dune_reason
```

`is_dune=True` means the profile crosses a dune polygon. `simulate_dune=True` means the profile is allowed to run through the erosion model.

### Step 2 — Geometry detection

`build_dune_geometry()` selects only `is_dune=True` profiles and detects:

```text
d_toe_final, z_toe_final
d_crest, z_crest
d_heel_final, z_heel_final
```

The search window is based on the dune-polygon interval plus buffers:

```text
d_dune_landward_polygon - 100 m
d_dune_seaward_polygon + 100 m
```

The detector then smooths toe, crest and heel by `Playa`:

```text
d_toe_final_smooth
d_crest_smooth
d_heel_final_smooth
```

### Step 3 — Model gate

`run_dune_model_batch()` loops over all profiles, but only simulates profiles where:

```text
is_dune == True
simulate_dune == True
geometry is finite and physically valid
```

Profiles that do not pass this gate are copied unchanged and tagged with `dune_model_status` and `dune_model_message`.

### Step 4 — Calculation mesh

For simulated profiles, `dune_real_profile_bridge.py` builds the calculation geometry using the smoothed topographic features:

```text
landward boundary: d_heel_final_smooth, with fallback if missing
crest: d_crest_smooth
toe: d_toe_final_smooth
beach slope: recalculated from the 50 m seaward of the smoothed toe
```

The polygon is retained as a spatial reference and fallback, but it does not hard-truncate the numerical mesh when a valid topographic heel is available.

### Step 5 — Dune erosion model

The storm model runs on the calculation geometry and returns final model positions and profile geometry.

### Step 6 — Translation back to the real profile

The model output is translated back to the original profile geometry. Failed or skipped profiles keep their original `d/z/x/y` arrays. Successful profiles receive:

```text
d_dune_eroded
z_dune_eroded
x_dune_eroded
y_dune_eroded
```

### Step 7 — Diagnostics and maps

The workflow saves:

```text
detection plots per profile
translation plots per simulated profile
planview map for all profiles
planview maps by Playa
```

These plots are essential for checking whether errors come from polygon intersection, toe/crest/heel detection, the storm model, or the translation back to the real profile.
