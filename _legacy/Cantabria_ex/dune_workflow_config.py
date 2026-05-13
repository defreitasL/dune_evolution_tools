from __future__ import annotations

"""User-editable configuration for the Cantabria dune workflow.

Current workflow philosophy:

* dune polygons decide which profiles enter the workflow and bound the detection
  search window;
* profile topography defines heel, crest and frontal toe;
* alongshore smoothing removes isolated outliers before the model is run;
* the model uses the smoothed toe/crest/heel geometry and a local beach slope
  estimated from the 50 m seaward of the smoothed frontal toe.
"""

from pathlib import Path

import numpy as np

from dune_evolution_tools.params import DuneToeStormParams


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs_dune_workflow"
PLOTS_DIR = OUTPUT_DIR / "plots"

# PROFILES_PATH = DATA_DIR / "Cantabria_profiles_35m_checked.pkl"
PROFILES_PATH = DATA_DIR / "Cantabria_profiles_35m_checked.parquet"

STRUCTURES_PATH = DATA_DIR / "defensas_costeras.gpkg"

PROXIES_PKL = OUTPUT_DIR / "01_profiles_with_dune_proxies.pkl"
GEOMETRY_PKL = OUTPUT_DIR / "02_profiles_with_dune_geometry.pkl"
ERODED_PKL = OUTPUT_DIR / "03_profiles_with_eroded_dunes.pkl"

DETECTION_PLOTS_DIR = PLOTS_DIR / "detection"
TRANSLATION_PLOTS_DIR = PLOTS_DIR / "translation"
PLANVIEW_PLOT_PATH = PLOTS_DIR / "planview_toe_crest_initial_final.png"
PLANVIEW_BY_PLAYA_DIR = PLOTS_DIR / "planview_by_playa"


# -----------------------------------------------------------------------------
# Column names / conventions
# -----------------------------------------------------------------------------
PROFILE_ID_COL = "id"
D_COL = "d"
Z_COL = "z_corregido"          # elevation column used by the bridge/model
DETECTION_Z_COL = "z_corregido"     # detector input elevation column
X_COL = "X"
Y_COL = "Y"
IS_DUNE_COL = "is_dune"

STRUCTURE_CLASS_COL = "Clase"
DUNE_VALUES = ("Duna", "duna")
ASSUME_ALL_POLYGONS_ARE_DUNES_IF_NO_CLASS = False
STRUCTURES_FALLBACK_CRS = "EPSG:4326"

# Simulation gate
# ``is_dune`` means that the profile crosses a dune polygon and therefore enters
# the detection stage. ``simulate_dune`` is stricter: it marks the profiles that
# are allowed to run through the erosion model.
#
# With the profile convention used here, ``d`` increases seaward. A dune is not
# simulated when the selected dune-polygon seaward edge lies landward/behind the
# non-erodible line distance (``dist_lnero``).
SIMULATE_DUNE_COL = "simulate_dune"
SIMULATE_DUNE_REASON_COL = "simulate_dune_reason"
NON_ERODIBLE_DISTANCE_COL = "dist_lnero"
SIMULATE_DUNE_REFERENCE_COL = "d_dune_seaward_polygon"
SIMULATE_DUNE_TOLERANCE_M = 0.0
SIMULATE_DUNE_IF_NON_ERODIBLE_MISSING = True


# -----------------------------------------------------------------------------
# Dune proxy extraction from polygons
# -----------------------------------------------------------------------------
EXACT_PROXY_FALLBACK = True
CREST_WIDTH_FRACTION = 0.35
MIN_CREST_WIDTH_M = 0.0
MAX_CREST_WIDTH_M = 20.0


# -----------------------------------------------------------------------------
# Dune geometry detection
# -----------------------------------------------------------------------------
# Detection backend.
#
# Options:
#
#   "ensemble"
#       Use only the internal geometry ensemble: curvature, breakpoint, relative
#       relief, perpendicular distance and physical heel-crest-toe scoring.
#
#   "ensemble_with_pybeach"
#       Keep the internal ensemble, but add pybeach toe/crest predictions as
#       extra candidates. This is the safest way to use pybeach operationally.
#
#   "pybeach_only"
#       Use pybeach as the only source for toe and crest. Because pybeach does
#       not provide a dune-heel detector, the heel is still estimated by a small
#       internal fallback needed by the erosion-model mesh. Toe/crest are not
#       selected by the internal ensemble in this mode.
DETECTION_MODE = "ensemble_with_pybeach"

USE_PYBEACH = DETECTION_MODE in {"ensemble_with_pybeach", "pybeach_only"}
PYBEACH_METHODS = ("ml", "mc", "rr", "pd")
PYBEACH_ML_MODELS = ("mixed_clf",)
PYBEACH_MIN_POINTS = 20
# By default pybeach receives a lightly smoothed profile.  The returned d-values
# are then evaluated back on the original profile, so the final elevations remain
# tied to the observed/corrected topography.
PYBEACH_USE_SMOOTHED_PROFILE = True
PYBEACH_SMOOTH_MEDIAN_WINDOW_M = 7.0
PYBEACH_SMOOTH_MEAN_WINDOW_M = 17.0
PYBEACH_ONLY_TOE_METHOD_PRIORITY = ("ml", "mc", "rr", "pd")
PYBEACH_ONLY_REFINE_CREST_FREEBOARD = False
PYBEACH_ONLY_ALLOW_INTERNAL_CREST_FALLBACK = False

# Clean detector: fit one physically consistent heel-crest-toe geometry inside a
# polygon-derived search window.  The polygon is a spatial prior only; the final
# calculation mesh is allowed to follow the topographic heel detected from the
# profile.
DETECTION_KWARGS = dict(
    detection_mode=DETECTION_MODE,
    use_pybeach=USE_PYBEACH,
    pybeach_methods=PYBEACH_METHODS,
    pybeach_ml_models=PYBEACH_ML_MODELS,
    pybeach_min_points=PYBEACH_MIN_POINTS,
    pybeach_use_smoothed_profile=PYBEACH_USE_SMOOTHED_PROFILE,
    pybeach_smooth_median_window_m=PYBEACH_SMOOTH_MEDIAN_WINDOW_M,
    pybeach_smooth_mean_window_m=PYBEACH_SMOOTH_MEAN_WINDOW_M,
    pybeach_only_toe_method_priority=PYBEACH_ONLY_TOE_METHOD_PRIORITY,
    pybeach_only_refine_crest_freeboard=PYBEACH_ONLY_REFINE_CREST_FREEBOARD,
    pybeach_only_allow_internal_crest_fallback=PYBEACH_ONLY_ALLOW_INTERNAL_CREST_FALLBACK,
    detector="constrained_piecewise",
    search_landward_col="d_dune_landward_polygon",
    search_seaward_col="d_dune_seaward_polygon",
    search_landward_buffer_m=100.0,
    search_seaward_buffer_m=100.0,
    median_window_m=7.0,
    sg_window_m=17.0,
    sg_polyorder=3,
    min_points_in_search=20,
    max_crest_candidates=14,
    min_toe_crest_gap_m=4.0,
    max_toe_from_crest_m=150.0,
    min_heel_crest_gap_m=4.0,
    preferred_min_heel_crest_gap_m=15.0,
    max_heel_from_crest_m=160.0,
    min_crest_toe_relief_m=0.20,
    beach_slope_window_m=50.0,
    beach_slope_min=0.005,
    beach_slope_max=0.20,
    max_beach_to_dune_slope_ratio=0.90,
    polygon_prior_weight=0.05,
    min_confidence_ok=0.35,
)


# -----------------------------------------------------------------------------
# Calculation-domain geometry
# -----------------------------------------------------------------------------
# The numerical domain is: smoothed topographic heel -> smoothed frontal dune toe.
# The polygon columns are retained as spatial references and fallbacks, but they
# no longer truncate the model mesh when a valid topographic heel is available.
TOE_COL = "d_toe_final"
SMOOTHED_TOE_COL = "d_toe_final_smooth"
CREST_COL = "d_crest"
SMOOTHED_CREST_COL = "d_crest_smooth"
HEEL_COL = "d_heel_final"
SMOOTHED_HEEL_COL = "d_heel_final_smooth"
BERM_COL = "d_berm"
Z_TOE_COL = "z_toe_final"
Z_CREST_COL = "z_crest"
POLYGON_LANDWARD_COL = "d_dune_landward_polygon"
POLYGON_SEAWARD_COL = "d_dune_seaward_polygon"
# Kept for compatibility with plotting/helpers; the model itself prefers the
# smoothed topographic heel through SMOOTHED_HEEL_COL.
DUNE_LANDWARD_COL = POLYGON_LANDWARD_COL
DUNE_SEAWARD_COL = POLYGON_SEAWARD_COL   # diagnostic only
DUNE_PROXY_COL = "Y_df_AI_proxy"              # diagnostic only

REFINE_CREST_FROM_PROFILE = True
CREST_REFINE_WINDOW_M = 50.0
CREST_REFINE_MIN_GAIN_M = 0.02
MIN_TOE_CREST_GAP_M = 2.0
MIN_HEEL_CREST_GAP_M = 2.0
MIN_CREST_TOE_RELIEF_M = 0.20

# Beachface slope used by the storm model.  By default it is recalculated from
# the profile seaward of the model toe, because imported slope columns can vary
# strongly depending on the preprocessing window.
USE_INPUT_BEACH_SLOPE = False
BEACH_SLOPE_COL = "mean_beach_slope"
BEACH_SLOPE_MIN = 0.01
BEACH_SLOPE_MAX = 0.12
BEACH_SLOPE_SEAWARD_WINDOW_M = 50.0

# Longitudinal smoothing/QA of the detected toe.  The smoothed toe is the model
# control point whenever it is available.
SMOOTH_TOE_LONGITUDINALLY = True
TOE_SMOOTHING_KWARGS = dict(
    outlier_window=9,
    xy_k=2.5,
    xy_abs_dev_m=8.0,
    z_k=2.5,
    z_abs_dev_m=0.60,
    smooth_window=9,
    smooth_polyorder=2,
    max_candidate_spread_m=20.0,
)
SMOOTH_CREST_LONGITUDINALLY = True
SMOOTH_HEEL_LONGITUDINALLY = True
CREST_SMOOTHING_KWARGS = dict(TOE_SMOOTHING_KWARGS)
HEEL_SMOOTHING_KWARGS = dict(TOE_SMOOTHING_KWARGS)

# The Larson effective-slope expression is ill-conditioned when the foreshore is
# as steep as the dune face.  This safeguard only limits the model foreshore
# slope; it does not move the observed toe, crest or landward boundary.
ENFORCE_MODEL_SLOPE_SEPARATION = True
MIN_DUNE_TO_BEACH_SLOPE_RATIO = 1.15
MIN_DUNE_TO_BEACH_SLOPE_GAP = 0.005
MAX_MODEL_DUNE_FACE_SLOPE = 1.5

# Keep a finite landward segment in the calculation trapezoid.  Without this,
# volume matching can collapse the backdune side into a degenerate plateau/point
# and the mapped profiles look triangular.
MIN_LANDWARD_BACK_SLOPE_LENGTH_M = 3.0


# -----------------------------------------------------------------------------
# Real-profile merge
# -----------------------------------------------------------------------------
# Single merge strategy: translate the final calculation profile back to the
# original profile mesh.  The merge domain follows the final modelled landward
# extent, so dune retreat can write behind the initial calculation domain.
BLEND_WIDTH_M = 10.0
MAX_VERTICAL_CHANGE_M = np.inf

# Hard QC for unstable model outputs.  Failed profiles are copied unchanged and
# tagged as failed rather than writing corrupted geometry to disk.
MAX_MODEL_RETREAT_M = 120.0
MAX_CREST_LOWERING_M = 5.0

CLEAN_PLOTS_ON_RUN = True


# -----------------------------------------------------------------------------
# Example storm forcing
# -----------------------------------------------------------------------------
TIME_S = np.arange(0.0, 48.0 * 3600.0 + 3600.0, 3600.0)
T = np.full_like(TIME_S, 11.0)
TWL = np.full_like(TIME_S, 4.2)
TWL[10:30] += 0.8
TWL[30:] += 0.2
H0 = None
RU = None
RUNUP_MODE = "stockdon"

BASE_PARAMS = DuneToeStormParams(
    Ds=5.0,                  # overwritten by geometry
    z0_init=3.0,             # overwritten by geometry
    tan_beta_f=0.05,         # overwritten by geometry
    Cs=1.8e-3,
    A_overwash=3.0,
    crest_erosion=True,
    k_crest=0.7,
    crest_width_m=10.0,
    use_profile_mesh=False,
)


# -----------------------------------------------------------------------------
# Progress bars
# -----------------------------------------------------------------------------
PROGRESS_BARS = True
PROGRESS_BAR_LEAVE = True


# -----------------------------------------------------------------------------
# Plot controls
# -----------------------------------------------------------------------------
SAVE_DETECTION_PLOTS = True
SAVE_TRANSLATION_PLOTS = True
SAVE_PLANVIEW_PLOT = True

DETECTION_PLOT_KWARGS = dict(xlim=(0, 1500), ylim=(-10, 20))
TRANSLATION_PLOT_KWARGS = dict(xlim=(0, 1500), ylim=(-10, 20))

PLANVIEW_FIGSIZE = (13, 11)
PLANVIEW_BASEMAP = True
PLANVIEW_BASEMAP_SOURCE = "Esri.WorldImagery"
PLANVIEW_BY_PLAYA = True
PLANVIEW_PLAYA_COL = "Playa"
SAVE_GLOBAL_PLANVIEW_PLOT = False
PLANVIEW_PLOT_CRS = "EPSG:3857"
PLANVIEW_EXTENT_BUFFER_RATIO = 0.04

COPY_ORIGINAL_PROFILE_WHEN_SKIPPED = True
