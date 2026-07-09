import os
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "outputs", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 7 target datasets: name -> (filepath, target column name, short code)
TARGETS = {
    "NH3-N loss":  (os.path.join(DATA_DIR, "__16", "data_for_NH3-N loss (%).csv"),  "NH3-N loss (%)",  "NH3"),
    "N2O-N loss":  (os.path.join(DATA_DIR, "__16", "data_for_N2O-N loss (%).csv"),  "N2O-N loss (%)",  "N2O"),
    "CH4-C loss":  (os.path.join(DATA_DIR, "__16", "data_for_CH4-C loss (%).csv"),  "CH4-C loss (%)",  "CH4"),
    "CO2-C loss":  (os.path.join(DATA_DIR, "__16", "data_for_CO2-C loss (%).csv"),  "CO2-C loss (%)",  "CO2"),
    "TC loss":     (os.path.join(DATA_DIR, "__16", "data_for_TC loss (%).csv"),     "TC loss (%)",     "TC"),
    "TN loss":     (os.path.join(DATA_DIR, "__16", "data_for_TN loss (%).csv"),     "TN loss (%)",     "TN"),
    "Final GI":    (os.path.join(DATA_DIR, "__18", "data_for_Final GI (%).csv"),    "Final GI (%)",    "GI"),
}

# 20 features common to all 7 datasets (used for cross-target comparison: Fig4/5/8)
COMMON_FEATURES = [
    'Material_Main','Material_2','Material_3','Additive_1','Additive_2','Additive_3','Additive_4',
    ' Composting Method','M1_is Enclosed','V2_Ventilation Interval (min)','Turning times','Period (d)',
    'V1_Ventilation Type','V4_Ventilation Day','V3_Ventilation Duration (min)','Compost volume (m3)',
    'Application Rate (%DW)','V5_Ventilation rate (L/min/kg iniDW)','Initial Moisture Content (%)','Initial C/N (%)'
]

# Short display labels for common features (for plotting)
FEATURE_LABELS = {
    'Material_Main': 'Primary feedstock type',
    'Material_2': 'Secondary feedstock type',
    'Material_3': 'Tertiary feedstock type',
    'Additive_1': 'Additive 1 type',
    'Additive_2': 'Additive 2 type',
    'Additive_3': 'Additive 3 type',
    'Additive_4': 'Additive 4 type',
    ' Composting Method': 'Composting method',
    'M1_is Enclosed': 'Enclosed system (Y/N)',
    'V2_Ventilation Interval (min)': 'Aeration interval (min)',
    'Turning times': 'Turning frequency',
    'Period (d)': 'Composting period (d)',
    'V1_Ventilation Type': 'Aeration type',
    'V4_Ventilation Day': 'Aeration onset day',
    'V3_Ventilation Duration (min)': 'Aeration duration (min)',
    'Compost volume (m3)': 'Pile volume (m3)',
    'Application Rate (%DW)': 'Bulking agent rate (%DW)',
    'V5_Ventilation rate (L/min/kg iniDW)': 'Aeration rate (L/min/kg)',
    'Initial Moisture Content (%)': 'Initial moisture (%)',
    'Initial C/N (%)': 'Initial C/N ratio',
}

# Feature groups for grouped heatmaps / network coloring
FEATURE_GROUPS = {
    'Material_Main': 'Feedstock', 'Material_2': 'Feedstock', 'Material_3': 'Feedstock',
    'Additive_1': 'Additive', 'Additive_2': 'Additive', 'Additive_3': 'Additive', 'Additive_4': 'Additive',
    ' Composting Method': 'Process design', 'M1_is Enclosed': 'Process design', 'Compost volume (m3)': 'Process design',
    'Application Rate (%DW)': 'Process design',
    'V1_Ventilation Type': 'Aeration', 'V2_Ventilation Interval (min)': 'Aeration',
    'V3_Ventilation Duration (min)': 'Aeration', 'V4_Ventilation Day': 'Aeration',
    'V5_Ventilation rate (L/min/kg iniDW)': 'Aeration',
    'Turning times': 'Management practice', 'Period (d)': 'Management practice',
    'Initial Moisture Content (%)': 'Initial substrate', 'Initial C/N (%)': 'Initial substrate',
}

# Fixed display order and colours for feature-group annotations. Keeping this
# mapping central avoids the same category changing colour across figures.
FEATURE_GROUP_ORDER = [
    "Additive",
    "Aeration",
    "Feedstock",
    "Initial substrate",
    "Management practice",
    "Process design",
]
FEATURE_GROUP_COLORS = {
    "Additive": "#D58D89",
    "Aeration": "#78BFC2",
    "Feedstock": "#83B893",
    "Initial substrate": "#8588C2",
    "Management practice": "#E2B36E",
    "Process design": "#86A9CC",
}

TARGET_ORDER = ["NH3-N loss", "N2O-N loss", "CH4-C loss", "CO2-C loss", "TC loss", "TN loss", "Final GI"]
TARGET_LABELS = {
    "NH3-N loss": r"NH$_3$-N loss",
    "N2O-N loss": r"N$_2$O-N loss",
    "CH4-C loss": r"CH$_4$-C loss",
    "CO2-C loss": r"CO$_2$-C loss",
    "TC loss": "TC loss",
    "TN loss": "TN loss",
    "Final GI": "Final GI",
}
# Unified low-saturation Nature_new palette derived from the author's reference swatches.
# Warm rose tones denote nitrogen endpoints, cool blue/lilac tones denote carbon/GHG
# endpoints, and muted green denotes product quality.
TARGET_COLORS = {
    "NH3-N loss": "#C97979",
    "N2O-N loss": "#D78FB5",
    "TN loss":    "#A96F8F",
    "CH4-C loss": "#777FBC",
    "CO2-C loss": "#72B7BA",
    "TC loss":    "#7D9FC4",
    "Final GI":   "#83AE8F",
}
# direction: 1 if a HIGHER value is BETTER (desirable), -1 if higher value is WORSE
TARGET_DESIRABLE_DIRECTION = {
    "NH3-N loss": -1, "N2O-N loss": -1, "CH4-C loss": -1, "CO2-C loss": -1,
    "TC loss": -1, "TN loss": -1, "Final GI": 1,
}

# Original reference swatches plus darker derivatives for marks on white backgrounds.
NATURE_NEW_SOURCE = [
    "#DF9E9B", "#99BADF", "#D8E7CA", "#99CDCE", "#999ACD",
    "#FFD0E9", "#F6D58B", "#C7A3CC", "#B7B7B7",
]
PALETTE = [
    "#D58D89", "#78BFC2", "#83B893", "#8588C2", "#E2B36E",
    "#86A9CC", "#D78FB5", "#A96F8F", "#72B7BA", "#9B9B9B",
]

# Shared colormaps so every figure reads consistently:
#  - diverging correlation / similarity matrices  -> CMAP_DIVERGE
#  - sequential magnitude heatmaps (R2, importance) -> CMAP_SEQ
#  - continuous covariate colouring in scatter plots -> CMAP_CONT
CMAP_DIVERGE = LinearSegmentedColormap.from_list(
    "NatureNew_diverging", ["#6F8FB8", "#F6F4EE", "#C97979"], N=256
)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "NatureNew_sequential", ["#FAF3CB", "#D8E7CA", "#99CDCE", "#99BADF", "#777FBC"], N=256
)
CMAP_CONT = LinearSegmentedColormap.from_list(
    "NatureNew_continuous", ["#F6D58B", "#D8E7CA", "#99CDCE", "#86A9CC", "#777FBC"], N=256
)
