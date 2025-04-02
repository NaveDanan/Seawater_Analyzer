import os
from pathlib import Path

# --- Base Paths ---
# Use Pathlib for better path management
BASE_DIR = Path(__file__).resolve().parent.parent # Project root directory
DEFAULT_DATA_DIR = BASE_DIR / "data"

# --- Resource Manager Config ---
PHREEQC_LIB_URLS = {
    "linux": "https://github.com/rispr/phreeqc_web/blob/main/Iphreeqc_compiled/libiphreeqc-3.7.3.so?raw=true",
    "windows": "URL_TO_WINDOWS_DLL", # Replace with actual URL if available
    "darwin": "URL_TO_MACOS_DYLIB", # Replace with actual URL if available
}
# Simplified name mapping for downloaded lib
PHREEQC_LIB_FILENAMES = {
    "linux": "libiphreeqc.so",
    "windows": "libiphreeqc.dll",
    "darwin": "libiphreeqc.dylib",
}
DEFAULT_PHREEQC_DB_URL_BASE = "https://raw.githubusercontent.com/rispr/phreeqc_web/main/database/"
DEFAULT_DATABASE_DIR = DEFAULT_DATA_DIR / "databases"
LOG_FILE = DEFAULT_DATA_DIR / "logs" / "seawater_analyzer.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- Output Paths ---
OUTPUT_DIR = DEFAULT_DATA_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
PLOT_DIR = OUTPUT_DIR / "plots"
RAW_DATA_DIR = OUTPUT_DIR / "raw_data"
PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_data"

# --- Simulation Defaults ---
DEFAULT_PHREEQC_DB = "llnl.dat" # Default database
DEFAULT_TEMPERATURE = 25.0
DEFAULT_PH = 8.0
DEFAULT_PE = 4.0
DEFAULT_DENSITY = 1.025 # kg/L (typical seawater)

# --- Plotting Defaults ---
DEFAULT_PLOT_STYLE = {
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0
}
DEFAULT_PLOT_FORMAT = "png"

# --- Data Processing ---
# Example standard seawater composition (approximate values in mg/L or mg/kg)
# Source: e.g., Millero, Chemical Oceanography
STANDARD_SEAWATER_COMPOSITION = {
    "Cl": 19353,
    "Na": 10760,
    "Mg": 1294,
    "S": 905,  # As S, calculate SO4 = S * (96.06 / 32.06) = 2710 mg/L
    "K": 399,
    "Ca": 412,
    "Br": 67,
    "B": 4.4,
    "Sr": 8.1,
    # Note: Alkalinity is usually measured, not fixed
}
# Molar masses needed for conversions (g/mol)
MOLAR_MASSES = {
    "Ca": 40.08,
    "Mg": 24.305,
    "C": 12.011,
    "O": 15.999,
    "H": 1.008,
    # Add others as needed
}