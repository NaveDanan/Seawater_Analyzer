# Seawater Composition Analysis and Simulation Backend

This Python package provides backend tools for processing experimental seawater data, performing geochemical simulations using PHREEQC, analyzing results, and generating visualizations and reports.

## Features

*   **Resource Management:** Handles downloading and caching of PHREEQC libraries and databases.
*   **PHREEQC Integration:** Wrapper around `phreeqpy` for running simulations.
*   **Data Processing:** Includes functions for common seawater calculations (e.g., Ca hardness), data validation, and unit conversions (basic).
*   **Visualization:** Generates summary plots, time series, scatter plots, etc., using Matplotlib/Seaborn.
*   **Input/Output:** Supports reading inputs and exporting results to formats like Excel, CSV, JSON.
*   **Error Handling:** Comprehensive error checking and logging.
*   **Modularity:** Components designed for independent use or integration into larger applications (e.g., GUIs, web services, notebooks).

## Installation

```bash
pip install -r requirements.txt
pip install .
# Or for development:
pip install -e .