import logging

# Configure basic logging as soon as the package is imported
# This ensures logs are captured even if initialize_resources isn't called explicitly first
# Note: This sets up a basic handler. initialize_resources can refine it (e.g., add file logging).
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Get the root logger for the package
log = logging.getLogger(__name__)
log.info("Seawater Analyzer package initializing.")
# Optionally add a NullHandler to prevent "No handler found" warnings if no
# configuration is performed by the user application.
# from logging import NullHandler
# log.addHandler(NullHandler())


# Expose key classes and functions for easier import
from .exceptions import *
from .resource_manager import ResourceManager, initialize_resources
from .simulator import PhreeqcSimulator
from .processor import DataProcessor, validate_input, process_results, export_results, calculate_ca_hardness # Expose calculation too
from .plotter import PlotManager, generate_plots # Expose main plotter func too

__version__ = "0.1.0" # Example version

# Optionally define __all__ to control `from seawater_analyzer import *`
__all__ = [
    # Exceptions
    'SeawaterAnalyzerError', 'ResourceManagerError', 'PhreeqcSimulationError',
    'DataProcessingError', 'PlottingError', 'InputValidationError',
    # Classes
    'ResourceManager', 'PhreeqcSimulator', 'DataProcessor', 'PlotManager',
    # Functions
    'initialize_resources', 'validate_input', 'process_results', 'export_results',
    'calculate_ca_hardness', 'generate_plots'
]