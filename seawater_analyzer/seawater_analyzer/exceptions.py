"""Custom exceptions for the seawater_analyzer package."""

class SeawaterAnalyzerError(Exception):
    """Base exception for this package."""
    pass

class ResourceManagerError(SeawaterAnalyzerError):
    """Exceptions related to resource management (downloads, paths)."""
    pass

class PhreeqcSimulationError(SeawaterAnalyzerError):
    """Exceptions related to PHREEQC simulation execution or setup."""
    pass

class DataProcessingError(SeawaterAnalyzerError):
    """Exceptions related to data validation, calculation, or formatting."""
    pass

class PlottingError(SeawaterAnalyzerError):
    """Exceptions related to plot generation or export."""
    pass

class InputValidationError(DataProcessingError):
    """Specific exception for input data validation failures."""
    pass