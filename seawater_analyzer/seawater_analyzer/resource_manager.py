import logging
from pathlib import Path
import platform
import shutil # For copying files if needed

from . import config
from . import utils
from .exceptions import ResourceManagerError

logger = logging.getLogger(__name__)

class ResourceManager:
    """Handles management of external resources like PHREEQC library and databases."""

    def __init__(self,
                 database_dir: Path = config.DEFAULT_DATABASE_DIR,
                 output_dir: Path = config.OUTPUT_DIR,
                 log_file: Path = config.LOG_FILE):
        """
        Initializes the ResourceManager.

        Args:
            database_dir (Path): Directory to store downloaded database files.
            output_dir (Path): Base directory for storing outputs (reports, plots, etc.).
            log_file (Path): Path to the log file.
        """
        self.database_dir = Path(database_dir)
        self.output_dir = Path(output_dir)
        self.log_file = Path(log_file)
        self._phreeqc_lib_path = None

        self._setup_directories()
        # Note: Logging setup might be called externally once
        # utils.setup_logger(self.log_file)

    def _setup_directories(self):
        """Creates necessary directories if they don't exist."""
        try:
            self.database_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
            config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
            config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Required directories ensured.")
        except OSError as e:
            logger.error(f"Error creating directories: {e}")
            raise ResourceManagerError(f"Could not create necessary directories: {e}") from e

    def get_phreeqc_lib_path(self, force_download=False) -> Path:
        """
        Gets the path to the appropriate IPhreeqc library for the current OS.
        Downloads the library if it's not found or if force_download is True.

        Args:
            force_download (bool): If True, forces download even if file exists.

        Returns:
            Path: The absolute path to the IPhreeqc library file.

        Raises:
            ResourceManagerError: If the OS is unsupported or download fails.
        """
        if self._phreeqc_lib_path and self._phreeqc_lib_path.exists() and not force_download:
            return self._phreeqc_lib_path

        system = utils.get_platform_system()
        lib_filename = config.PHREEQC_LIB_FILENAMES.get(system)
        lib_url = config.PHREEQC_LIB_URLS.get(system)

        if not lib_filename or not lib_url:
            raise ResourceManagerError(f"PHREEQC library URL/filename not configured for system: {system}")

        # Store library alongside databases for simplicity, or choose another location
        lib_destination = self.database_dir / lib_filename

        if not lib_destination.exists() or force_download:
            logger.info(f"PHREEQC library not found or download forced. Downloading from {lib_url}...")
            try:
                 # Use a temporary name during download
                temp_dest = lib_destination.with_suffix(lib_destination.suffix + '.tmp')
                utils.download_file(lib_url, temp_dest)
                 # Rename to final name after successful download
                shutil.move(str(temp_dest), str(lib_destination))
                logger.info(f"PHREEQC library saved to {lib_destination}")
            except (ResourceManagerError, OSError) as e:
                logger.error(f"Failed to obtain PHREEQC library: {e}")
                raise ResourceManagerError("PHREEQC library download/move failed.") from e
        else:
            logger.info(f"Using existing PHREEQC library found at {lib_destination}")

        # Make executable on Linux/Mac if needed (best effort)
        if system in ["linux", "darwin"]:
             try:
                 lib_destination.chmod(lib_destination.stat().st_mode | 0o111) # Add execute permission
             except OSError:
                 logger.warning(f"Could not set execute permission on {lib_destination}. Simulation might fail.")


        self._phreeqc_lib_path = lib_destination.resolve() # Store absolute path
        return self._phreeqc_lib_path

    def get_database_path(self, db_name: str, force_download=False) -> Path:
        """
        Gets the path to a specific PHREEQC database file. Downloads if needed.

        Args:
            db_name (str): The filename of the database (e.g., "llnl.dat").
            force_download (bool): If True, forces download even if file exists.

        Returns:
            Path: The absolute path to the database file.

        Raises:
            ResourceManagerError: If the download fails.
        """
        if not db_name:
            raise ValueError("Database name cannot be empty.")

        db_path = self.database_dir / db_name
        db_url = config.DEFAULT_PHREEQC_DB_URL_BASE + db_name

        if not db_path.exists() or force_download:
            logger.info(f"Database '{db_name}' not found or download forced. Downloading from {db_url}...")
            try:
                temp_dest = db_path.with_suffix(db_path.suffix + '.tmp')
                utils.download_file(db_url, temp_dest)
                shutil.move(str(temp_dest), str(db_path))
                logger.info(f"Database '{db_name}' saved to {db_path}")
            except (ResourceManagerError, OSError) as e:
                logger.error(f"Failed to download/move database '{db_name}': {e}")
                raise ResourceManagerError(f"Database '{db_name}' download/move failed.") from e
        else:
            logger.info(f"Using existing database found at {db_path}")

        return db_path.resolve()

    def get_output_path(self, category: str, filename: str) -> Path:
        """
        Constructs a full path for an output file within the configured directories.

        Args:
            category (str): The type of output ('report', 'plot', 'raw', 'processed').
            filename (str): The desired base filename (e.g., "experiment_results.xlsx").

        Returns:
            Path: The full path for the output file.

        Raises:
            ValueError: If the category is invalid.
        """
        category_map = {
            'report': config.REPORT_DIR,
            'plot': config.PLOT_DIR,
            'raw': config.RAW_DATA_DIR,
            'processed': config.PROCESSED_DATA_DIR
        }
        if category.lower() not in category_map:
            raise ValueError(f"Invalid output category '{category}'. Must be one of {list(category_map.keys())}")

        target_dir = category_map[category.lower()]
        target_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
        return target_dir / filename

# Standalone function to initialize resources easily
def initialize_resources(data_dir=None, log_level=logging.INFO) -> ResourceManager:
    """
    Initializes the ResourceManager, sets up logging, and downloads core resources.

    Args:
        data_dir (str or Path, optional): Custom base directory for data. Defaults to config.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        ResourceManager: An initialized ResourceManager instance.
    """
    if data_dir:
        data_path = Path(data_dir)
        db_dir = data_path / "databases"
        out_dir = data_path / "outputs"
        log_f = data_path / "logs" / "seawater_analyzer.log"
    else:
        db_dir = config.DEFAULT_DATABASE_DIR
        out_dir = config.OUTPUT_DIR
        log_f = config.LOG_FILE

    utils.setup_logger(log_f, level=log_level)
    logger.info("Initializing Seawater Analyzer Resources...")

    rm = ResourceManager(database_dir=db_dir, output_dir=out_dir, log_file=log_f)
    try:
        rm.get_phreeqc_lib_path() # Ensure library is available
        rm.get_database_path(config.DEFAULT_PHREEQC_DB) # Ensure default db is available
        logger.info("Core resources checked/downloaded.")
    except ResourceManagerError as e:
        logger.critical(f"Failed to initialize core resources: {e}", exc_info=True)
        # Depending on severity, you might want to re-raise or exit
        raise # Re-raise the critical error

    return rm