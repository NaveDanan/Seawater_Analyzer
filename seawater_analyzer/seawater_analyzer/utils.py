import platform
import logging
import requests
from pathlib import Path
from . import config
from .exceptions import ResourceManagerError

def get_platform_system():
    """Returns the simplified platform name ('linux', 'windows', 'darwin')."""
    system = platform.system().lower()
    if "linux" in system:
        return "linux"
    elif "windows" in system:
        return "windows"
    elif "darwin" in system: # macOS
        return "darwin"
    else:
        raise ResourceManagerError(f"Unsupported operating system: {system}")

def download_file(url: str, destination: Path, chunk_size=8192):
    """Downloads a file from a URL to a destination path."""
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        logging.info(f"Successfully downloaded {url} to {destination}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        # Optionally remove partially downloaded file
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                logging.warning(f"Could not remove partial download: {destination}")
        raise ResourceManagerError(f"Download failed for {url}") from e
    except OSError as e:
        logging.error(f"Failed to write downloaded file to {destination}: {e}")
        raise ResourceManagerError(f"File write error for {destination}") from e

def setup_logger(log_file: Path = config.LOG_FILE, level=logging.INFO):
    """Configures the root logger for the application."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also print logs to console
        ]
    )
    logging.info("Logger initialized.")
    return logging.getLogger(__name__) # Return a logger instance for the module

# Example of a simple caching decorator (can be expanded)
def simple_cache(func):
    """Very basic caching decorator based on function arguments."""
    cache = {}
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper