import sys
import logging
from PySide6.QtWidgets import QApplication

# Import backend init function and main window
from seawater_analyzer import initialize_resources, ResourceManagerError
from .main_window import MainWindow

# Configure logging early (backend's __init__ does basic setup,
# initialize_resources refines it with file logging)
# The MainWindow will add its own handler for the GUI log widget

def run_gui():
    """Initializes backend and runs the PySide6 application."""

    resource_manager = None
    try:
        # Initialize backend resources (downloads libs/dbs if needed)
        # Set log level here if desired
        resource_manager = initialize_resources(log_level=logging.INFO)
        logging.info("Backend resources initialized successfully.")

    except ResourceManagerError as e:
        logging.critical(f"FATAL: Failed to initialize critical backend resources: {e}", exc_info=True)
        # Show a simple message box if possible, as GUI isn't fully up yet
        app_temp = QApplication.instance() # Check if already exists
        if not app_temp:
             app_temp = QApplication(sys.argv)
        # Need to import QMessageBox here directly
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Initialization Error",
                             f"Failed to initialize application resources:\n{e}\n\n"
                             "Check internet connection and file permissions.\n"
                             "See log file for details.")
        sys.exit(1) # Exit if critical resources fail
    except Exception as e:
         logging.critical(f"FATAL: Unexpected error during initialization: {e}", exc_info=True)
         app_temp = QApplication.instance()
         if not app_temp:
             app_temp = QApplication(sys.argv)
         from PySide6.QtWidgets import QMessageBox
         QMessageBox.critical(None, "Initialization Error",
                              f"An unexpected error occurred during startup:\n{e}")
         sys.exit(1)


    # --- Run the GUI Application ---
    app = QApplication(sys.argv)
    # Apply any global styling here if desired (e.g., Fusion style)
    # app.setStyle("Fusion")

    main_window = MainWindow(resource_manager)
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()