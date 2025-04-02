import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from typing import Dict, Any, Optional

# Import backend components carefully to avoid circular dependencies if worker needs GUI elements (it shouldn't)
from seawater_analyzer import PhreeqcSimulator, DataProcessor, PhreeqcSimulationError, DataProcessingError, InputValidationError

class SimulationWorker(QObject):
    """Worker object to run PHREEQC simulations in a separate thread."""

    # Signals emitted by the worker
    simulation_success = Signal(pd.DataFrame, pd.DataFrame) # raw_df, processed_df
    simulation_error = Signal(str) # Error message
    status_update = Signal(str) # Progress messages

    def __init__(self, simulator: PhreeqcSimulator, processor: DataProcessor, parent=None):
        super().__init__(parent)
        self.simulator = simulator
        self.processor = processor
        self._is_running = False

    @Slot(dict, str) # Decorator specifies the types expected by the slot
    def run_simulation_task(self, sim_params: Dict[str, Any], db_name: Optional[str] = None):
        """The task to be executed in the background thread."""
        if self._is_running:
            self.status_update.emit("Simulation already in progress.")
            return

        self._is_running = True
        raw_results_df = pd.DataFrame()
        processed_df = pd.DataFrame()

        try:
            self.status_update.emit("Validating input parameters...")
            # Validation should ideally happen *before* starting the thread,
            # but can be done here as a double check or if params are complex
            self.processor.validate_input(sim_params)

            self.status_update.emit(f"Running PHREEQC simulation (Database: {db_name or 'default'})...")
            raw_results_df = self.simulator.run_simulation(sim_params, db_name=db_name)

            if raw_results_df.empty:
                self.status_update.emit("Simulation completed but produced no output data.")
                # Emit success with empty dataframes or a specific signal?
                # Let's emit success, GUI can check if empty.
            else:
                self.status_update.emit("Simulation successful. Processing results...")
                processed_df = self.processor.process_simulation_results(raw_results_df, sim_params)
                self.status_update.emit("Results processed.")

            # Emit the results (even if empty)
            self.simulation_success.emit(raw_results_df, processed_df)

        except InputValidationError as e:
            error_msg = f"Input Validation Error: {e}"
            self.status_update.emit(error_msg)
            self.simulation_error.emit(error_msg)
        except PhreeqcSimulationError as e:
            error_msg = f"PHREEQC Simulation Error: {e}"
            self.status_update.emit(error_msg)
            self.simulation_error.emit(error_msg)
        except DataProcessingError as e:
             error_msg = f"Data Processing Error: {e}"
             self.status_update.emit(error_msg)
             self.simulation_error.emit(error_msg)
        except Exception as e:
            # Catch any other unexpected errors
            error_msg = f"An unexpected error occurred: {e}"
            self.status_update.emit(error_msg)
            self.simulation_error.emit(f"Unexpected Error: {e}")
        finally:
            self._is_running = False
            self.status_update.emit("Simulation task finished.")