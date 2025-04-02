import logging
import pandas as pd
import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
from pathlib import Path
from typing import Dict, Any, Optional

from .resource_manager import ResourceManager
from .exceptions import PhreeqcSimulationError

logger = logging.getLogger(__name__)

# Basic PHREEQC input template (can be customized further)
DEFAULT_PQC_INPUT_TEMPLATE = """
SOLUTION 1
    title           Seawater Simulation
    temp            {temp}
    pH              {pH}
    pe              {pe}
    redox           pe
    units           mg/kgw # Or mg/L depending on how composition is provided
    density         {density} # kg/L
    # Alkalinity provided separately if available
    {alkalinity_line}
    # Major Ions - concentrations in mg/kgw (adjust if units differ)
    {ion_lines}
    -water          1 # kg

SELECTED_OUTPUT
    -file           simulation_output.sel # Define an output file name
    -reset          false
    -high_precision true
    # -simulation   true # Include simulation number
    # -state        true # Include solution number etc.
    # Basic properties
    -temperature    true
    -pH             true
    -pe             true
    -density        true
    -alkalinity     true # Calculated alkalinity
    -ionic_strength true
    -charge_balance true
    # Totals for key elements (mol/kgw)
    -totals         {total_elements}
    # Saturation indices for common minerals
    -saturation_indices {saturation_indices}

USER_PUNCH
    -headings       rho(kg/L) SC(uS/cm) pCO2(atm) # Add more as needed
    -start
    10 PUNCH RHO
    20 PUNCH SC
    30 PUNCH 10^SI("CO2(g)")
    -end
"""

class PhreeqcSimulator:
    """Wraps PHREEQC simulation execution using phreeqpy."""

    def __init__(self, resource_manager: ResourceManager):
        """
        Initializes the simulator.

        Args:
            resource_manager: An initialized ResourceManager instance.
        """
        self.rm = resource_manager
        self.phreeqc_lib_path = self.rm.get_phreeqc_lib_path()
        self.iphreeqc: Optional[phreeqc_mod.IPhreeqc] = None
        self.current_db_path: Optional[Path] = None

    def _initialize_iphreeqc(self):
        """Creates an instance of the IPhreeqc object."""
        if self.iphreeqc is None:
            try:
                logger.info(f"Initializing IPhreeqc using library: {self.phreeqc_lib_path}")
                self.iphreeqc = phreeqc_mod.IPhreeqc(str(self.phreeqc_lib_path))
                logger.info("IPhreeqc instance created successfully.")
            except Exception as e: # phreeqpy might raise various errors on load
                logger.error(f"Failed to initialize IPhreeqc: {e}", exc_info=True)
                raise PhreeqcSimulationError(f"IPhreeqc initialization failed: {e}") from e

    def load_database(self, db_name: str):
        """
        Loads a specified PHREEQC database file.

        Args:
            db_name (str): The name of the database file (e.g., "llnl.dat").
        """
        self._initialize_iphreeqc() # Ensure IPhreeqc exists
        db_path = self.rm.get_database_path(db_name)

        if db_path == self.current_db_path:
            logger.debug(f"Database '{db_name}' is already loaded.")
            return

        try:
            logger.info(f"Loading PHREEQC database: {db_path}")
            load_status = self.iphreeqc.load_database(str(db_path))
            if load_status != 0:
                 # Try to get error messages
                 errors = self.iphreeqc.get_error_string()
                 logger.error(f"PHREEQC failed to load database '{db_name}'. Status: {load_status}. Errors:\n{errors}")
                 raise PhreeqcSimulationError(f"Failed to load database '{db_name}'. PHREEQC errors:\n{errors}")
            self.current_db_path = db_path
            logger.info(f"Database '{db_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error during database loading '{db_name}': {e}", exc_info=True)
            self.current_db_path = None # Reset if loading failed
            raise PhreeqcSimulationError(f"Failed loading database '{db_name}': {e}") from e

    def _generate_input_script(self, params: Dict[str, Any]) -> str:
        """
        Generates the PHREEQC input string from a dictionary of parameters.

        Args:
            params (Dict[str, Any]): Dictionary containing simulation parameters.
                                     Expected keys: 'temp', 'pH', 'pe', 'density',
                                     'composition' (dict of ion: concentration),
                                     'alkalinity' (optional, in meq/kgw),
                                     'selected_elements' (list, e.g., ["Ca", "C(4)"]),
                                     'selected_phases' (list, e.g., ["Calcite", "Aragonite"]).

        Returns:
            str: The formatted PHREEQC input script.
        """
        # --- Validate required parameters ---
        required = ['temp', 'pH', 'pe', 'density', 'composition', 'selected_elements', 'selected_phases']
        for req in required:
            if req not in params:
                raise InputValidationError(f"Missing required simulation parameter: '{req}'")

        # --- Prepare components for the template ---
        comp = params['composition']
        ion_lines = "\n    ".join([f"{ion:<15} {conc}" for ion, conc in comp.items() if conc is not None])

        # Handle optional alkalinity
        alkalinity_line = ""
        if 'alkalinity' in params and params['alkalinity'] is not None:
             # Assuming alkalinity is provided in meq/kgw
             alkalinity_line = f"Alkalinity      {params['alkalinity']} as HCO3" # Or specify units if different

        total_elements_str = " ".join(params['selected_elements'])
        saturation_indices_str = " ".join(params['selected_phases'])

        # --- Format the template ---
        try:
            pqc_input = DEFAULT_PQC_INPUT_TEMPLATE.format(
                temp=params['temp'],
                pH=params['pH'],
                pe=params['pe'],
                density=params['density'],
                alkalinity_line=alkalinity_line,
                ion_lines=ion_lines,
                total_elements=total_elements_str,
                saturation_indices=saturation_indices_str
            )
            logger.debug(f"Generated PHREEQC input script:\n{pqc_input}")
            return pqc_input
        except KeyError as e:
             raise InputValidationError(f"Error formatting input script. Missing key: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error generating input script: {e}", exc_info=True)
             raise PhreeqcSimulationError("Failed to generate PHREEQC input script.") from e


    def _parse_results(self) -> pd.DataFrame:
        """
        Retrieves and parses the SELECTED_OUTPUT results into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the simulation results.

        Raises:
            PhreeqcSimulationError: If results retrieval or parsing fails.
        """
        if not self.iphreeqc:
             raise PhreeqcSimulationError("IPhreeqc instance not available for parsing results.")

        try:
            output = self.iphreeqc.get_selected_output_array()
            if not output or len(output) < 2: # Expect header row + at least one data row
                errors = self.iphreeqc.get_error_string()
                warnings = self.iphreeqc.get_warning_string()
                log_msg = f"PHREEQC simulation produced no usable output."
                if errors: log_msg += f"\nErrors:\n{errors}"
                if warnings: log_msg += f"\nWarnings:\n{warnings}"
                logger.warning(log_msg)
                # Return empty DataFrame or raise error? Let's return empty for now.
                # raise PhreeqcSimulationError("Simulation produced no usable output.")
                return pd.DataFrame()

            # First row is headers
            headers = [str(h).strip() for h in output[0]]
            # Subsequent rows are data
            data = output[1:]

            df = pd.DataFrame(data, columns=headers)

            # Attempt to convert columns to numeric where possible
            for col in df.columns:
                try:
                    # Use pd.to_numeric with errors='ignore' to skip non-numeric columns
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except ValueError:
                     logger.warning(f"Could not convert column '{col}' to numeric. Keeping as object.")
                     pass # Keep column as object type if conversion fails entirely


            logger.info(f"Successfully parsed simulation results into DataFrame with shape {df.shape}.")
            logger.debug(f"Result DataFrame head:\n{df.head()}")
            return df

        except Exception as e:
            logger.error(f"Failed to parse PHREEQC results: {e}", exc_info=True)
            raise PhreeqcSimulationError("Error parsing simulation results.") from e

    def run_simulation(self, params: Dict[str, Any], db_name: Optional[str] = None) -> pd.DataFrame:
        """
        Runs a PHREEQC simulation with the given parameters and database.

        Args:
            params (Dict[str, Any]): Dictionary of simulation parameters (see _generate_input_script).
            db_name (Optional[str]): Name of the database file to use. If None, uses the
                                     last loaded database or the default.

        Returns:
            pd.DataFrame: DataFrame containing the parsed simulation results.

        Raises:
            PhreeqcSimulationError: If simulation setup or execution fails.
            InputValidationError: If input parameters are invalid.
        """
        self._initialize_iphreeqc()

        if db_name:
            self.load_database(db_name)
        elif not self.current_db_path:
            # Load default if no database specified and none loaded yet
            self.load_database(config.DEFAULT_PHREEQC_DB)
        # Else: use the already loaded database

        pqc_input = self._generate_input_script(params)

        # Clear previous results/errors before running
        self.iphreeqc.clear_accumulator()
        self.iphreeqc.set_output_file_on(False) # We parse from memory using get_selected_output_array
        self.iphreeqc.set_error_file_on(False)
        self.iphreeqc.set_log_file_on(False)
        self.iphreeqc.set_selected_output_file_on(False) # Important!

        logger.info("Running PHREEQC simulation...")
        try:
            run_status = self.iphreeqc.run_string(pqc_input)
            errors = self.iphreeqc.get_error_string()
            warnings = self.iphreeqc.get_warning_string()

            if run_status != 0 or errors:
                 log_msg = f"PHREEQC simulation finished with errors. Status: {run_status}."
                 if errors: log_msg += f"\nErrors:\n{errors}"
                 if warnings: log_msg += f"\nWarnings:\n{warnings}"
                 logger.error(log_msg)
                 # Raise error even if run_status is 0 but errors exist
                 raise PhreeqcSimulationError(f"PHREEQC simulation failed. Errors:\n{errors or 'Unknown'}")

            if warnings:
                 logger.warning(f"PHREEQC simulation finished with warnings:\n{warnings}")

            logger.info("PHREEQC simulation completed.")

        except Exception as e:
             # Catch potential errors within run_string itself or re-raise from above
             logger.error(f"Exception during PHREEQC run: {e}", exc_info=True)
             if not isinstance(e, PhreeqcSimulationError): # Avoid wrapping our own error type
                 raise PhreeqcSimulationError(f"Simulation execution failed: {e}") from e
             else:
                 raise e # Re-raise the specific simulation error

        # Parse results after successful run
        results_df = self._parse_results()

        # Optional: Save raw output string?
        # raw_output_path = self.rm.get_output_path('raw', f"simulation_{timestamp}.pqc_out")
        # with open(raw_output_path, 'w') as f:
        #     f.write(self.iphreeqc.get_dump_string()) # Or manage output files differently

        return results_df