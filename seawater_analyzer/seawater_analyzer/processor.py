import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

from . import config
from .exceptions import DataProcessingError, InputValidationError

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles processing, validation, and formatting of seawater data."""

    def __init__(self):
        """Initializes the DataProcessor."""
        # Can add configuration options here if needed
        pass

    def validate_input(self, data: Dict[str, Any], required_fields: Optional[list] = None) -> bool:
        """
        Validates the input data dictionary for required fields and basic types/ranges.

        Args:
            data (Dict[str, Any]): Dictionary containing input data (e.g., measurements, params).
            required_fields (Optional[list]): List of keys that must be present in the data.
                                             If None, performs basic checks on common fields.

        Returns:
            bool: True if validation passes.

        Raises:
            InputValidationError: If validation fails, containing details.
        """
        logger.debug(f"Validating input data: {data}")
        errors = []

        # Check for required fields if provided
        if required_fields:
            for field in required_fields:
                if field not in data or data[field] is None:
                    errors.append(f"Missing required input field: '{field}'.")

        # Basic type/range checks for common parameters (can be expanded)
        if 'pH' in data and not isinstance(data['pH'], (int, float)):
             errors.append(f"Invalid type for 'pH': Expected number, got {type(data['pH'])}.")
        elif 'pH' in data and not (0 <= data['pH'] <= 14):
             errors.append(f"pH value {data['pH']} is outside the typical range [0, 14].")

        if 'temp' in data and not isinstance(data['temp'], (int, float)):
             errors.append(f"Invalid type for 'temp': Expected number, got {type(data['temp'])}.")
        # Add reasonable temp range check if needed (e.g., -2 to 40 C for seawater)

        if 'alkalinity' in data and data['alkalinity'] is not None and not isinstance(data['alkalinity'], (int, float)):
             errors.append(f"Invalid type for 'alkalinity': Expected number or None, got {type(data['alkalinity'])}.")
        elif 'alkalinity' in data and data['alkalinity'] is not None and data['alkalinity'] < 0:
            errors.append("Alkalinity cannot be negative.")

        # Check composition dictionary if present
        if 'composition' in data:
            if not isinstance(data['composition'], dict):
                errors.append("Invalid type for 'composition': Expected dictionary.")
            else:
                for ion, conc in data['composition'].items():
                    if conc is not None and not isinstance(conc, (int, float)):
                         errors.append(f"Invalid concentration type for '{ion}': Expected number or None, got {type(conc)}.")
                    elif conc is not None and conc < 0:
                         errors.append(f"Negative concentration provided for '{ion}': {conc}.")

        if errors:
            error_message = "Input validation failed:\n" + "\n".join(errors)
            logger.error(error_message)
            raise InputValidationError(error_message)

        logger.info("Input data validation successful.")
        return True

    def calculate_ca_hardness(self, titrant_volume_A: float, sample_volume: float = 50.0, dilution_factor: float = 0.04, normality_B: float = 1.0) -> float:
        """
        Calculates Calcium hardness (as mg/L CaCO3 equivalent, then convert to mg/L Ca).
        Based on the formula: Ca hardness (mg/L CaCO3) = (A * B * 400.8) / (sample_volume * dilution)
        Then convert CaCO3 eq to Ca: mg/L Ca = mg/L CaCO3 * (MolarMass(Ca) / MolarMass(CaCO3))
        Molar Mass CaCO3 approx 100.09 g/mol. Molar Mass Ca = 40.08 g/mol. Ratio ~ 0.4004

        Args:
            titrant_volume_A (float): Volume of EDTA titrant used (mL).
            sample_volume (float): Total volume of the *diluted* sample titrated (mL). Default 50.
            dilution_factor (float): Fraction of original sample in the diluted solution
                                    (e.g., 2mL original in 50mL total -> 2/50 = 0.04). Default 0.04.
            normality_B (float): Normality of the EDTA titrant (often standardized, ~1.0). Default 1.0.
                                 Note: The formula seems to assume N=1 or incorporates it somehow.
                                 Let's strictly follow the reference script's formula interpretation first.
                                 The 400.8 factor likely combines molar mass and volume factors (e.g., 100.09 * 1000 / 250 if B=0.01M EDTA?).
                                 **Revisiting the formula from the script:**
                                 `ca_hardness = (A * B * 400.8) / (volume * dilution)`
                                 This formula *directly* gives mg/L Ca. The factor 400.8 likely represents
                                 (MolarMass_Ca * 1000 / Normality_factor). Let's assume it's correct as given.
                                 It calculates concentration in the *original* sample.

        Returns:
            float: Calcium concentration in the original sample (mg/L as Ca).

        Raises:
            DataProcessingError: If inputs are invalid.
        """
        logger.debug(f"Calculating Ca hardness with A={titrant_volume_A}, vol={sample_volume}, dil={dilution_factor}, B={normality_B}")

        if sample_volume <= 0:
            raise DataProcessingError("Total volume of diluted sample must be greater than zero.")
        if not (0 < dilution_factor <= 1):
            raise DataProcessingError("Dilution factor must be between 0 (exclusive) and 1 (inclusive).")
        if titrant_volume_A < 0:
            raise DataProcessingError("Titrant volume (A) cannot be negative.")
        if normality_B <= 0:
             raise DataProcessingError("Normality (B) must be positive.") # Added check

        # Applying the formula from the reference script directly
        # This calculates concentration in the *original* undiluted sample
        try:
            ca_concentration_mg_L = (titrant_volume_A * normality_B * 400.8) / (sample_volume * dilution_factor)
        except ZeroDivisionError:
            # Should be caught by checks above, but just in case
            raise DataProcessingError("Division by zero encountered in hardness calculation. Check volume and dilution.")

        logger.info(f"Calculated Ca concentration: {ca_concentration_mg_L:.2f} mg/L")
        return ca_concentration_mg_L

    def convert_concentration(self, value: float, input_unit: str, output_unit: str, element: Optional[str] = None, density: float = 1.0) -> float:
        """
        Converts concentration between different units (e.g., mg/L, ppm, mol/kg, meq/L).
        (This is a placeholder - requires careful implementation based on specific needs).

        Args:
            value (float): The concentration value to convert.
            input_unit (str): The unit of the input value (e.g., 'mg/L', 'mmol/kgw', 'ppm').
            output_unit (str): The desired output unit.
            element (Optional[str]): The element/ion name (needed for molar mass conversions, e.g., 'Ca', 'Mg').
            density (float): Density of the solution (kg/L), needed for mass/volume conversions. Default 1.0.

        Returns:
            float: The converted concentration value.

        Raises:
            DataProcessingError: If conversion is not supported or element info is missing.
            NotImplementedError: If a specific conversion path isn't implemented yet.
        """
        logger.debug(f"Attempting conversion: {value} {input_unit} -> {output_unit} (Element: {element}, Density: {density})")

        if input_unit == output_unit:
            return value

        # --- Add specific conversion logic here ---
        # Example: mg/L to mmol/kgw for Ca
        if input_unit == 'mg/L' and output_unit == 'mmol/kgw' and element == 'Ca':
            if not element: raise DataProcessingError("Element name required for molar conversion.")
            molar_mass = config.MOLAR_MASSES.get(element)
            if not molar_mass: raise DataProcessingError(f"Molar mass for element '{element}' not found.")
            # mg/L -> g/L -> mol/L -> mol/kg_solution -> mol/kgw (approx if density used)
            mol_L = (value / 1000) / molar_mass
            mol_kg_solution = mol_L / density
            # This assumes mol/kg_solution is close enough to mol/kgw for dilute solutions
            # A more precise conversion requires knowing water mass fraction
            mmol_kgw = mol_kg_solution * 1000
            return mmol_kgw

        # Example: mmol/kgw to mg/L for Ca
        elif input_unit == 'mmol/kgw' and output_unit == 'mg/L' and element == 'Ca':
            if not element: raise DataProcessingError("Element name required for molar conversion.")
            molar_mass = config.MOLAR_MASSES.get(element)
            if not molar_mass: raise DataProcessingError(f"Molar mass for element '{element}' not found.")
            # mmol/kgw -> mol/kgw -> mol/kg_solution -> mol/L -> g/L -> mg/L
            mol_kgw = value / 1000
            mol_kg_solution = mol_kgw # Approximate, see above
            mol_L = mol_kg_solution * density
            g_L = mol_L * molar_mass
            mg_L = g_L * 1000
            return mg_L

        # Add other conversions (ppm, meq/L, etc.) as needed
        else:
            raise NotImplementedError(f"Conversion from '{input_unit}' to '{output_unit}' for element '{element}' is not implemented.")

        # Placeholder return, should be replaced by actual conversion logic
        # return value

    def process_simulation_results(self, results_df: pd.DataFrame, input_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Performs post-processing on the DataFrame returned by the PhreeqcSimulator.
        Calculates derived values, formats columns, adds metadata.

        Args:
            results_df (pd.DataFrame): The raw results DataFrame from PhreeqcSimulator.
            input_params (Dict[str, Any]): The original input parameters used for the simulation
                                           (might be needed for context or unit conversions).

        Returns:
            pd.DataFrame: The processed and enhanced DataFrame.

        Raises:
            DataProcessingError: If expected columns are missing or calculations fail.
        """
        if results_df.empty:
            logger.warning("Received empty DataFrame for processing. Returning empty.")
            return results_df

        processed_df = results_df.copy()
        logger.debug(f"Processing simulation results. Input columns: {list(processed_df.columns)}")

        # --- Calculate Derived Values (examples) ---

        # DIC (Dissolved Inorganic Carbon) in mM
        # Assumes 'C(4)' total is in output (mol/kgw) and density ('rho(kg/L)') is available
        dic_col_name = next((col for col in processed_df.columns if 'C(4)' in col), None) # Find C(4) total column
        rho_col_name = next((col for col in processed_df.columns if 'rho' in col.lower()), 'rho(kg/L)') # Find density, default name

        if dic_col_name and rho_col_name in processed_df.columns:
            try:
                # mol/kgw -> mol/kg_solution (approx) -> mol/L -> mmol/L (mM)
                mol_kgw = pd.to_numeric(processed_df[dic_col_name], errors='coerce')
                rho = pd.to_numeric(processed_df[rho_col_name], errors='coerce')
                # Handle potential NaNs from conversion or missing data
                mol_L = mol_kgw * rho
                processed_df['DIC_mM'] = mol_L * 1000
                logger.info("Calculated DIC_mM column.")
            except KeyError as e:
                logger.warning(f"Could not calculate DIC: Missing column {e}.")
            except Exception as e:
                logger.warning(f"Error calculating DIC: {e}", exc_info=True)
        else:
            logger.warning(f"Could not calculate DIC: Missing '{dic_col_name or 'C(4)'}' or '{rho_col_name}' column.")

        # pCO2 (partial pressure of CO2) in ppm
        # Assumes 'si_CO2(g)' or 'pCO2(atm)' is in output
        pco2_col_atm = next((col for col in processed_df.columns if 'pco2(atm)' in col.lower()), None)
        si_co2_col = next((col for col in processed_df.columns if 'si_co2(g)' in col.lower()), None)

        if pco2_col_atm:
             try:
                 pco2_atm = pd.to_numeric(processed_df[pco2_col_atm], errors='coerce')
                 processed_df['pCO2_ppm'] = pco2_atm * 1e6
                 logger.info("Calculated pCO2_ppm column from pCO2(atm).")
             except Exception as e:
                 logger.warning(f"Error calculating pCO2 from atm column: {e}", exc_info=True)

        elif si_co2_col:
            try:
                si_co2 = pd.to_numeric(processed_df[si_co2_col], errors='coerce')
                # pCO2 (atm) = 10 ^ SI(CO2(g))
                pco2_atm = 10**si_co2
                processed_df['pCO2_ppm'] = pco2_atm * 1e6 # Convert atm to ppm
                logger.info("Calculated pCO2_ppm column from si_CO2(g).")
            except KeyError:
                 logger.warning(f"Could not calculate pCO2: Missing column '{si_co2_col}'.")
            except Exception as e:
                logger.warning(f"Error calculating pCO2 from SI: {e}", exc_info=True)
        else:
            logger.warning("Could not calculate pCO2: No suitable SI or pCO2 column found.")

        # Rename columns for clarity if desired (example)
        rename_map = {
            'temp': 'Temperature_C',
            'Alk': 'Alkalinity_meq_kgw', # Check actual output name from PHREEQC
            'mu': 'Ionic_Strength',     # Check actual output name
            'si_Calcite': 'SI_Calcite', # Keep common ones simple
            'si_Aragonite': 'SI_Aragonite',
            # Add more renames based on typical PHREEQC output columns
        }
        # Only rename columns that actually exist in the DataFrame
        actual_rename_map = {k: v for k, v in rename_map.items() if k in processed_df.columns}
        if actual_rename_map:
             processed_df = processed_df.rename(columns=actual_rename_map)
             logger.debug(f"Renamed columns: {actual_rename_map}")

        # --- Add Metadata (example) ---
        # Could add input parameters back into the DataFrame if running single point simulations
        # For multi-step simulations (KINETICS, TRANSPORT), this might not make sense.
        # Example for single SOLUTION block result:
        if len(processed_df) == 1:
             for key, value in input_params.items():
                 if key not in processed_df.columns and key != 'composition': # Avoid overwriting, skip complex dicts
                     processed_df[f'input_{key}'] = value

        logger.info("Finished processing simulation results.")
        return processed_df

    def format_for_export(self, data: Union[pd.DataFrame, Dict], format_type: str = 'excel') -> Any:
        """
        Prepares data for export into a specific format (e.g., selecting columns, ordering).

        Args:
            data (Union[pd.DataFrame, Dict]): The data to format (usually processed results).
            format_type (str): The target format ('excel', 'csv', 'json').

        Returns:
            Any: The formatted data, ready for saving (e.g., a potentially modified DataFrame).
        """
        logger.info(f"Formatting data for {format_type} export.")

        if isinstance(data, dict):
            # Convert dict to DataFrame for easier handling, assuming simple structure
            try:
                df = pd.DataFrame([data]) # Create a single-row DataFrame
            except Exception as e:
                raise DataProcessingError(f"Could not convert input dict to DataFrame for formatting: {e}")
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input data must be a pandas DataFrame or a dictionary.")

        if df.empty:
            logger.warning("Cannot format empty data for export.")
            return df # Return empty DataFrame

        # --- Example Formatting Steps ---
        # 1. Select relevant columns for the report
        # 2. Order columns logically
        # 3. Round numerical data to appropriate precision

        # Define desired columns and order for an Excel report (example)
        if format_type == 'excel':
            desired_columns = [
                # Input conditions
                'input_Experiment_ID', 'input_Date', # Assuming these were added earlier
                'Temperature_C', 'pH',
                'input_Ca_measured', 'input_Alkalinity_measured', # Use names indicating source
                # Calculated outputs
                'Alkalinity_meq_kgw', # PHREEQC calculated
                'DIC_mM',
                'SI_Calcite', 'SI_Aragonite',
                 'pCO2_ppm',
                 'Ionic_Strength',
                 'charge_balance', # Useful diagnostic
                # Add other relevant columns from df.columns
            ]
            # Filter to columns that actually exist in the DataFrame
            export_columns = [col for col in desired_columns if col in df.columns]
            # Add any remaining columns from the original df not in the desired list (optional)
            # remaining_cols = [col for col in df.columns if col not in export_columns]
            # final_cols = export_columns + remaining_cols
            final_cols = export_columns # Keep it clean for the report

            formatted_df = df[final_cols].copy()

            # Round numeric columns (example)
            numeric_cols = formatted_df.select_dtypes(include=np.number).columns
            rounding_map = {
                 'Temperature_C': 1, 'pH': 2, 'input_Ca_measured': 1, 'input_Alkalinity_measured': 3,
                 'Alkalinity_meq_kgw': 4, 'DIC_mM': 3, 'SI_Calcite': 3, 'SI_Aragonite': 3,
                 'pCO2_ppm': 1, 'Ionic_Strength': 4, 'charge_balance': 5
            }
            for col in numeric_cols:
                if col in rounding_map:
                     formatted_df[col] = formatted_df[col].round(rounding_map[col])

            logger.debug(f"Formatted DataFrame for Excel export with columns: {list(formatted_df.columns)}")
            return formatted_df

        elif format_type == 'csv':
             # CSV might not need as much formatting, return as is or select columns
             logger.debug("Formatting for CSV (currently returns original DataFrame).")
             return df # Or apply column selection if needed

        elif format_type == 'json':
             logger.debug("Formatting for JSON (currently returns original DataFrame).")
             # JSON export often works directly from DataFrame, maybe orientation needs adjustment
             return df # Or convert to dict: df.to_dict(orient='records')

        else:
            logger.warning(f"Unsupported export format '{format_type}'. Returning original data.")
            return df

# Standalone function interfaces (matching request)
_processor_instance = DataProcessor() # Create a default instance

def validate_input(data):
    return _processor_instance.validate_input(data)

def process_results(raw_data, input_params=None):
    # Assume raw_data is the DataFrame from simulator
    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("process_results expects a pandas DataFrame as raw_data.")
    if input_params is None:
        input_params = {} # Provide empty dict if not given
    return _processor_instance.process_simulation_results(raw_data, input_params)

def export_results(data, format_type, output_path: Path):
    """
    Formats data and saves it to a file.

    Args:
        data (Union[pd.DataFrame, Dict]): Data to export.
        format_type (str): 'excel', 'csv', 'json'.
        output_path (Path): Full path to save the output file.
    """
    formatted_data = _processor_instance.format_for_export(data, format_type)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        if format_type == 'excel':
            if not isinstance(formatted_data, pd.DataFrame):
                 raise DataProcessingError("Formatted data is not a DataFrame for Excel export.")
            formatted_data.to_excel(output_path, index=False, engine='openpyxl')
        elif format_type == 'csv':
             if not isinstance(formatted_data, pd.DataFrame):
                 raise DataProcessingError("Formatted data is not a DataFrame for CSV export.")
             formatted_data.to_csv(output_path, index=False)
        elif format_type == 'json':
             # Decide on JSON format (e.g., records)
             if isinstance(formatted_data, pd.DataFrame):
                 formatted_data.to_json(output_path, orient='records', indent=4)
             elif isinstance(formatted_data, dict): # Allow exporting raw dicts too
                 import json
                 with open(output_path, 'w') as f:
                     json.dump(formatted_data, f, indent=4)
             else:
                 raise DataProcessingError("Formatted data is not DataFrame or dict for JSON export.")
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        logger.info(f"Successfully exported results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export results to {output_path}: {e}", exc_info=True)
        raise DataProcessingError(f"Export failed for {output_path}") from e