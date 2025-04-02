import logging
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QDoubleSpinBox, QDateEdit, QTextEdit,
    QTableView, QSplitter, QMessageBox, QFileDialog, QProgressBar, QTabWidget,
    QSizePolicy, QAbstractItemView, QHeaderView, QGroupBox
)
from PySide6.QtCore import Qt, QThread, Slot, QDate
from PySide6.QtGui import QAction, QKeySequence

# Import backend components
from seawater_analyzer import (
    ResourceManager, PhreeqcSimulator, DataProcessor, PlotManager,
    initialize_resources, calculate_ca_hardness, export_results, generate_plots,
    config, # For default paths etc.
    SeawaterAnalyzerError, InputValidationError, PlottingError
)

# Import GUI components
from .models import PandasTableModel
from .worker import SimulationWorker
from .widgets.plot_widget import PlotWidget

logger = logging.getLogger(__name__) # Use logger setup by backend's init

class MainWindow(QMainWindow):
    """Main application window for the Seawater Analyzer."""

    def __init__(self, resource_manager: ResourceManager, parent=None):
        super().__init__(parent)
        self.rm = resource_manager
        self.simulator = PhreeqcSimulator(self.rm)
        self.processor = DataProcessor()
        # Make plotter use configured plot dir from rm
        self.plotter = PlotManager(output_dir=self.rm.get_output_path('plot', ''))

        # Store simulation results temporarily
        self._current_raw_result_df = pd.DataFrame()
        self._current_processed_result_df = pd.DataFrame()
        self._loaded_history_df = pd.DataFrame()

        # Default Excel file path (consider making this configurable)
        self.history_excel_file = self.rm.get_output_path('report', "experiment_history.xlsx")


        self.setWindowTitle("Seawater Analysis & Simulation Tool")
        self.setGeometry(100, 100, 1200, 800) # x, y, width, height

        # --- Setup Backend Worker Thread ---
        self.thread = QThread()
        self.worker = SimulationWorker(self.simulator, self.processor)
        self.worker.moveToThread(self.thread)

        # Connect worker signals to main thread slots
        self.worker.simulation_success.connect(self.on_simulation_success)
        self.worker.simulation_error.connect(self.on_simulation_error)
        self.worker.status_update.connect(self.update_status_bar)
        self.thread.started.connect(lambda: logger.info("Simulation thread started."))
        self.thread.finished.connect(lambda: logger.info("Simulation thread finished."))
        # Cleanup thread on exit? Optional.
        # self.destroyed.connect(self.thread.quit)

        self.thread.start()

        # --- Initialize UI ---
        self._create_actions()
        self._create_menu_bar()
        self._create_status_bar()
        self._create_central_widget()

        # --- Initial State ---
        self.update_status_bar("Application initialized. Ready.")
        self._load_history_data() # Load history on startup

    def _create_actions(self):
        """Create QAction objects for menus and toolbars."""
        self.quit_action = QAction("&Quit", self)
        self.quit_action.setShortcut(QKeySequence.Quit)
        self.quit_action.setStatusTip("Exit the application")
        self.quit_action.triggered.connect(self.close)

        self.load_history_action = QAction("&Load History File", self)
        self.load_history_action.setStatusTip("Load experiment history from an Excel file")
        self.load_history_action.triggered.connect(self._load_history_data_dialog)

        self.save_current_action = QAction("&Save Current Experiment", self)
        self.save_current_action.setStatusTip("Save the current simulation results to the history file")
        self.save_current_action.triggered.connect(self._save_current_experiment)
        self.save_current_action.setEnabled(False) # Disabled initially

        self.about_action = QAction("&About", self)
        self.about_action.setStatusTip("Show information about the application")
        self.about_action.triggered.connect(self._show_about_dialog)


    def _create_menu_bar(self):
        """Create the main menu bar."""
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.load_history_action)
        file_menu.addAction(self.save_current_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self.about_action)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximumSize(150, 15)
        self.progress_bar.setVisible(False) # Hide initially
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _create_central_widget(self):
        """Create the main layout and widgets."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Tab Widget for different sections ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Tab 1: Input and Simulation ---
        input_sim_widget = QWidget()
        input_sim_layout = QHBoxLayout(input_sim_widget) # Horizontal split
        self.tab_widget.addTab(input_sim_widget, "Experiment Input & Simulation")

        # Left side: Input Form
        input_group = QGroupBox("Experiment Details")
        input_form_layout = QFormLayout()
        input_group.setLayout(input_form_layout)

        self.exp_id_input = QLineEdit(f"Exp_{datetime.now().strftime('%Y%m%d_%H%M')}")
        self.date_input = QDateEdit(QDate.currentDate())
        self.date_input.setCalendarPopup(True)
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(0, 50); self.temp_input.setValue(config.DEFAULT_TEMPERATURE); self.temp_input.setSuffix(" °C")
        self.ph_input = QDoubleSpinBox()
        self.ph_input.setRange(0, 14); self.ph_input.setDecimals(2); self.ph_input.setValue(config.DEFAULT_PH)

        # Calcium Hardness Calculation Inputs
        ca_group = QGroupBox("Calcium Hardness (Titration)")
        ca_layout = QFormLayout()
        ca_group.setLayout(ca_layout)
        self.titrant_a_input = QDoubleSpinBox()
        self.titrant_a_input.setRange(0, 100); self.titrant_a_input.setSuffix(" mL")
        self.sample_vol_input = QDoubleSpinBox()
        self.sample_vol_input.setRange(0.1, 1000); self.sample_vol_input.setValue(50.0); self.sample_vol_input.setSuffix(" mL")
        self.dilution_input = QDoubleSpinBox()
        self.dilution_input.setRange(0.001, 1.0); self.dilution_input.setValue(0.04); self.dilution_input.setDecimals(3)
        self.ca_result_label = QLabel("Ca [mg/L]: ---") # Display calculated Ca

        ca_layout.addRow("Titrant Vol (A):", self.titrant_a_input)
        ca_layout.addRow("Diluted Sample Vol:", self.sample_vol_input)
        ca_layout.addRow("Dilution Factor:", self.dilution_input)
        ca_layout.addRow(self.ca_result_label)

        # Connect Ca inputs to calculation function
        self.titrant_a_input.valueChanged.connect(self._update_ca_calculation)
        self.sample_vol_input.valueChanged.connect(self._update_ca_calculation)
        self.dilution_input.valueChanged.connect(self._update_ca_calculation)

        self.alkalinity_input = QDoubleSpinBox()
        self.alkalinity_input.setRange(0, 10); self.alkalinity_input.setDecimals(3); self.alkalinity_input.setValue(2.4); self.alkalinity_input.setSuffix(" meq/kgw") # Adjust unit/default

        input_form_layout.addRow("Experiment ID:", self.exp_id_input)
        input_form_layout.addRow("Date:", self.date_input)
        input_form_layout.addRow("Temperature:", self.temp_input)
        input_form_layout.addRow("pH:", self.ph_input)
        input_form_layout.addRow("Alkalinity:", self.alkalinity_input)
        input_form_layout.addRow(ca_group) # Add Ca group box

        # Right side: Simulation Control and Results Display
        sim_results_layout = QVBoxLayout()
        self.run_sim_button = QPushButton("Run Simulation")
        self.run_sim_button.setStyleSheet("QPushButton { background-color: lightgreen; font-weight: bold; }")
        self.run_sim_button.clicked.connect(self.trigger_simulation)

        # Display key results
        results_group = QGroupBox("Simulation Quick Results")
        results_layout = QFormLayout()
        results_group.setLayout(results_layout)
        self.dic_result_label = QLabel("DIC [mM]: ---")
        self.si_calcite_label = QLabel("SI Calcite: ---")
        self.si_aragonite_label = QLabel("SI Aragonite: ---")
        self.pco2_result_label = QLabel("pCO2 [ppm]: ---")
        self.charge_bal_label = QLabel("Charge Balance: ---")

        results_layout.addRow(self.dic_result_label)
        results_layout.addRow(self.si_calcite_label)
        results_layout.addRow(self.si_aragonite_label)
        results_layout.addRow(self.pco2_result_label)
        results_layout.addRow(self.charge_bal_label)

        # Table for full results
        self.results_table_view = QTableView()
        self.results_table_model = PandasTableModel()
        self.results_table_view.setModel(self.results_table_model)
        self.results_table_view.setAlternatingRowColors(True)
        self.results_table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)


        sim_results_layout.addWidget(self.run_sim_button)
        sim_results_layout.addWidget(results_group)
        sim_results_layout.addWidget(QLabel("Full Simulation Output:"))
        sim_results_layout.addWidget(self.results_table_view)

        input_sim_layout.addWidget(input_group, 1) # Stretch factor 1
        input_sim_layout.addWidget(QWidget(), 0) # Spacer? Or adjust stretch factors
        input_sim_layout.addLayout(sim_results_layout, 2) # Stretch factor 2


        # --- Tab 2: History and Visualization ---
        history_viz_widget = QWidget()
        history_viz_layout = QVBoxLayout(history_viz_widget)
        self.tab_widget.addTab(history_viz_widget, "History & Visualization")

        # Top: History Table and Filters
        history_group = QGroupBox("Experiment History")
        history_group_layout = QVBoxLayout()
        history_group.setLayout(history_group_layout)

        filter_layout = QHBoxLayout()
        self.hist_start_date = QDateEdit(self.date_input.minimumDate()) # Use min from input? Or calculate from data
        self.hist_start_date.setCalendarPopup(True)
        self.hist_end_date = QDateEdit(QDate.currentDate())
        self.hist_end_date.setCalendarPopup(True)
        filter_button = QPushButton("Apply Filters & Plot")
        filter_button.clicked.connect(self._filter_and_plot_history)

        filter_layout.addWidget(QLabel("Start Date:"))
        filter_layout.addWidget(self.hist_start_date)
        filter_layout.addWidget(QLabel("End Date:"))
        filter_layout.addWidget(self.hist_end_date)
        filter_layout.addStretch()
        filter_layout.addWidget(filter_button)

        self.history_table_view = QTableView()
        self.history_table_model = PandasTableModel()
        self.history_table_view.setModel(self.history_table_model)
        self.history_table_view.setAlternatingRowColors(True)
        self.history_table_view.setSortingEnabled(True) # Enable sorting
        self.history_table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive) # Allow resize
        self.history_table_view.verticalHeader().setVisible(False) # Hide row numbers


        history_group_layout.addLayout(filter_layout)
        history_group_layout.addWidget(self.history_table_view)

        # Bottom: Plotting Area
        plot_group = QGroupBox("Data Visualization")
        plot_group_layout = QVBoxLayout()
        plot_group.setLayout(plot_group_layout)
        self.plot_widget = PlotWidget()
        plot_group_layout.addWidget(self.plot_widget)

        # Splitter for History and Plot
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(history_group)
        splitter.addWidget(plot_group)
        splitter.setStretchFactor(0, 1) # History table gets more initial space
        splitter.setStretchFactor(1, 1)
        history_viz_layout.addWidget(splitter)


        # --- Log Area (Below Tabs) ---
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(100) # Fixed height for log area
        self.log_output.setStyleSheet("QTextEdit { background-color: #f0f0f0; font-family: Consolas, monospace; }")
        main_layout.addWidget(QLabel("Log Messages:"))
        main_layout.addWidget(self.log_output)

        # --- Configure Log Handler ---
        # Add a handler to send specific logs to the QTextEdit
        self.log_handler = QtLogHandler(self.log_output)
        # Format logs going to the GUI
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        self.log_handler.setFormatter(formatter)
        # Add handler to the root logger or the specific logger used by the backend
        logging.getLogger('seawater_analyzer').addHandler(self.log_handler)
        logging.getLogger('seawater_analyzer').setLevel(logging.INFO) # Set level for GUI display


    # --- UI Update Slots ---

    @Slot(str)
    def update_status_bar(self, message):
        """Updates the status bar message."""
        self.status_bar.showMessage(message, 5000) # Show for 5 seconds
        # Also append important messages to the log window
        if "Error" in message or "Fail" in message or "Success" in message or "Start" in message or "Finish" in message:
             logger.info(message) # Let the handler put it in the log widget


    @Slot(pd.DataFrame, pd.DataFrame)
    def on_simulation_success(self, raw_df, processed_df):
        """Handles successful simulation completion."""
        self.update_status_bar("Simulation successful.")
        self._current_raw_result_df = raw_df
        self._current_processed_result_df = processed_df

        # Update results table view
        self.results_table_model.updateData(raw_df) # Show raw PHREEQC output
        self.results_table_view.resizeColumnsToContents()

        # Update quick result labels
        if not processed_df.empty:
             # Take the last row if multiple steps (or first if only one)
             last_result = processed_df.iloc[-1]
             self.dic_result_label.setText(f"DIC [mM]: {last_result.get('DIC_mM', 'N/A'):.3f}")
             self.si_calcite_label.setText(f"SI Calcite: {last_result.get('SI_Calcite', 'N/A'):.3f}")
             self.si_aragonite_label.setText(f"SI Aragonite: {last_result.get('SI_Aragonite', 'N/A'):.3f}")
             self.pco2_result_label.setText(f"pCO2 [ppm]: {last_result.get('pCO2_ppm', 'N/A'):.1f}")
             self.charge_bal_label.setText(f"Charge Balance: {last_result.get('charge_balance', 'N/A'):.4e}")
             self.save_current_action.setEnabled(True) # Enable saving now
        else:
             self._clear_quick_results()
             self.save_current_action.setEnabled(False) # Disable saving if no results

        self.run_sim_button.setEnabled(True) # Re-enable button
        self.progress_bar.setVisible(False)

    @Slot(str)
    def on_simulation_error(self, error_message):
        """Handles simulation errors."""
        self.update_status_bar(f"Simulation Error: {error_message}")
        QMessageBox.critical(self, "Simulation Error", error_message)
        self._current_raw_result_df = pd.DataFrame() # Clear results on error
        self._current_processed_result_df = pd.DataFrame()
        self._clear_quick_results()
        self.results_table_model.updateData(pd.DataFrame()) # Clear table
        self.run_sim_button.setEnabled(True) # Re-enable button
        self.progress_bar.setVisible(False)
        self.save_current_action.setEnabled(False)


    # --- Backend Interaction Methods ---

    def trigger_simulation(self):
        """Gathers input data and starts the simulation worker task."""
        self.update_status_bar("Preparing simulation...")
        self._clear_quick_results()
        self.results_table_model.updateData(pd.DataFrame()) # Clear table
        self.save_current_action.setEnabled(False) # Disable saving until success

        try:
            # 1. Calculate current Ca concentration
            ca_conc_mgL = self._calculate_ca_hardness()
            if ca_conc_mgL is None: # Error occurred during calculation
                # Error already shown by _calculate_ca_hardness
                self.update_status_bar("Cannot run simulation due to invalid Ca input.")
                return

            # 2. Gather other parameters
            sim_params = {
                'temp': self.temp_input.value(),
                'pH': self.ph_input.value(),
                'pe': 4.0,  # Default assumption
                'density': 1.025, # Default assumption (could be input)
                'alkalinity': self.alkalinity_input.value(), # Assumed meq/kgw
                'composition': config.STANDARD_SEAWATER_COMPOSITION.copy(), # Use default base
                 # Specify desired outputs (make configurable later?)
                 'selected_elements': ["Ca", "Mg", "Na", "K", "Cl", "S(6)", "C(4)"],
                 'selected_phases': ["Calcite", "Aragonite", "Dolomite", "Brucite", "Gypsum", "CO2(g)"]
            }
            # Override Ca in composition with calculated value (ensure correct units for PHREEQC - mg/kgw?)
            # Assuming mg/L ~= mg/kgw for this example
            sim_params['composition']['Ca'] = ca_conc_mgL

            # Add metadata for processing and saving later
            sim_params['input_Experiment_ID'] = self.exp_id_input.text()
            sim_params['input_Date'] = self.date_input.date().toString(Qt.DateFormat.ISODate)
            sim_params['input_Ca_measured'] = ca_conc_mgL
            sim_params['input_Alkalinity_measured'] = self.alkalinity_input.value() # Save the input Alk value


            # 3. Validate (Quick check before threading) - worker validates again
            # self.processor.validate_input(sim_params) # Optional pre-check

            # 4. Disable button, show progress, trigger worker
            self.run_sim_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0) # Indeterminate progress
            self.update_status_bar("Starting simulation task...")

            # Call the worker's slot via signal/slot mechanism is better if worker needs init args
            # Or directly call the slot if worker instance exists (as here)
            # Make sure to pass parameters that are thread-safe (dicts, basic types are fine)
            db_to_use = config.DEFAULT_PHREEQC_DB # Or get from UI element
            self.worker.run_simulation_task(sim_params, db_to_use)


        except InputValidationError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")
            self.update_status_bar(f"Input Error: {e}")
        except Exception as e:
             QMessageBox.critical(self, "Error", f"Failed to prepare simulation: {e}")
             logger.error(f"Error preparing simulation: {e}", exc_info=True)
             self.update_status_bar(f"Error: {e}")


    def _update_ca_calculation(self):
         """Calls calculation and updates the label."""
         ca_conc = self._calculate_ca_hardness()
         if ca_conc is not None:
             self.ca_result_label.setText(f"Ca [mg/L]: {ca_conc:.1f}")
         else:
             self.ca_result_label.setText("Ca [mg/L]: Error")


    def _calculate_ca_hardness(self) -> Optional[float]:
        """Calculates Ca hardness from input fields."""
        try:
            a = self.titrant_a_input.value()
            vol = self.sample_vol_input.value()
            dil = self.dilution_input.value()
            # Using the backend function
            ca_conc = calculate_ca_hardness(titrant_volume_A=a, sample_volume=vol, dilution_factor=dil)
            return ca_conc
        except ValueError as e:
             # Handle potential errors from the backend function (e.g., division by zero)
             QMessageBox.warning(self, "Calculation Error", f"Invalid input for Ca calculation: {e}")
             logger.warning(f"Ca calculation error: {e}")
             return None
        except Exception as e:
             QMessageBox.critical(self, "Calculation Error", f"Unexpected error in Ca calculation: {e}")
             logger.error(f"Unexpected Ca calculation error: {e}", exc_info=True)
             return None

    def _clear_quick_results(self):
        """Resets the quick result labels."""
        self.dic_result_label.setText("DIC [mM]: ---")
        self.si_calcite_label.setText("SI Calcite: ---")
        self.si_aragonite_label.setText("SI Aragonite: ---")
        self.pco2_result_label.setText("pCO2 [ppm]: ---")
        self.charge_bal_label.setText("Charge Balance: ---")

    def _save_current_experiment(self):
        """Saves the processed results of the current simulation."""
        if self._current_processed_result_df.empty:
            QMessageBox.warning(self, "Save Error", "No simulation results available to save.")
            return

        try:
            # Prepare data row for saving (usually the last row of processed results)
            # The 'input_*' fields should already be in the processed_df from process_results
            experiment_row_df = self._current_processed_result_df.iloc[[-1]] # Keep as DataFrame

            # Define required columns for the history file (match plotter/reference)
            history_cols = [
                'input_Date', 'input_Experiment_ID', 'Temperature_C', 'pH',
                'input_Ca_measured', 'input_Alkalinity_measured', # Use the input measured values
                'DIC_mM', 'SI_Calcite', 'pCO2_ppm'
                # Add SI_Aragonite, Ionic_Strength etc. if needed for plotting/analysis later
            ]
            # Select and rename columns to match expected history format if needed
            save_data = pd.DataFrame()
            rename_map_save = {
                'input_Date': 'Date',
                'input_Experiment_ID': 'Experiment_ID',
                'input_Ca_measured': 'Ca_measured [ppm]', # Assuming mg/L ~ ppm
                'input_Alkalinity_measured': 'Alkalinity', # Assuming units match history
                'SI_Calcite': 'SI_Calcite', # Keep same
                'pCO2_ppm': 'pCO2 [ppm]',
                'DIC_mM': 'DIC_concentration [mM]',
                # Add others... pH, Temperature_C should exist
            }
            for backend_col, hist_col in rename_map_save.items():
                if backend_col in experiment_row_df.columns:
                    save_data[hist_col] = experiment_row_df[backend_col]
                else:
                    logger.warning(f"Column '{backend_col}' not found in processed results for saving.")
            # Add columns that don't need renaming
            if 'pH' in experiment_row_df.columns: save_data['pH'] = experiment_row_df['pH']
            if 'Temperature_C' in experiment_row_df.columns: save_data['Temperature'] = experiment_row_df['Temperature_C']


            # Ensure all target history columns exist, fill with NaN if missing in source
            # This requires knowing the definitive schema of the history file
            target_hist_cols = ['Date', 'Experiment_ID', 'Temperature', 'pH', 'Ca_measured [ppm]', 'Alkalinity', 'DIC_concentration [mM]', 'SI_Calcite', 'pCO2 [ppm]']
            for col in target_hist_cols:
                 if col not in save_data.columns:
                     save_data[col] = pd.NA


            # Reorder to match target schema
            save_data = save_data[target_hist_cols]


            self.update_status_bar(f"Saving experiment to {self.history_excel_file}...")

            # Append to Excel file
            mode = 'a' if os.path.exists(self.history_excel_file) else 'w'
            header = not os.path.exists(self.history_excel_file) # Write header only if file doesn't exist
            excel_writer_kwargs = {'if_sheet_exists': 'overlay' if mode == 'a' else None} if pd.__version__ >= '1.4.0' else {}


            # Need 'openpyxl' installed
            with pd.ExcelWriter(self.history_excel_file, engine='openpyxl', mode=mode, **excel_writer_kwargs) as writer:
                # Find the next available row if appending
                startrow = 0
                if mode == 'a':
                    try:
                        # Read existing sheet to find length (potential performance issue for large files)
                        existing_df = pd.read_excel(self.history_excel_file)
                        startrow = len(existing_df)
                    except FileNotFoundError:
                         header = True # File disappeared? Write header.
                    except Exception as e:
                         logger.warning(f"Could not read existing excel file to find length: {e}. Appending might overwrite.")
                         # Fallback: Just append, risk overwrite if sheet exists weirdly in 'a' mode without if_sheet_exists
                         pass


                save_data.to_excel(writer, sheet_name='Experiments', index=False, header=header, startrow=startrow if not header else 0)


            self.update_status_bar("Experiment saved successfully.")
            self._load_history_data() # Reload history view after saving

        except ImportError:
             QMessageBox.critical(self, "Save Error", "The 'openpyxl' library is required to save Excel files. Please install it (`pip install openpyxl`).")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save experiment data: {e}")
            logger.error(f"Failed to save experiment: {e}", exc_info=True)
            self.update_status_bar(f"Error saving experiment: {e}")

    def _load_history_data_dialog(self):
        """Opens a dialog to select and load a history file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Experiment History",
            str(self.rm.output_dir / "reports"), # Start directory
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if filename:
            self.history_excel_file = Path(filename)
            self._load_history_data()

    def _load_history_data(self):
        """Loads data from the specified Excel file into the history table."""
        self.update_status_bar(f"Loading history from {self.history_excel_file}...")
        try:
            if not os.path.exists(self.history_excel_file):
                 logger.warning(f"History file not found: {self.history_excel_file}. Creating empty table.")
                 self._loaded_history_df = pd.DataFrame() # Start empty if no file
            else:
                 # Specify sheet name if necessary
                 self._loaded_history_df = pd.read_excel(self.history_excel_file, sheet_name='Experiments')
                 # Basic data cleaning/type conversion
                 if 'Date' in self._loaded_history_df.columns:
                     self._loaded_history_df['Date'] = pd.to_datetime(self._loaded_history_df['Date'], errors='coerce')
                 # Convert numeric columns, coercing errors
                 for col in ['Temperature', 'pH', 'Ca_measured [ppm]', 'Alkalinity', 'DIC_concentration [mM]', 'SI_Calcite', 'pCO2 [ppm]']:
                      if col in self._loaded_history_df.columns:
                          self._loaded_history_df[col] = pd.to_numeric(self._loaded_history_df[col], errors='coerce')

            self.history_table_model.updateData(self._loaded_history_df)
            self.history_table_view.resizeColumnsToContents()
            self.update_status_bar("History loaded successfully.")
            logger.info(f"Loaded {len(self._loaded_history_df)} experiments from history.")
            # Set date filters based on loaded data
            if not self._loaded_history_df.empty and 'Date' in self._loaded_history_df.columns:
                 min_date = self._loaded_history_df['Date'].min()
                 max_date = self._loaded_history_df['Date'].max()
                 if pd.notna(min_date): self.hist_start_date.setDate(QDate(min_date.year, min_date.month, min_date.day))
                 if pd.notna(max_date): self.hist_end_date.setDate(QDate(max_date.year, max_date.month, max_date.day))

            # Trigger initial plot after loading
            self._filter_and_plot_history()

        except ImportError:
             QMessageBox.critical(self, "Load Error", "The 'openpyxl' library is required to load Excel files. Please install it (`pip install openpyxl`).")
             self._loaded_history_df = pd.DataFrame()
             self.history_table_model.updateData(self._loaded_history_df)
             self.update_status_bar("Error: Missing 'openpyxl'.")
        except FileNotFoundError:
             self.update_status_bar(f"History file not found: {self.history_excel_file}")
             self._loaded_history_df = pd.DataFrame()
             self.history_table_model.updateData(self._loaded_history_df)
             self.plot_widget.clear_plot() # Clear plot if no file
        except Exception as e:
             QMessageBox.critical(self, "Load Error", f"Failed to load history data: {e}")
             logger.error(f"Failed to load history: {e}", exc_info=True)
             self._loaded_history_df = pd.DataFrame() # Clear data on error
             self.history_table_model.updateData(self._loaded_history_df)
             self.update_status_bar(f"Error loading history: {e}")
             self.plot_widget.clear_plot() # Clear plot on error


    def _filter_and_plot_history(self):
        """Filters the loaded history data and generates/updates the plot."""
        if self._loaded_history_df.empty:
            logger.info("No history data loaded, skipping plot.")
            self.plot_widget.clear_plot()
            return

        try:
            start_qdate = self.hist_start_date.date()
            end_qdate = self.hist_end_date.date()
            start_date = pd.Timestamp(start_qdate.year(), start_qdate.month(), start_qdate.day())
            end_date = pd.Timestamp(end_qdate.year(), end_qdate.month(), end_qdate.day()).normalize() + pd.Timedelta(days=1) # Include end date

            # Filter DataFrame (handle NaT in Date column)
            filtered_df = self._loaded_history_df.copy()
            if 'Date' in filtered_df.columns:
                 date_mask = (filtered_df['Date'].notna()) & \
                             (filtered_df['Date'] >= start_date) & \
                             (filtered_df['Date'] < end_date) # Use < for end date + 1 day
                 filtered_df = filtered_df[date_mask]
            else:
                 logger.warning("History data missing 'Date' column for filtering.")


            if filtered_df.empty:
                self.update_status_bar("No data found for the selected date range.")
                self.plot_widget.clear_plot()
                return

            self.update_status_bar(f"Generating plot for {len(filtered_df)} experiments...")

            # Use the backend PlotManager to generate the figure object
            # Make sure column names match those expected by plot_simulation_summary
            # Requires: 'Experiment_ID', 'Date', 'Ca_measured [ppm]', 'DIC_concentration [mM]', 'pH', 'SI_Calcite'
            fig = self.plotter.generate_plots(filtered_df, plot_type='summary',
                                              x_col='Experiment_ID', date_col='Date',
                                              ca_col='Ca_measured [ppm]', dic_col='DIC_concentration [mM]',
                                              ph_col='pH', si_col='SI_Calcite',
                                              ca_units='ppm') # Specify units of the Ca column

            # Update the plot widget with the generated figure
            self.plot_widget.update_plot(fig)
            self.update_status_bar("Plot updated.")

            # Optional: Save the plot automatically or via button
            # plot_filename = f"history_plot_{start_qdate.toString('yyyyMMdd')}_{end_qdate.toString('yyyyMMdd')}"
            # self.plotter.export_plot(fig=fig, filename=plot_filename)


        except PlottingError as e:
            QMessageBox.warning(self, "Plotting Error", f"Could not generate plot: {e}")
            logger.error(f"Plotting error: {e}", exc_info=True)
            self.update_status_bar(f"Plotting Error: {e}")
            self.plot_widget.clear_plot() # Clear on error
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred during plotting: {e}")
            logger.error(f"Unexpected plotting error: {e}", exc_info=True)
            self.update_status_bar(f"Plotting Error: {e}")
            self.plot_widget.clear_plot() # Clear on error


    def _show_about_dialog(self):
        """Displays a simple About dialog."""
        about_text = """
        <b>Seawater Analysis & Simulation Tool</b>
        <p>Version: 0.1.0 (GUI)</p>
        <p>This application uses the `seawater_analyzer` backend
        to process experimental data and run PHREEQC simulations.</p>
        <p>Built with PySide6 and Matplotlib.</p>
        """
        QMessageBox.about(self, "About Seawater Analyzer", about_text)


    def closeEvent(self, event):
        """Handles the window close event."""
        reply = QMessageBox.question(self, 'Confirm Quit',
                                     "Are you sure you want to quit?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Exiting application.")
            # Stop the worker thread gracefully
            if self.thread.isRunning():
                 self.thread.quit()
                 self.thread.wait(1000) # Wait max 1 sec
            event.accept()
        else:
            event.ignore()


# --- Custom Log Handler ---
class QtLogHandler(logging.Handler):
    """A logging handler that emits a Qt signal with the log message."""
    def __init__(self, text_widget: QTextEdit):
        super().__init__()
        self.widget = text_widget
        # Keep track of the Qt object emitting the signal
        self.emitter = LogEmitter()

        # Connect the emitter's signal to the widget's append method
        # Use QueuedConnection to ensure thread safety if logs come from other threads
        self.emitter.log_message.connect(self.widget.append, Qt.ConnectionType.QueuedConnection)


    def emit(self, record):
        """Formats the log record and emits the signal."""
        try:
            msg = self.format(record)
            # Emit the signal with the formatted message
            self.emitter.log_message.emit(msg)
        except Exception:
            self.handleError(record)

class LogEmitter(QObject):
     """Simple QObject to emit signals for logging."""
     log_message = Signal(str)