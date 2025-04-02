import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from . import config
from .exceptions import PlottingError

logger = logging.getLogger(__name__)

class PlotManager:
    """Handles generation and export of plots using Matplotlib and Seaborn."""

    def __init__(self, output_dir: Path = config.PLOT_DIR, style_params: Optional[Dict] = None):
        """
        Initializes the PlotManager.

        Args:
            output_dir (Path): Directory to save generated plots.
            style_params (Optional[Dict]): Dictionary of Matplotlib rcParams for custom styling.
                                           If None, uses defaults from config.py.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style_params = style_params if style_params is not None else config.DEFAULT_PLOT_STYLE
        self._apply_style()
        self.current_fig: Optional[plt.Figure] = None # Store the last generated figure

    def _apply_style(self):
        """Applies the configured Matplotlib style."""
        try:
            plt.rcParams.update(self.style_params)
            logger.debug("Applied Matplotlib style parameters.")
        except Exception as e:
            logger.warning(f"Could not apply Matplotlib style: {e}")

    def _save_figure(self, fig: plt.Figure, filename: str, format: Optional[str] = None, **savefig_kwargs):
        """Internal helper to save a figure to the output directory."""
        if not filename:
            raise ValueError("Filename cannot be empty for saving plot.")

        save_format = format if format else config.DEFAULT_PLOT_FORMAT
        filepath = (self.output_dir / filename).with_suffix(f".{save_format}")

        try:
            # Ensure directory exists again just before saving
            filepath.parent.mkdir(parents=True, exist_ok=True)
            # Default bbox_inches='tight' is often useful
            kwargs = {'bbox_inches': 'tight', 'dpi': config.DEFAULT_PLOT_STYLE.get('savefig.dpi', 300)}
            kwargs.update(savefig_kwargs) # Allow overriding defaults
            fig.savefig(filepath, format=save_format, **kwargs)
            logger.info(f"Plot saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save plot to {filepath}: {e}", exc_info=True)
            raise PlottingError(f"Error saving plot '{filename}': {e}") from e

    def generate_plots(self, dataset: pd.DataFrame, plot_type: str, **kwargs) -> plt.Figure:
        """
        Generic function to generate different plot types based on 'plot_type'.

        Args:
            dataset (pd.DataFrame): Data to plot.
            plot_type (str): Type of plot ('summary', 'time_series', 'scatter', 'correlation').
            **kwargs: Additional arguments specific to the plot type.

        Returns:
            plt.Figure: The generated Matplotlib figure object.

        Raises:
            PlottingError: If plot generation fails or type is unknown.
            ValueError: If required kwargs for a plot type are missing.
        """
        if dataset.empty:
             raise ValueError("Cannot generate plots from empty dataset.")

        logger.info(f"Generating plot of type '{plot_type}'...")

        if plot_type == 'summary':
             fig = self.plot_simulation_summary(dataset, **kwargs)
        elif plot_type == 'time_series':
             fig = self.plot_time_series(dataset, **kwargs)
        elif plot_type == 'scatter':
             fig = self.plot_scatter(dataset, **kwargs)
        elif plot_type == 'correlation':
             fig = self.plot_correlation_matrix(dataset, **kwargs)
        # Add more plot types here
        else:
             raise ValueError(f"Unknown plot type: '{plot_type}'")

        self.current_fig = fig # Store the generated figure
        plt.close(fig) # Close the figure to prevent display in non-interactive backend envs
                       # The caller can use fig.show() or save it if needed.
        logger.info(f"'{plot_type}' plot generated.")
        return fig


    def plot_simulation_summary(self, dataset: pd.DataFrame,
                                x_col: str = 'Experiment_ID', date_col: Optional[str] = 'Date',
                                ca_col: str = 'input_Ca_measured', # Using input Ca
                                dic_col: str = 'DIC_mM',
                                ph_col: str = 'pH', si_col: str = 'SI_Calcite',
                                title1: str = "Measured Calcium and Calculated DIC",
                                title2: str = "Saturation Index (Calcite) vs. pH",
                                fig_size: Tuple[int, int] = (18, 14),
                                ca_units: str = 'mg/L') -> plt.Figure:
        """
        Generates a two-panel summary plot similar to the reference script.
        Top: Bar chart of Calcium (converted to mM) and DIC (mM).
        Bottom: Scatter plot of SI vs pH.

        Args:
            dataset (pd.DataFrame): DataFrame containing processed experiment/simulation data.
                                     Requires columns specified by *_col arguments.
            x_col (str): Column to use for x-axis labels on the bar chart (e.g., 'Experiment_ID').
            date_col (Optional[str]): Column with date information to add to x-labels.
            ca_col (str): Column containing Calcium measurements (in units specified by ca_units).
            dic_col (str): Column containing calculated DIC (expected in mM).
            ph_col (str): Column containing pH values.
            si_col (str): Column containing Saturation Index values (e.g., 'SI_Calcite').
            title1 (str): Title for the top panel (bar chart).
            title2 (str): Title for the bottom panel (scatter plot).
            fig_size (Tuple[int, int]): Figure size (width, height).
            ca_units (str): Units of the input calcium column ('mg/L' or 'ppm').

        Returns:
            plt.Figure: The generated Matplotlib figure.

        Raises:
            PlottingError: If required columns are missing or plotting fails.
        """
        required_cols = [x_col, ca_col, dic_col, ph_col, si_col]
        if date_col: required_cols.append(date_col)
        missing_cols = [col for col in required_cols if col not in dataset.columns]
        if missing_cols:
             raise PlottingError(f"Missing required columns for summary plot: {missing_cols}")

        df = dataset.copy()

        # Convert Ca to mM for plotting consistency with DIC
        if ca_units.lower() in ['mg/l', 'ppm']: # Assume ppm = mg/L for seawater density ~1
             ca_molar_mass = config.MOLAR_MASSES.get('Ca', 40.08)
             df['Ca_mM'] = pd.to_numeric(df[ca_col], errors='coerce') / ca_molar_mass
             ca_plot_col = 'Ca_mM'
             ca_label = 'Calcium (mM)'
        else:
            # If Ca is already in mM or other units, use directly (adjust label)
            df[ca_col] = pd.to_numeric(df[ca_col], errors='coerce')
            ca_plot_col = ca_col
            ca_label = f'Calcium ({ca_units})'
            logger.warning(f"Calcium units '{ca_units}' not 'mg/L' or 'ppm'. Plotting directly.")

        df[dic_col] = pd.to_numeric(df[dic_col], errors='coerce')
        df[ph_col] = pd.to_numeric(df[ph_col], errors='coerce')
        df[si_col] = pd.to_numeric(df[si_col], errors='coerce')

        # Drop rows where essential plotting data is missing
        plot_df = df.dropna(subset=[x_col, ca_plot_col, dic_col, ph_col, si_col]).reset_index()
        if plot_df.empty:
             raise PlottingError("No valid data remains after handling missing values for summary plot.")

        # Create figure and gridspec
        fig = plt.figure(figsize=fig_size) # Use facecolor from rcParams
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3) # Add vertical space

        # --- Top Panel: Bar Chart (Ca vs DIC) ---
        ax1 = fig.add_subplot(gs[0])
        ax2 = ax1.twinx() # Twin axis for DIC

        num_entries = len(plot_df)
        x = np.arange(num_entries)
        width = 0.35 # Width of the bars

        # Plot bars
        rects1 = ax1.bar(x - width/2, plot_df[ca_plot_col], width, label=ca_label, color='skyblue', edgecolor='black')
        rects2 = ax2.bar(x + width/2, plot_df[dic_col], width, label='DIC (mM)', color='salmon', edgecolor='black')

        # Labels and Title
        ax1.set_ylabel(ca_label)
        ax2.set_ylabel('DIC (mM)')
        ax1.set_title(title1)
        ax1.set_xticks(x)

        # Create X-axis labels (ID + Date if available)
        if date_col:
             # Ensure date column is datetime type for formatting
             try:
                 plot_df[date_col] = pd.to_datetime(plot_df[date_col])
                 x_labels = [f"{row[x_col]}\n({row[date_col].strftime('%Y-%m-%d')})" for idx, row in plot_df.iterrows()]
             except Exception: # Handle cases where date conversion fails
                 logger.warning(f"Could not format date column '{date_col}'. Using only '{x_col}'.")
                 x_labels = plot_df[x_col].astype(str).tolist()
        else:
             x_labels = plot_df[x_col].astype(str).tolist()

        ax1.set_xticklabels(x_labels, rotation=45, ha='right')

        # Add bar labels (optional, can get crowded)
        # ax1.bar_label(rects1, padding=3, fmt='%.1f')
        # ax2.bar_label(rects2, padding=3, fmt='%.1f')

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        ax1.tick_params(axis='x', rotation=45) # Ensure rotation is applied

        # --- Bottom Panel: Scatter Plot (SI vs pH) ---
        ax3 = fig.add_subplot(gs[1])

        # Use a colormap for potentially many points
        colors = cm.viridis(np.linspace(0, 1, num_entries))
        scatter = ax3.scatter(plot_df[ph_col], plot_df[si_col], c=colors, s=80, edgecolors='k', alpha=0.8)

        # Add labels for points (if dataset is small enough)
        if num_entries <= 20: # Add labels only if not too cluttered
             for i, row in plot_df.iterrows():
                 label = row[x_col] # Use basic ID for label
                 ax3.text(row[ph_col], row[si_col], f' {label}', va='center', ha='left', fontsize='small')
        # Alternatively, create a legend if labels are too much
        # Or use interactive plotting tools (outside scope of basic plotter)

        ax3.set_xlabel("pH")
        ax3.set_ylabel(si_col.replace('_', ' ')) # Nicer label
        ax3.set_title(title2)
        ax3.grid(True, linestyle='--', alpha=0.6)

        # Optional: Add colorbar if points are colored by a continuous variable instead of index
        # cbar = fig.colorbar(scatter)
        # cbar.set_label('Color Variable')

        fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap

        return fig

    def plot_time_series(self, dataset: pd.DataFrame, time_col: str, value_cols: List[str],
                          title: Optional[str] = None, xlabel: str = "Time", ylabels: Optional[List[str]] = None,
                          fig_size: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Generates a time series plot for one or more variables.

        Args:
            dataset (pd.DataFrame): DataFrame containing time series data.
            time_col (str): Name of the column containing datetime objects or timestamps.
            value_cols (List[str]): List of column names for the variables to plot on the y-axis.
            title (Optional[str]): Title for the plot.
            xlabel (str): Label for the x-axis.
            ylabels (Optional[List[str]]): Labels for the y-axis. If None, uses column names.
                                          If one label provided, uses it for the primary axis.
                                          If multiple, requires multiple axes (not implemented here).
            fig_size (Tuple[int, int]): Figure size.

        Returns:
            plt.Figure: The generated Matplotlib figure.

        Raises:
            PlottingError: If required columns are missing or data is unsuitable.
        """
        required_cols = [time_col] + value_cols
        missing_cols = [col for col in required_cols if col not in dataset.columns]
        if missing_cols:
            raise PlottingError(f"Missing required columns for time series plot: {missing_cols}")

        df = dataset.copy()

        # Ensure time column is datetime type
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            raise PlottingError(f"Could not convert time column '{time_col}' to datetime: {e}")

        df = df.sort_values(by=time_col) # Ensure data is sorted by time

        fig, ax = plt.subplots(figsize=fig_size)

        if ylabels and len(ylabels) != len(value_cols):
             logger.warning("Number of ylabels provided does not match number of value_cols. Using column names.")
             ylabels = None

        for i, col in enumerate(value_cols):
             y_data = pd.to_numeric(df[col], errors='coerce')
             label = ylabels[i] if ylabels else col.replace('_', ' ') # Use provided label or format column name
             ax.plot(df[time_col], y_data, marker='o', linestyle='-', markersize=4, label=label)

        ax.set_xlabel(xlabel)
        if len(value_cols) == 1 and ylabels:
             ax.set_ylabel(ylabels[0])
        elif len(value_cols) == 1:
             ax.set_ylabel(value_cols[0].replace('_', ' '))
        else:
             ax.set_ylabel("Value") # Generic label if multiple variables on same axis

        if title:
            ax.set_title(title)

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.autofmt_xdate() # Improve formatting of date labels
        fig.tight_layout()

        return fig


    def plot_scatter(self, dataset: pd.DataFrame, x_col: str, y_col: str,
                     color_col: Optional[str] = None, size_col: Optional[str] = None,
                     style_col: Optional[str] = None, # For marker style
                     title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                     fig_size: Tuple[int, int] = (8, 7)) -> plt.Figure:
        """
        Generates a scatter plot using Seaborn for enhanced aesthetics.

        Args:
            dataset (pd.DataFrame): DataFrame containing the data.
            x_col (str): Column name for the x-axis.
            y_col (str): Column name for the y-axis.
            color_col (Optional[str]): Column name to determine point color (hue).
            size_col (Optional[str]): Column name to determine point size.
            style_col (Optional[str]): Column name to determine point marker style.
            title (Optional[str]): Title for the plot.
            xlabel (Optional[str]): Label for the x-axis (defaults to x_col).
            ylabel (Optional[str]): Label for the y-axis (defaults to y_col).
            fig_size (Tuple[int, int]): Figure size.

        Returns:
            plt.Figure: The generated Matplotlib figure.

        Raises:
            PlottingError: If required columns are missing.
        """
        required_cols = [x_col, y_col]
        optional_cols = [color_col, size_col, style_col]
        for col in optional_cols:
            if col: required_cols.append(col)

        missing_cols = [col for col in required_cols if col not in dataset.columns]
        if missing_cols:
            raise PlottingError(f"Missing required columns for scatter plot: {missing_cols}")

        df = dataset.copy()
        # Ensure numeric types for axes, size
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        if size_col:
             df[size_col] = pd.to_numeric(df[size_col], errors='coerce')

        # Drop rows with NaN in essential plotting columns
        plot_df = df.dropna(subset=[x_col, y_col] + ([size_col] if size_col else []))
        if plot_df.empty:
             raise PlottingError("No valid data remains after handling missing values for scatter plot.")

        fig, ax = plt.subplots(figsize=fig_size)

        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=color_col,
            size=size_col,
            style=style_col,
            ax=ax,
            edgecolor='k', # Add edge color for visibility
            alpha=0.7       # Add transparency
        )

        ax.set_xlabel(xlabel if xlabel else x_col.replace('_', ' '))
        ax.set_ylabel(ylabel if ylabel else y_col.replace('_', ' '))
        if title:
            ax.set_title(title)

        ax.grid(True, linestyle='--', alpha=0.6)
        if color_col or size_col or style_col:
             # Adjust legend position if it exists
             ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        fig.tight_layout(rect=[0, 0, 0.85 if (color_col or size_col or style_col) else 0.98, 1]) # Adjust for legend

        return fig

    def plot_correlation_matrix(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None,
                                title: str = "Correlation Matrix", cmap: str = 'coolwarm',
                                fig_size: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plots a heatmap of the correlation matrix for numeric columns.

        Args:
            dataset (pd.DataFrame): DataFrame containing the data.
            columns (Optional[List[str]]): Specific list of numeric columns to include.
                                           If None, uses all numeric columns.
            title (str): Title for the plot.
            cmap (str): Colormap for the heatmap.
            fig_size (Tuple[int, int]): Figure size.

        Returns:
            plt.Figure: The generated Matplotlib figure.
        """
        if columns:
            numeric_df = dataset[columns].select_dtypes(include=np.number)
        else:
            numeric_df = dataset.select_dtypes(include=np.number)

        if numeric_df.empty or numeric_df.shape[1] < 2:
             raise PlottingError("Need at least two numeric columns for correlation matrix.")

        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, ax=ax,
                    cbar_kws={"shrink": 0.8}) # Adjust color bar size
        ax.set_title(title)
        # Improve layout for long labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        fig.tight_layout()

        return fig


    def export_plot(self, fig: Optional[plt.Figure] = None, filename: str = "plot", format: Optional[str] = None, **savefig_kwargs):
        """
        Exports a given figure or the last generated figure.

        Args:
            fig (Optional[plt.Figure]): The figure object to save. If None, saves self.current_fig.
            filename (str): Base filename for the saved plot (without extension).
            format (Optional[str]): File format (e.g., 'png', 'pdf', 'svg'). Uses default if None.
            **savefig_kwargs: Additional arguments passed to fig.savefig().
        """
        figure_to_save = fig if fig is not None else self.current_fig

        if figure_to_save is None:
            raise PlottingError("No figure provided or previously generated to export.")

        self._save_figure(figure_to_save, filename, format, **savefig_kwargs)

# Standalone function interface
# This might be less useful if state (current_fig) is important,
# but can work for generating/saving in one call.
def generate_plots(dataset, plot_type='summary', output_dir=None, **kwargs) -> plt.Figure:
     """Standalone function to generate plots using PlotManager."""
     pm = PlotManager(output_dir=output_dir if output_dir else config.PLOT_DIR)
     return pm.generate_plots(dataset, plot_type, **kwargs)