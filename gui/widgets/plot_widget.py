from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt # Import for type hinting if needed

class PlotWidget(QWidget):
    """A custom widget to display a Matplotlib figure."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), dpi=100) # Initial figure
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_plot(self, fig: plt.Figure):
        """Clears the current figure and draws the new one."""
        self.figure.clear()
        # Matplotlib's FigureCanvas expects a Figure instance.
        # We need to effectively copy the contents of the generated `fig`
        # onto `self.figure`. A simple way is to redraw.
        # Or, more efficiently if possible, replace the canvas figure.

        # Method 1: Replace the figure (might have side effects?)
        # self.canvas.figure = fig
        # fig.set_canvas(self.canvas)

        # Method 2: Clear internal figure and use fig's axes onto self.figure
        # This is generally safer if axes layouts are compatible.
        if fig.get_axes():
            try:
                # If it's a simple plot with one axes
                if len(fig.get_axes()) == 1:
                    ax_orig = fig.get_axes()[0]
                    ax_new = self.figure.add_subplot(111)
                    # Copy relevant properties (more might be needed)
                    ax_new.set_title(ax_orig.get_title())
                    ax_new.set_xlabel(ax_orig.get_xlabel())
                    ax_new.set_ylabel(ax_orig.get_ylabel())
                    ax_new.grid(ax_orig.get_visible()) # Check grid state

                    # Copy plotted data
                    for line in ax_orig.get_lines():
                        ax_new.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(),
                                    color=line.get_color(), marker=line.get_marker(),
                                    linestyle=line.get_linestyle())
                    for collection in ax_orig.collections: # For scatter plots etc.
                        # This is complex - might need specific handling per collection type
                         try: # Basic scatter copy
                             ax_new.scatter(collection.get_offsets()[:,0], collection.get_offsets()[:,1],
                                            s=collection.get_sizes(), c=collection.get_facecolors(),
                                            marker=collection.get_paths()[0] if collection.get_paths() else 'o',
                                            label=collection.get_label(), edgecolors=collection.get_edgecolors())
                         except Exception:
                             pass # Ignore complex collections for now

                    if ax_orig.get_legend():
                         ax_new.legend()

                else: # Handle multi-panel figures (more complex copy)
                    # This requires replicating the gridspec and copying each axes
                    # For simplicity now, just show a message
                     ax = self.figure.add_subplot(111)
                     ax.text(0.5, 0.5, 'Multi-panel plot display needs specific handling',
                             ha='center', va='center')
                     print("Warning: Displaying complex multi-panel plots needs specific implementation.")

            except Exception as e:
                 print(f"Error copying plot elements: {e}")
                 self.figure.clear()
                 ax = self.figure.add_subplot(111)
                 ax.text(0.5, 0.5, 'Error displaying plot', ha='center', va='center')
        else:
             # If the input figure has no axes (e.g., empty)
             ax = self.figure.add_subplot(111)
             ax.text(0.5, 0.5, 'No plot data received', ha='center', va='center')


        self.canvas.draw()

    def clear_plot(self):
        """Clears the plot area."""
        self.figure.clear()
        # Add a placeholder axes to show it's cleared
        ax = self.figure.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, 'Plot Area', ha='center', va='center', fontsize=12, color='gray')
        self.canvas.draw()