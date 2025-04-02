import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex
from PySide6.QtGui import QColor

class PandasTableModel(QAbstractTableModel):
    """A model to interface a pandas DataFrame with QTableView."""
    def __init__(self, data=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._data = data

    def rowCount(self, parent=QModelIndex()) -> int:
        """Return the number of rows in the DataFrame."""
        if parent.isValid():
            return 0
        return len(self._data)

    def columnCount(self, parent=QModelIndex()) -> int:
        """Return the number of columns in the DataFrame."""
        if parent.isValid():
            return 0
        return len(self._data.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Return data specific to the role."""
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if not 0 <= row < self.rowCount():
            return None
        if not 0 <= col < self.columnCount():
            return None

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            value = self._data.iloc[row, col]
            # Format values for display if needed (e.g., floats)
            if isinstance(value, (float, pd.Float64Dtype)):
                 # Handle potential pandas nullable floats
                 if pd.isna(value): return ""
                 return f"{value:.4f}" # Adjust precision as needed
            if pd.isna(value):
                 return "" # Display empty string for NaNs
            return str(value)

        # Example: Add background color for specific conditions
        # if role == Qt.BackgroundRole:
        #     # Example: Highlight negative charge balance
        #     if self._data.columns[col] == 'charge_balance':
        #         try:
        #              val = float(self._data.iloc[row, col])
        #              if abs(val) > 0.1: # Tolerance
        #                  return QColor(Qt.yellow)
        #         except (ValueError, TypeError):
        #              pass

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        """Return header data."""
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self._data.columns[section])
            except IndexError:
                return None
        elif orientation == Qt.Orientation.Vertical:
            try:
                return str(self._data.index[section])
            except IndexError:
                return None
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:
        """Set data (if model is editable - not typically needed for display)."""
        if role == Qt.ItemDataRole.EditRole and index.isValid():
            row = index.row()
            col = index.column()
            try:
                self._data.iloc[row, col] = value
                self.dataChanged.emit(index, index, [role])
                return True
            except Exception as e:
                print(f"Error setting data: {e}") # Log properly
                return False
        return False

    def flags(self, index) -> Qt.ItemFlag:
        """Return item flags (e.g., editable, selectable)."""
        # Make cells selectable but not editable by default
        flags = super().flags(index)
        # Uncomment to make editable:
        # flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self._data

    def updateData(self, data: pd.DataFrame):
         """Update the model with a new DataFrame."""
         self.beginResetModel()
         self._data = data.copy() # Work with a copy
         self.endResetModel()