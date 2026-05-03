from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class InputLoader(ABC):
    """Base class for input loaders."""

    def __init__(self, load_dir: str | Path):
        self.load_dir = Path(load_dir)

    @property
    def source_dir(self) -> Path:
        return self.load_dir

    @abstractmethod
    def load_inputs(self) -> list[tuple[Path, str]]:
        """Load inputs from the configured source directory.

        Returns:
            A list of tuples containing the file path the input was loaded from and the input string itself.
        """
        raise NotImplementedError


class HuiInputLoader(InputLoader):
    """Loader for Hui inputs from CSV files."""

    def __init__(self, load_dir: str | Path = "results/Hui", column_index: int = 1):
        super().__init__(load_dir)
        self.column_index = column_index

    def load_inputs(self) -> list[tuple[Path, str]]:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Hui input directory not found: {self.source_dir}")

        csv_files = list(self.source_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.source_dir}")

        inputs: list[tuple[Path, str]] = []
        for csv_file in csv_files:
            try:
                user_input = get_csv_column_title(csv_file, self.column_index)
                inputs.append((csv_file, user_input))
            except Exception as e:
                logger.error(f"Failed to load input from {csv_file}: {e}")

        return inputs


def get_csv_column_title(csv_file: Path, column_index: int = 1) -> str:
    """Extract the title of a specific column from a CSV file."""
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        if column_index < len(headers):
            return headers[column_index]
        raise IndexError(f"Column index {column_index} not found in CSV")
