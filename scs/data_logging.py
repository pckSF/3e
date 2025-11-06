from __future__ import annotations

import csv
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
)

from flax import nnx
import numpy as np
import orbax.checkpoint as ocp

if TYPE_CHECKING:
    import jax

PrimitiveTypes: TypeAlias = str | int | float | bool


def _create_file_logger(log_dir: Path, timestamp: str) -> logging.Logger:
    """Creates a logger that writes to a file in the specified directory.

    Args:
        log_dir: The directory where the log file will be created.
        timestamp: A timestamp string to create a unique logger name.
    """
    # Create a logger instance specific to the containing object
    logger = logging.getLogger(f"{__name__}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_file = log_dir / "data_logger.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class DataLogger:
    def __init__(self, log_dir: str | Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_log_dir = Path(log_dir).resolve() / timestamp
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir: Path = resolved_log_dir

        self.logger: logging.Logger = _create_file_logger(resolved_log_dir, timestamp)
        self.logger.info(f"DataLogger initialized at {resolved_log_dir}")

        self.checkpointer: ocp.Checkpointer = ocp.StandardCheckpointer()

    def wait_until_finished(self) -> None:
        """Waits for all asynchronous checkpointing operations to complete."""
        self.checkpointer.wait_until_finished()

    def log_info(self, message: str) -> None:
        """Logs an informational message."""
        self.logger.info(message)

    def _flatten_array(self, array: jax.Array | np.ndarray) -> list[int | float]:
        """Flattens a JAX or NumPy array into a 1D list of numbers."""
        array = np.asarray(array)
        if array.ndim == 0:
            return [array.item()]
        if array.ndim == 1:
            return array.tolist()
        if array.ndim == 2:
            return array.reshape(-1).tolist()
        raise ValueError(
            f"Array with ndim > 2 not supported; "
            f"Received array with shape {array.shape}"
        )

    def save_csv_row(
        self, filename: str, row: jax.Array | np.ndarray | list[Any]
    ) -> None:
        """Appends a single row to a specified CSV file in the log directory.

        Args:
            filename: The name of the CSV file (without extension).
            row: A list or array of values to be written as a row in the CSV.
        """
        filepath = (self.log_dir / filename).with_suffix(".csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(row, list):
            row = self._flatten_array(row)
        with open(filepath, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)

    def store_metadata(
        self,
        filename: str,
        data: dict[str, PrimitiveTypes | list[PrimitiveTypes]],
    ) -> None:
        """Saves a dictionary of metadata as a JSON file.

        Args:
            filename: The name of the JSON file (without extension).
            data: A dictionary containing metadata.
        """
        filepath = (self.log_dir / filename).with_suffix(".json")
        if filepath.exists():
            raise FileExistsError(f"Metadata file {filepath} already exists.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as json_file:
            json.dump(data, json_file, indent=4)
        self.logger.info(f"Metadata saved to {filepath}")

    def save_checkpoint(
        self,
        filename: str,
        data: nnx.State,
    ) -> None:
        """Saves an NNX State object as a checkpoint.

        Args:
            filename: The name of the checkpoint directory.
            data: The NNX State object to be saved.
        """
        if not isinstance(data, nnx.State):
            raise TypeError(
                f"Unsupported type for checkpoint: Expected nnx.State; "
                f"received {type(data)}."
            )
        checkpoint_numbers = (
            int(p.stem.split("_")[-1])
            for p in self.log_dir.glob(f"{filename}_*")
            if p.stem.split("_")[-1].isdigit()
        )
        count = max(checkpoint_numbers, default=0) + 1
        self.checkpointer.save(
            self.log_dir / f"{filename}_{count:05d}",
            data,
        )
        self.logger.info(
            f"Checkpoint saved to {self.log_dir / f'{filename}_{count:05d}'}"
        )
