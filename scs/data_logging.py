from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path


def _create_file_logger(log_dir: Path, timestamp: str) -> logging.Logger:
    """Creates a logger that writes to a file in the specified directory."""
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
    def __init__(self, log_dir: str | Path = "logs") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(log_dir).resolve()
        self.log_dir = log_dir / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = _create_file_logger(self.log_dir, timestamp)
        self.logger.info(f"DataLogger initialized at {self.log_dir}")
