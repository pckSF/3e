from __future__ import annotations

import atexit
import csv
from datetime import datetime
import json
import logging
import multiprocessing
from pathlib import Path
import pickle
from typing import (
    TYPE_CHECKING,
    Any,
)

import jax
import numpy as np

if TYPE_CHECKING:
    from multiprocessing.connection import Connection


def _flatten_array(array: jax.Array | np.ndarray) -> list[int | float]:
    array = np.asarray(array)
    if array.ndim == 0:
        return [array.item()]
    elif array.ndim == 1:
        return array.tolist()
    elif array.ndim == 2:
        return array.reshape(-1).tolist()
    else:
        raise ValueError(
            f"Array with ndim > 2 not supported; "
            f"Received array with shape {array.shape}"
        )


def _preprocess_csv_row(row: Any) -> list[int | float]:
    if isinstance(row, list):
        return row
    elif isinstance(row, (jax.Array, np.ndarray)):
        return _flatten_array(row)
    else:
        raise TypeError(
            f"Unsupported type for CSV row: {type(row)}. "
            f"Expected list, jax.Array, or np.ndarray."
        )


def _write_csv_row(log_dir: Path, filename: str, row: list[int | float]) -> None:
    filepath = (log_dir / filename).with_suffix(".csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row)


def _process_metadata(
    metadata: dict[str, str | int | float],
) -> dict[str, str | int | float]:
    if not isinstance(metadata, dict):
        raise TypeError(f"Metadata must be a dictionary; received {type(metadata)}")
    return metadata


def _save_metadata(
    log_dir: Path, filename: str, data: dict[str, str | int | float]
) -> None:
    filepath = (log_dir / filename).with_suffix(".json")
    if filepath.exists():
        raise FileExistsError(f"Metadata file {filepath} already exists.")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)


def _preprocess_checkpoint(checkpoint: Any) -> Any:
    return checkpoint


def _save_checkpoint(
    log_dir: Path,
    filename: str,
    checkpoint: Any,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    nums = (
        int(p.stem.split("_")[-1])
        for p in log_dir.glob(f"{filename}_*.pkl")
        if p.stem.split("_")[-1].isdigit()
    )
    count = max(nums, default=0) + 1
    filepath = (log_dir / f"{filename}_{count:05d}").with_suffix(".pkl")
    with open(filepath, "wb") as pkl_file:
        pickle.dump(checkpoint, pkl_file)


INSTRUCTION_SET = {
    "csv": (_write_csv_row, _preprocess_csv_row),
    "metadata": (_save_metadata, _process_metadata),
    "checkpoint": (_save_checkpoint, _preprocess_checkpoint),
}


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


def spawn_logger_process(log_dir: Path) -> Connection:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(log_dir).resolve()
    log_dir = log_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = _create_file_logger(log_dir, timestamp)
    logger.info(f"DataLogger initialized at {log_dir}")

    def logger_process(pipe_conn: Connection) -> None:
        while True:
            try:
                message = (
                    pipe_conn.recv()
                )  # This call blocks until a message is received
                if isinstance(message, tuple):
                    if not _check_message(message, logger):
                        continue
                    _process_message(log_dir, message, logger)
                elif isinstance(message, str) and message == "shutdown":
                    logger.info("Shutdown command received. Exiting.")
                    break
                else:
                    logger.info(f"No process defined for message: {message}")

            except EOFError:
                logger.info("Pipe connection closed unexpectedly. Shutting down.")
                break

    pipe_send, pipe_receive = multiprocessing.Pipe()
    process = multiprocessing.Process(target=logger_process, args=(pipe_receive,))
    process.daemon = True
    process.start()

    def cleanup():
        logger.info("Parent process is exiting. Terminating writer process...")
        if process.is_alive():
            pipe_send.close()
            process.terminate()
            process.join(timeout=1)
        logger.info("Writer process terminated.")

    atexit.register(cleanup)

    return pipe_send


def _check_message(message: tuple, logger: logging.Logger) -> bool:
    if not len(message) == 3:
        logger.warning(f"Expected message tuple of length 3, got {len(message)}")
        return False
    if not isinstance(message[1], str):
        logger.warning(
            f"Expected second element of message to be str; received {type(message[1])}"
        )
        return False
    if message[2] not in INSTRUCTION_SET:
        logger.warning(
            f"Expected third element of message to be one of {INSTRUCTION_SET}; "
            f"received {message[2]}"
        )
        return False
    return True


def _process_message(log_dir: Path, message: tuple, logger: logging.Logger) -> None:
    data, filename, instruction = message
    process_function, preprocess_function = INSTRUCTION_SET[instruction]
    try:
        data = preprocess_function(data)
        process_function(log_dir, filename, data)
        logger.info(f"Processed message for {instruction}: {filename}")
    except Exception as e:
        logger.error(f"Error processing message: {message} : {e}")
