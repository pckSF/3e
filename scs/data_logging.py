from __future__ import annotations

import atexit
import csv
from datetime import datetime
import json
import logging
import multiprocessing
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    TypeAlias,
)

from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

PrimitiveTypes: TypeAlias = str | int | float | bool


class MessageShutdown:
    pass


class DataLoggerMessage(NamedTuple):
    data: Any
    filename: str


class MessageCSVRow(DataLoggerMessage):
    data: jax.Array | np.ndarray | list[int | float]


class MessageMetadata(DataLoggerMessage):
    data: dict[str, PrimitiveTypes | list[PrimitiveTypes]]


class MessageCheckpoint(DataLoggerMessage):
    data: nnx.State


def _flatten_array(array: jax.Array | np.ndarray) -> list[int | float]:
    array = np.asarray(array)
    if array.ndim == 0:
        return [array.item()]
    if array.ndim == 1:
        return array.tolist()
    if array.ndim == 2:
        return array.reshape(-1).tolist()
    raise ValueError(
        f"Array with ndim > 2 not supported; Received array with shape {array.shape}"
    )


def _preprocess_csv_row(
    data: jax.Array | np.ndarray | list[int | float],
) -> list[int | float]:
    if isinstance(data, list):
        return data
    if isinstance(data, (jax.Array, np.ndarray)):
        return _flatten_array(data)
    raise TypeError(
        f"Unsupported type for CSV row: Expected list, jax.Array, or np.ndarray; "
        f"received {type(data)}."
    )


def _write_csv_row(log_dir: Path, filename: str, row: list[int | float]) -> None:
    filepath = (log_dir / filename).with_suffix(".csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row)


def _save_metadata(
    log_dir: Path,
    filename: str,
    data: dict[str, PrimitiveTypes | list[PrimitiveTypes]],
) -> None:
    filepath = (log_dir / filename).with_suffix(".json")
    if filepath.exists():
        raise FileExistsError(f"Metadata file {filepath} already exists.")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)


def _save_checkpoint(
    log_dir: Path,
    filename: str,
    data: nnx.State,
) -> None:
    if not isinstance(data, nnx.State):
        raise TypeError(
            f"Unsupported type for checkpoint: Expected nnx.State; "
            f"received {type(data)}."
        )
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(
            log_dir / filename,
            data,
        )


INSTRUCTION_SET: dict[
    type[DataLoggerMessage],
    tuple[Callable[[Path, str, Any], None], Callable[[Any], Any]],
] = {
    MessageCSVRow: (_write_csv_row, _preprocess_csv_row),
    MessageMetadata: (_save_metadata, lambda x: x),
    MessageCheckpoint: (_save_checkpoint, lambda x: x),
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


def spawn_logger_process(log_dir: str | Path) -> Connection:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_log_dir = Path(log_dir).resolve() / timestamp
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    logger = _create_file_logger(resolved_log_dir, timestamp)
    logger.info(f"DataLogger initialized at {resolved_log_dir}")

    def logger_process(pipe_conn: Connection) -> None:
        while True:
            try:
                message: DataLoggerMessage | MessageShutdown = pipe_conn.recv()
                if isinstance(message, DataLoggerMessage):
                    _process_message(resolved_log_dir, message, logger)
                elif isinstance(message, MessageShutdown):
                    logger.info("Shutdown command received. Exiting.")
                    break
                else:
                    logger.info(
                        f"Received invalid message type. Expected DataLoggerMessage "
                        f"or MessageShutdown; received {type(message)}"
                    )

            except EOFError:
                logger.info("Pipe connection closed unexpectedly. Shutting down.")
                break
            except Exception as e:
                logger.exception(
                    f"An unexpected error occurred in the logger process: {e}"
                )

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


def _process_message(
    log_dir: Path, message: DataLoggerMessage, logger: logging.Logger
) -> None:
    write_function, preprocess_function = INSTRUCTION_SET[type(message)]
    try:
        processed_data = preprocess_function(message.data)
        write_function(log_dir, message.filename, processed_data)
        logger.info(f"Processed message for {type(message)}: {message.filename}")
    except Exception as e:
        logger.exception(
            f"Error processing message {type(message)} with {message.filename}: {e} "
        )
