from __future__ import annotations

import atexit
from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection


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
                    pass
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
