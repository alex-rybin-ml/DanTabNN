"""Logging configuration."""

import logging
import os
import sys

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger.

    Parameters
    ----------
    name : str
        Logger name (usually __name__),
    level : int
        logging level.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid duplicates handles
        for h in list(logger.handlers):
            logger.removeHandler(h)
    logger.setLevel(level)
    # Check env var for clean output (no timestamps in benchmark mode)
    if os.environ.get("DANTABNN_CLEAN_LOG", "0") == "1":
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
