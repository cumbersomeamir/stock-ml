"""Logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional

from trading_lab.config.settings import get_settings


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    auto_log_file: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (if None and auto_log_file=True, creates one automatically)
        auto_log_file: If True and log_file is None, automatically create log file in artifacts/logs

    Returns:
        Configured logger instance
    """
    settings = get_settings()
    log_level = level or settings.log_level

    # Create logger
    logger = logging.getLogger("trading_lab")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with rich formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None and auto_log_file:
        # Auto-create log file in artifacts/logs directory
        log_dir = settings.get_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"trading_lab_{timestamp}.log"

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger

