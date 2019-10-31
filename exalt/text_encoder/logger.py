import logging
import os
import sys
from typing import Optional
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
)


def get_console_handler() -> logging.StreamHandler:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(file_path: str) -> logging.FileHandler:
    file_handler = TimedRotatingFileHandler(file_path, when="midnight")
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(
    logger_name,
    file_path: Optional[str] = None,
    level: str = "INFO",
    propagate: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(get_console_handler())
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logger.addHandler(get_file_handler(file_path))
    logger.propagate = propagate
    return logger

