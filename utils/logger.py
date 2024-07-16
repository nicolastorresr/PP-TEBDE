import logging
import os
from logging.handlers import RotatingFileHandler
from utils.config import LoggingConfig

def setup_logger(config: LoggingConfig) -> logging.Logger:
    """
    Set up and configure the logger based on the provided configuration.

    Args:
        config (LoggingConfig): Logging configuration object.

    Returns:
        logging.Logger: Configured logger object.
    """
    # Create logger
    logger = logging.getLogger('PP-TEBDE')
    logger.setLevel(config.level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Create file handler and set level to debug
    os.makedirs(os.path.dirname(config.file_path), exist_ok=True)
    file_handler = RotatingFileHandler(
        config.file_path, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def get_logger() -> logging.Logger:
    """
    Get the configured logger.

    Returns:
        logging.Logger: Configured logger object.
    """
    return logging.getLogger('PP-TEBDE')

class LoggerMixin:
    """
    A mixin class to add logging capabilities to other classes.
    """
    @property
    def logger(self):
        return get_logger()

def log_exception(logger: logging.Logger, exception: Exception):
    """
    Log an exception with its traceback.

    Args:
        logger (logging.Logger): The logger to use.
        exception (Exception): The exception to log.
    """
    logger.exception(f"An exception occurred: {str(exception)}")

def log_error(logger: logging.Logger, message: str):
    """
    Log an error message.

    Args:
        logger (logging.Logger): The logger to use.
        message (str): The error message to log.
    """
    logger.error(message)

def log_warning(logger: logging.Logger, message: str):
    """
    Log a warning message.

    Args:
        logger (logging.Logger): The logger to use.
        message (str): The warning message to log.
    """
    logger.warning(message)

def log_info(logger: logging.Logger, message: str):
    """
    Log an info message.

    Args:
        logger (logging.Logger): The logger to use.
        message (str): The info message to log.
    """
    logger.info(message)

def log_debug(logger: logging.Logger, message: str):
    """
    Log a debug message.

    Args:
        logger (logging.Logger): The logger to use.
        message (str): The debug message to log.
    """
    logger.debug(message)
