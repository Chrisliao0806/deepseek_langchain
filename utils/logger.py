import logging
import sys

def setup_logging(log_level: str, log_filename: str = "code_translator.log"):
    """
    Set up logging to both console and a specified log file.
    :param log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    :param log_filename: The file path to write the log records.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # setting a logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Clear default handlers to avoid duplicate additions
    while logger.handlers:
        logger.handlers.pop()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # FileHandler: Output logs to a file
    file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # StreamHandler: Output logs to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)