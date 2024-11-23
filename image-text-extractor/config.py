# config.py

"""
Configuration settings for the VisionOCR application.
"""

import logging
import os

# Logging configuration
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging():
    """
    Set up logging configuration for all modules.
    """
    numeric_level = getattr(logging, LOGGING_LEVEL.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {LOGGING_LEVEL}")

    logging.basicConfig(
        level=numeric_level,
        format=LOGGING_FORMAT,
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler("app.log"),  # File handler
        ],
    )

    # Set specific loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)


# Supported image types
SUPPORTED_IMAGE_TYPES = [".png", ".jpg", ".jpeg", ".gif", ".webp"]

# Together AI model configuration
TOGETHER_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"

# Other configurations can be added here
