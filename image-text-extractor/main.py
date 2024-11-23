# main.py

"""
Entry point for the VisionOCR application.
"""

import logging
import uvicorn
from config import LOGGING_LEVEL

def setup_logging():
    """
    Set up logging configuration.
    """
    numeric_level = getattr(logging, LOGGING_LEVEL.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {LOGGING_LEVEL}')

    logging.basicConfig(level=numeric_level)
    logging.getLogger("PIL").setLevel(logging.WARNING)

if __name__ == "__main__":
    setup_logging()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
