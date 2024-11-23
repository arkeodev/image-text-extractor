# main.py

"""
Entry point for the VisionOCR application.
"""

import logging

import uvicorn
from config import setup_logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        setup_logging()
        logger.info("Starting VisionOCR application...")
        uvicorn.run("api:app", host="0.0.0.0", port=8008, reload=True)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}", exc_info=True)
        raise
