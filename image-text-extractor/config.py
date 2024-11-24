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
# TOGETHER_MODEL_NAME = "meta-llama/Llama-Vision-Free"

# Ollama model configuration
OLLAMA_MODEL_NAME = "llama3.2-vision"  # Matches your local Ollama model name

# Other configurations can be added here

# Add these configurations
SUPPORTED_PROVIDERS = ["together", "ollama"]
DEFAULT_PROVIDER = "ollama"

# Default system prompt
SYSTEM_PROMPT = """Extract meaningful text content from the image while following these rules:

1. Focus on human-readable text content only
2. Ignore and exclude:
   - XML/HTML tags
   - Script contents
   - Debugging information
   - Internal system identifiers
3. For addresses and locations:
   - Keep them in their original format
   - Maintain proper spacing and punctuation
4. For mixed-language content:
   - Keep text in its original language
   - Maintain proper character encoding
5. Remove any duplicate content
6. Organize the output in a clear, structured format
7. Preserve the original formatting of:
   - Headers and titles
   - Main body text
   - Lists and bullet points
   - Contact information
   - Geographic locations

Output the text in a clean, human-readable format."""
