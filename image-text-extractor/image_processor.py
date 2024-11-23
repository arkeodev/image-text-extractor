# image_processor.py

"""
Module for processing images.
"""

import base64
import imghdr
import logging
import os
from typing import Optional

from config import SUPPORTED_IMAGE_TYPES


class ImageProcessor:
    """
    Class responsible for handling image validation and encoding.
    """

    def validate_image(self, image_path: str) -> bool:
        """
        Validate if the image exists and is of a supported type.

        Args:
            image_path (str): Path to the image file.

        Returns:
            bool: True if image is valid, False otherwise.
        """
        if not os.path.exists(image_path):
            logging.error(f"Image file not found: {image_path}")
            return False

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_IMAGE_TYPES:
            logging.error(f"Unsupported image type: {ext}")
            return False

        return True

    def get_mime_type(self, image_path: str) -> str:
        """
        Determine MIME type based on actual image format using imghdr.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: MIME type of the image.
        """
        img_type = imghdr.what(image_path)
        if img_type:
            return f"image/{img_type}"
        # Fallback to extension-based detection
        extension = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(extension, "image/jpeg")

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 with error handling.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded string of the image.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logging.error(f"Error encoding image: {str(e)}")
            raise
