# image_processor.py

"""
Module for processing images.
"""

import base64
import imghdr
import logging
import os
from io import BytesIO
from typing import Optional, Tuple

from config import SUPPORTED_IMAGE_TYPES, setup_logging
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Class responsible for handling image validation and encoding.
    """

    MAX_IMAGE_SIZE = (1024, 1024)  # Maximum dimensions for processed images

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

    def process_image(self, image_bytes: bytes) -> Tuple[bytes, str]:
        """
        Process and optimize image for OCR.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Tuple[bytes, str]: Processed image bytes and MIME type
        """
        try:
            # Open image with PIL
            with Image.open(BytesIO(image_bytes)) as img:
                # Convert to RGB if necessary
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                # Calculate new dimensions while maintaining aspect ratio
                width, height = img.size
                if width > self.MAX_IMAGE_SIZE[0] or height > self.MAX_IMAGE_SIZE[1]:
                    logger.info(
                        f"Resizing image from {img.size} to fit within {self.MAX_IMAGE_SIZE}"
                    )
                    img.thumbnail(self.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

                # Save optimized image
                output = BytesIO()
                img.save(output, format="JPEG", quality=85, optimize=True)
                processed_bytes = output.getvalue()

                logger.info(
                    f"Image processed: Original size: {len(image_bytes)}, New size: {len(processed_bytes)}"
                )
                return processed_bytes, "image/jpeg"

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            raise
