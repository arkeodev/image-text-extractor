# ocr_agent.py

"""
Module for the OCR agent using Together AI API.
"""

import logging

from config import TOGETHER_MODEL_NAME
from together import Together


class OcrAgent:
    """
    OCR agent that interacts with the Together AI API to perform OCR on images.
    """

    def __init__(self, api_key: str, model_name: str = TOGETHER_MODEL_NAME):
        """
        Initialize the OCR agent.

        Args:
            api_key (str): API key for Together AI.
            model_name (str): The Together AI model to use.
        """
        self.client = Together(api_key=api_key)
        self.model_name = model_name

    def extract_text(self, base64_image: str) -> str:
        """
        Extract text from an image using Together AI's vision model.

        Args:
            base64_image: Base64 encoded image string
        Returns:
            Extracted text from the image
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please read all the text into markdown format",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
            )

            # Extract the response text
            if hasattr(response, "choices") and len(response.choices) > 0:
                return response.choices[0].message.content
            return ""

        except Exception as e:
            logging.error(f"Error extracting text from image: {str(e)}")
            raise
