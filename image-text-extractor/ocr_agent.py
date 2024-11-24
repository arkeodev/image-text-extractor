# ocr_agent.py

"""
Module for the OCR agent using Together AI API.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import ollama
import requests
from config import OLLAMA_MODEL_NAME, TOGETHER_MODEL_NAME
from together import Together


class BaseOcrAgent(ABC):
    """Base class for OCR agents."""

    @abstractmethod
    def extract_text(self, base64_image: str) -> str:
        """Extract text from base64 encoded image."""
        pass


class TogetherOcrAgent(BaseOcrAgent):
    """OCR agent that uses Together AI API."""

    def __init__(self, api_key: str, model_name: str = TOGETHER_MODEL_NAME):
        self.client = Together(api_key=api_key)
        self.model_name = model_name

    def extract_text(self, base64_image: str) -> str:
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

            if hasattr(response, "choices") and len(response.choices) > 0:
                return response.choices[0].message.content
            return ""

        except Exception as e:
            logging.error(f"Error extracting text from image: {str(e)}")
            raise


class OllamaOcrAgent(BaseOcrAgent):
    """OCR agent that uses local Ollama instance."""

    def __init__(self, model_name: str = OLLAMA_MODEL_NAME):
        try:
            self.model_name = model_name
            self.client = ollama

            logging.info(f"Initializing Ollama agent with model: {self.model_name}")

            # Check if model exists and is responding
            try:
                models = self.client.list()
                model_exists = any(
                    str(model).split(":")[0] == self.model_name.split(":")[0]
                    for model in models.models
                )

                if not model_exists:
                    logging.info(
                        f"Model {self.model_name} not found locally. Pulling from repository..."
                    )
                    self.client.pull(self.model_name)
                    logging.info(f"Successfully pulled model {self.model_name}")
                else:
                    logging.info(f"Model {self.model_name} found locally")

                # Test model with a simple prompt
                logging.info("Testing model responsiveness...")
                test_response = self.client.generate(
                    model=self.model_name,
                    prompt="Test prompt",
                    options={
                        "num_predict": 1,
                        "temperature": 0.1,
                    },
                )
                if test_response:
                    logging.info("Model is responsive")

            except Exception as e:
                logging.error(f"Error during model initialization: {str(e)}")
                raise

        except ImportError:
            raise ImportError("Please install ollama package: pip install ollama")
        except Exception as e:
            logging.error(f"Error initializing Ollama agent: {str(e)}")
            raise

    def extract_text(self, base64_image: str) -> str:
        try:
            import base64
            import tempfile
            import time

            logging.info(f"Starting new image processing with model {self.model_name}")
            start_time = time.time()

            # Convert base64 to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(base64.b64decode(base64_image))
                temp_path = temp_file.name
                logging.info(f"Temporary file created at: {temp_path}")

            try:
                logging.info("Sending request to Ollama model...")

                # Use chat instead of generate for vision models
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": "Please read and describe all the text visible in this image:",
                            "images": [temp_path],
                        }
                    ],
                    options={
                        "temperature": 0.1,
                        "top_k": 10,
                        "top_p": 0.9,
                        "num_thread": 1,
                    },
                )

                inference_time = time.time() - start_time
                logging.info(
                    f"Model inference completed in {inference_time:.2f} seconds"
                )

                # Clean up the temporary file
                os.unlink(temp_path)
                logging.info("Cleaned up temporary file")

                if response and "response" in response:
                    logging.info(f"Response:")
                    extracted_text = response["response"]
                    logging.info(
                        f"Successfully extracted text (length: {len(extracted_text)} chars)"
                    )
                    total_time = time.time() - start_time
                    logging.info(f"Total processing time: {total_time:.2f} seconds")
                    return extracted_text
                else:
                    raise Exception("Unexpected response format from Ollama")

            except Exception as e:
                logging.error(f"Error during model inference: {str(e)}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logging.info("Cleaned up temporary file after error")
                raise

        except Exception as e:
            logging.error(f"Error extracting text from image using Ollama: {str(e)}")
            raise


def create_ocr_agent(provider: str, api_key: Optional[str] = None) -> BaseOcrAgent:
    """Factory function to create appropriate OCR agent."""
    if provider == "together":
        if not api_key:
            raise ValueError("API key required for Together AI")
        return TogetherOcrAgent(api_key=api_key)
    elif provider == "ollama":
        return OllamaOcrAgent()
    else:
        raise ValueError(f"Unsupported provider: {provider}")
