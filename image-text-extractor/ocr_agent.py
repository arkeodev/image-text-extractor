# ocr_agent.py

"""
Module for the OCR agent using Langchain and Together AI API.
"""

import logging
from typing import List, Dict
from langchain.llms import TogetherAI
from langchain import PromptTemplate, LLMChain
from config import TOGETHER_MODEL_NAME

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
        self.api_key = api_key
        self.model_name = model_name
        self.llm = TogetherAI(model_name=self.model_name, api_key=self.api_key)

    def analyze_image(self, base64_image: str, mime_type: str, system_prompt: str) -> str:
        """
        Analyze the image using Together AI API.

        Args:
            base64_image (str): Base64 encoded image string.
            mime_type (str): MIME type of the image.
            system_prompt (str): The prompt to guide the model.

        Returns:
            str: Extracted text from the image.
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]
            # Use Langchain's LLMChain to interact with Together AI
            prompt = PromptTemplate(
                input_variables=["messages"],
                template="{messages}"
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.run(messages=messages)
            return response
        except Exception as e:
            logging.error(f"Error during image analysis: {str(e)}")
            raise
