# api.py

"""
FastAPI interface for the VisionOCR application.
"""

import base64
import imghdr
import logging
import os
from typing import Dict

from config import SUPPORTED_IMAGE_TYPES
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from image_processor import ImageProcessor
from ocr_agent import OcrAgent
from starlette.requests import Request

app = FastAPI()
image_processor = ImageProcessor()

# Default system prompt
SYSTEM_PROMPT = """Convert the provided image into text. Ensure that all content from the page is included."""


def create_response(success: bool, data: Dict = None, error: Dict = None) -> Dict:
    """
    Create a standardized JSON response.

    Args:
        success (bool): Indicates if the request was successful.
        data (Dict): The data to include in the response.
        error (Dict): Error details if an error occurred.

    Returns:
        Dict: Standardized JSON response.
    """
    response = {"success": success, "data": data, "error": error}
    return response


@app.post("/ocr", response_model=Dict)
async def perform_ocr(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Form(...),
    system_prompt: str = Form(SYSTEM_PROMPT),
) -> JSONResponse:
    """
    Endpoint to perform OCR on an uploaded image using Together AI API.

    Args:
        file (UploadFile): Image file uploaded by the user.
        api_key (str): Together AI API key provided by the user.
        system_prompt (str): Prompt to guide the model (optional).

    Returns:
        JSONResponse: JSON containing the extracted text.
    """
    try:
        # Validate image type
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_IMAGE_TYPES:
            logging.error(f"Unsupported file type: {ext}")
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        # Read image and encode it
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        mime_type = imghdr.what(None, h=contents)
        if mime_type:
            mime_type = f"image/{mime_type}"
        else:
            mime_type = "image/jpeg"  # Default MIME type

        # Initialize OCR agent
        ocr_agent = OcrAgent(api_key=api_key)

        # Process image
        text = ocr_agent.analyze_image(base64_image, mime_type, system_prompt)

        # Create success response
        response_data = {"text": text}
        response = create_response(success=True, data=response_data)
        return JSONResponse(content=response, status_code=200)

    except HTTPException as http_exc:
        # Handle known HTTP exceptions
        response = create_response(
            success=False,
            error={"code": http_exc.status_code, "message": http_exc.detail},
        )
        return JSONResponse(content=response, status_code=http_exc.status_code)

    except Exception as e:
        # Log the error
        logging.error(f"Error processing image: {str(e)}")

        # Create error response
        response = create_response(
            success=False, error={"code": 500, "message": "Internal Server Error"}
        )
        return JSONResponse(content=response, status_code=500)
