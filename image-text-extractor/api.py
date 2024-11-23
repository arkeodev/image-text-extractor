# api.py

"""
FastAPI interface for the VisionOCR application.
"""

import base64
import imghdr
import logging
import os
from typing import Dict

from config import SUPPORTED_IMAGE_TYPES, setup_logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from image_processor import ImageProcessor
from ocr_agent import OcrAgent
from starlette.requests import Request

# Initialize logger for this module
logger = logging.getLogger(__name__)

app = FastAPI()
image_processor = ImageProcessor()

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


@app.on_event("startup")
async def startup_event():
    """Initialize application settings on startup."""
    setup_logging()
    logger.info("FastAPI application starting up...")


def create_response(success: bool, data: Dict = None, error: Dict = None) -> Dict:
    """Create a standardized JSON response."""
    response = {"success": success, "data": data, "error": error}
    return response


@app.post("/ocr", response_model=Dict)
async def perform_ocr(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Form(...),
    system_prompt: str = Form(SYSTEM_PROMPT),
) -> JSONResponse:
    """Process OCR request."""
    try:
        logger.info(f"Processing OCR request for file: {file.filename}")

        # Process image using ImageProcessor
        content = await file.read()
        processed_image, mime_type = image_processor.process_image(content)
        base64_image = base64.b64encode(processed_image).decode("utf-8")
        logger.info("Processing image with OCR agent")

        # Initialize OCR agent and extract text
        ocr_agent = OcrAgent(api_key=api_key)
        text = ocr_agent.extract_text(base64_image)

        # Create success response
        response = create_response(success=True, data={"text": text})
        logger.info("Successfully processed image")
        return JSONResponse(content=response)

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception: {http_exc.detail}", exc_info=True)
        response = create_response(
            success=False,
            error={"code": http_exc.status_code, "message": http_exc.detail},
        )
        return JSONResponse(content=response, status_code=http_exc.status_code)

    except Exception as e:
        logger.error(f"Unexpected error processing image: {str(e)}", exc_info=True)
        response = create_response(
            success=False, error={"code": 500, "message": "Internal Server Error"}
        )
        return JSONResponse(content=response, status_code=500)
