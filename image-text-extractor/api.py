# api.py

"""
FastAPI interface for the VisionOCR application.
"""

import base64
import imghdr
import logging
import os
import uuid
from typing import Dict, Optional

from config import DEFAULT_PROVIDER, SUPPORTED_PROVIDERS, SYSTEM_PROMPT, setup_logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from image_processor import ImageProcessor
from ocr_agent import create_ocr_agent
from starlette.requests import Request

# Initialize logger for this module
logger = logging.getLogger(__name__)

app = FastAPI()
image_processor = ImageProcessor()


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
    api_key: Optional[str] = Form(None),
    provider: str = Form(DEFAULT_PROVIDER),
    system_prompt: str = Form(SYSTEM_PROMPT),
) -> JSONResponse:
    """Process OCR request."""
    request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracking

    try:
        logger.info(f"[Request {request_id}] New OCR request received")
        logger.info(f"[Request {request_id}] Provider: {provider}")
        logger.info(
            f"[Request {request_id}] File: {file.filename} ({file.content_type})"
        )

        if provider not in SUPPORTED_PROVIDERS:
            logger.error(f"[Request {request_id}] Unsupported provider: {provider}")
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {provider}"
            )

        if provider == "together" and not api_key:
            logger.error(f"[Request {request_id}] Missing API key for Together AI")
            raise HTTPException(
                status_code=400, detail="API key required for Together AI"
            )

        # Process image using ImageProcessor
        logger.info(f"[Request {request_id}] Processing image...")
        content = await file.read()
        processed_image, mime_type = image_processor.process_image(content)
        base64_image = base64.b64encode(processed_image).decode("utf-8")
        logger.info(f"[Request {request_id}] Image processed successfully")

        # Initialize OCR agent and extract text
        logger.info(
            f"[Request {request_id}] Initializing OCR agent with provider: {provider}"
        )
        ocr_agent = create_ocr_agent(provider=provider, api_key=api_key)

        logger.info(f"[Request {request_id}] Extracting text from image...")
        text = ocr_agent.extract_text(base64_image)
        logger.info(f"[Request {request_id}] Text extraction completed")

        # Create success response
        response = create_response(success=True, data={"text": text})
        logger.info(f"[Request {request_id}] Request completed successfully")
        return JSONResponse(content=response)

    except HTTPException as http_exc:
        logger.error(
            f"[Request {request_id}] HTTP Exception: {http_exc.detail}", exc_info=True
        )
        response = create_response(
            success=False,
            error={"code": http_exc.status_code, "message": http_exc.detail},
        )
        return JSONResponse(content=response, status_code=http_exc.status_code)

    except Exception as e:
        logger.error(
            f"[Request {request_id}] Unexpected error: {str(e)}", exc_info=True
        )
        response = create_response(
            success=False, error={"code": 500, "message": str(e)}
        )
        return JSONResponse(content=response, status_code=500)
