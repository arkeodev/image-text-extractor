# tests/test_api.py

"""
Unit tests for api.py
"""

from unittest.mock import patch

import pytest
from api import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(app)


@patch("ocr_agent.OcrAgent.analyze_image")
def test_perform_ocr_success(mock_analyze_image, client):
    mock_analyze_image.return_value = "Extracted text"

    response = client.post(
        "/ocr",
        data={"api_key": "test_api_key"},
        files={"file": ("test.jpg", b"dummy image data", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {"text": "Extracted text"},
        "error": None,
    }


def test_perform_ocr_unsupported_file_type(client):
    response = client.post(
        "/ocr",
        data={"api_key": "test_api_key"},
        files={"file": ("test.txt", b"dummy content", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "data": None,
        "error": {"code": 400, "message": "Unsupported file type."},
    }
