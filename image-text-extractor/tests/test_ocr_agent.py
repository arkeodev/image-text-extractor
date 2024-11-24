# tests/test_ocr_agent.py

"""
Unit tests for ocr_agent.py
"""

from unittest.mock import MagicMock, patch

import pytest
from ocr_agent import TogetherOcrAgent


@pytest.fixture
def agent():
    return TogetherOcrAgent(api_key="test_api_key")


@pytest.fixture
def image_data():
    return {
        "base64_image": "base64_encoded_image_string",
        "mime_type": "image/jpeg",
        "system_prompt": "Test prompt",
    }


@patch("together.Together")
def test_extract_text_success(mock_together, agent, image_data):
    mock_instance = MagicMock()
    mock_instance.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Extracted text"))]
    )
    mock_together.return_value = mock_instance

    text = agent.extract_text(image_data["base64_image"])
    assert text == "Extracted text"


@patch("together.Together", side_effect=Exception("API Error"))
def test_extract_text_failure(mock_together, agent, image_data):
    with pytest.raises(Exception) as exc_info:
        agent.extract_text(image_data["base64_image"])
    assert "API Error" in str(exc_info.value)
