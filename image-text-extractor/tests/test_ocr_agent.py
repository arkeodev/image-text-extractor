# tests/test_ocr_agent.py

"""
Unit tests for ocr_agent.py
"""

from unittest.mock import MagicMock, patch

import pytest
from ocr_agent import OcrAgent


@pytest.fixture
def agent():
    return OcrAgent(api_key="test_api_key")


@pytest.fixture
def image_data():
    return {
        "base64_image": "base64_encoded_image_string",
        "mime_type": "image/jpeg",
        "system_prompt": "Test prompt",
    }


@patch("langchain.llms.TogetherAI")
def test_analyze_image_success(mock_togetherai, agent, image_data):
    mock_llm_instance = MagicMock()
    mock_llm_instance.run.return_value = "Extracted text"
    mock_togetherai.return_value = mock_llm_instance

    text = agent.analyze_image(
        image_data["base64_image"], image_data["mime_type"], image_data["system_prompt"]
    )
    assert text == "Extracted text"


@patch("langchain.llms.TogetherAI", side_effect=Exception("API Error"))
def test_analyze_image_failure(mock_togetherai, agent, image_data):
    with pytest.raises(Exception) as exc_info:
        agent.analyze_image(
            image_data["base64_image"],
            image_data["mime_type"],
            image_data["system_prompt"],
        )
    assert "API Error" in str(exc_info.value)
