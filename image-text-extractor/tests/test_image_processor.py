# tests/test_image_processor.py

"""
Unit tests for image_processor.py
"""

import os
from unittest.mock import patch, mock_open
import pytest
from image_processor import ImageProcessor

@pytest.fixture
def processor():
    return ImageProcessor()

@pytest.fixture
def image_paths():
    return {
        'valid': 'tests/images/valid_image.jpg',
        'invalid': 'tests/images/invalid_image.xyz',
        'nonexistent': 'tests/images/nonexistent.jpg'
    }

@patch('os.path.exists')
def test_validate_image_exists_and_supported(mock_exists, processor, image_paths):
    mock_exists.return_value = True
    with patch('os.path.splitext', return_value=('.jpg', '.jpg')):
        result = processor.validate_image(image_paths['valid'])
        assert result is True

@patch('os.path.exists')
def test_validate_image_not_exists(mock_exists, processor, image_paths):
    mock_exists.return_value = False
    result = processor.validate_image(image_paths['nonexistent'])
    assert result is False

@patch('os.path.exists')
def test_validate_image_unsupported_type(mock_exists, processor, image_paths):
    mock_exists.return_value = True
    with patch('os.path.splitext', return_value=('.xyz', '.xyz')):
        result = processor.validate_image(image_paths['invalid'])
        assert result is False

@patch('builtins.open', new_callable=mock_open, read_data=b'image data')
def test_encode_image_success(mock_file, processor, image_paths):
    with patch('os.path.exists', return_value=True):
        encoded_image = processor.encode_image(image_paths['valid'])
        assert isinstance(encoded_image, str)

@patch('builtins.open', side_effect=Exception('File open error'))
def test_encode_image_failure(mock_file, processor, image_paths):
    with patch('os.path.exists', return_value=True):
        with pytest.raises(Exception):
            processor.encode_image(image_paths['valid'])
