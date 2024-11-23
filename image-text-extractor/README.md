# ImageTextExtractor

An OCR application that extracts text from images using Langchain and Streamlit.

## Features

- Extract text from uploaded images
- Process multiple image formats (PNG, JPG, JPEG, GIF, WEBP)
- User-friendly Streamlit interface
- RESTful API endpoints
- Integration with Langchain for advanced text processing
- Together AI Vision model integration

## Prerequisites

- Python 3.12 or higher
- Poetry package manager
- Together AI API key

## Installation

1.Clone the repository:

```bash
   git clone https://github.com/yourusername/ImageTextExtractor.git
   cd ImageTextExtractor   
```

2.Install dependencies using Poetry:

```bash
   poetry install   
   ```

## Usage

### Streamlit UI

1.Start the FastAPI backend:

```bash
   poetry run python main.py
```

2.In a new terminal, launch the Streamlit interface:

```bash
   poetry run streamlit run ui.py
```

3.Open your browser and navigate to `http://localhost:8501`

4.Enter your Together AI API key

5.Upload an image and wait for the results

### REST API

The application exposes a REST API endpoint for OCR processing.

#### Endpoint: POST /ocr

**Request:**

- URL: `http://localhost:8000/ocr`
- Method: `POST`
- Content-Type: `multipart/form-data`

**Parameters:**

- `file`: Image file (supported formats: PNG, JPG, JPEG, GIF, WEBP)
- `api_key`: Together AI API key
- `system_prompt`: (Optional) Custom prompt for the vision model

**Example using curl:**

```bash
curl -X POST http://localhost:8000/ocr \
-F "file=@/path/to/your/image.jpg" \
-F "api_key=your_together_ai_api_key" \
-F "system_prompt=Convert the provided image into text"
```

**Response:**

```bash
poetry run pytest
```

### Environment Variables

The application uses the following configurations (defined in `config.py`):

- `LOGGING_LEVEL`: Default is "INFO"
- `SUPPORTED_IMAGE_TYPES`: [".png", ".jpg", ".jpeg", ".gif", ".webp"]
- `TOGETHER_MODEL_NAME`: "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Together AI](https://together.ai/) for providing the vision model
- [Langchain](https://python.langchain.com/) for the AI integration framework
- [Streamlit](https://streamlit.io/) for the user interface
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API implementation