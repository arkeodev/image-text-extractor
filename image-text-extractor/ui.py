# ui.py

"""
Streamlit UI for the VisionOCR application.
"""

import logging

import requests
import streamlit as st
from config import setup_logging

# Initialize logger for this module
logger = logging.getLogger(__name__)


def handle_api_error(response):
    """Handle API error responses."""
    try:
        error_data = response.json()
        error_message = error_data.get("error", {}).get(
            "message", "Unknown error occurred."
        )
        error_code = error_data.get("error", {}).get("code", 500)
        logger.error(f"API Error (Code {error_code}): {error_message}")
        return error_message
    except Exception as e:
        logger.error(f"Error parsing API response: {str(e)}")
        return "Failed to parse error response from server"


def main():
    """Main function to run the Streamlit app."""
    try:
        setup_logging()  # Initialize logging

        st.title("VisionOCR Application")
        st.write("Upload an image to perform OCR using Together AI.")

        api_key = st.text_input("Enter your Together AI API Key:", type="password")
        system_prompt = st.text_area(
            "System Prompt:",
            value="""Extract meaningful text content from the image while following these rules:

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

Output the text in a clean, human-readable format.""",
            height=300,  # Increased height to better display the longer prompt
        )

        uploaded_file = st.file_uploader(
            "Choose an image...", type=["png", "jpg", "jpeg", "gif", "webp"]
        )

        if uploaded_file is not None and api_key:
            try:
                logger.info(f"Processing image: {uploaded_file.name}")

                # Display the uploaded image
                st.image(
                    uploaded_file, caption="Uploaded Image.", use_container_width=True
                )
                st.write("Processing...")

                # Prepare the files and data for the request
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.read(),
                        uploaded_file.type,
                    )
                }
                data = {"api_key": api_key, "system_prompt": system_prompt}

                # Send request to the FastAPI endpoint
                try:
                    response = requests.post(
                        "http://localhost:8008/ocr", files=files, data=data
                    )
                    response_json = response.json()

                    if response.status_code == 200 and response_json.get("success"):
                        text = response_json["data"]["text"]
                        logger.info("Successfully processed image")
                        st.write("Extracted Text:")
                        st.text_area("OCR Output", text, height=200)
                    else:
                        error_message = handle_api_error(response)
                        st.error(f"Error: {error_message}")

                except requests.exceptions.ConnectionError:
                    logger.error("Failed to connect to the API server")
                    st.error(
                        "Could not connect to the server. Please ensure the API is running."
                    )
                except requests.exceptions.Timeout:
                    logger.error("Request timed out")
                    st.error("Request timed out. Please try again.")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request failed: {str(e)}")
                    st.error("Failed to process the request. Please try again.")

            except Exception as e:
                logger.error(f"Error processing image: {str(e)}", exc_info=True)
                st.error(f"An error occurred while processing the image: {str(e)}")

        elif not api_key:
            logger.warning("No API key provided")
            st.warning("Please enter your Together AI API Key.")

    except Exception as e:
        logger.critical(f"Application error: {str(e)}", exc_info=True)
        st.error("A critical error occurred. Please check the application logs.")


if __name__ == "__main__":
    main()
