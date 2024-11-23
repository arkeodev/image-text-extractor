# ui.py

"""
Streamlit UI for the VisionOCR application.
"""

import logging
import streamlit as st
from image_processor import ImageProcessor
from ocr_agent import OcrAgent

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("VisionOCR Application")
    st.write("Upload an image to perform OCR using Together AI.")

    api_key = st.text_input("Enter your Together AI API Key:", type="password")
    system_prompt = st.text_area(
        "System Prompt:",
        value="""Convert the provided image into structured text. Ensure that all content from the page is included, including:

- Headers and titles
- Main body text
- Tables and structured data
- Lists and bullet points
- Footnotes and citations
- Image captions and descriptions
- Any other visible text elements

Maintain the original formatting structure while ensuring clear and readable output.""",
        height=100
    )

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg", "gif", "webp"]
    )

    if uploaded_file is not None and api_key:
        try:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write("Processing...")

            # Prepare the files and data for the request
            files = {'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
            data = {'api_key': api_key, 'system_prompt': system_prompt}

            # Send request to the FastAPI endpoint
            response = requests.post("http://localhost:8000/ocr", files=files, data=data)
            response_json = response.json()

            if response.status_code == 200 and response_json.get('success'):
                text = response_json['data']['text']
                st.write("Extracted Text:")
                st.text_area("OCR Output", text, height=200)
            else:
                error_message = response_json.get('error', {}).get('message', 'Unknown error occurred.')
                st.error(f"Error: {error_message}")

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
    elif not api_key:
        st.warning("Please enter your Together AI API Key.")

if __name__ == "__main__":
    main()
