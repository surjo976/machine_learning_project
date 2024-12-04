# Image Captioning and Emotion Detection API

This project is a Flask-based web application that provides an API for image captioning, sentiment analysis, and emotion detection from images. It utilizes pre-trained models from the Hugging Face Transformers library and the FER library for emotion detection.

## Features

- **Image Captioning**: Generates a descriptive caption for an uploaded image using a pre-trained BLIP model.
- **Sentiment Analysis**: Analyzes the sentiment of the generated caption.
- **Emotion Detection**: Detects emotions from faces in the uploaded image using the FER library.

## Requirements

- Python 3.7+
- Flask
- transformers
- Pillow
- fer
- numpy
- torch

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Flask application**:
   ```bash
   python main.py
   ```

2. **Upload an image**:
   - Send a POST request to the `/upload` endpoint with an image file.
   - Example using `curl`:
     ```bash
     curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5000/upload
     ```

3. **Receive the response**:
   - The API will return a JSON response containing the image caption, sentiment analysis of the caption, and detected emotions from the image.

## API Endpoints

- **POST /upload**: Upload an image to receive a caption, sentiment analysis, and emotion detection.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FER: Facial Emotion Recognition](https://github.com/justinshenk/fer)