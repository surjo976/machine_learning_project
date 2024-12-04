from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from fer import FER
import numpy as np
import torch

app = Flask(__name__)

# Load the pre-trained model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    img = Image.open(image)

    # Preprocess the image and generate a caption
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Perform sentiment analysis on the caption
    sentiment = sentiment_pipeline(caption)

    # Convert the image to a NumPy array for emotion detection
    img_array = np.array(img)

    # Perform emotion detection on the image
    emotion_detector = FER()
    emotions = emotion_detector.detect_emotions(img_array)
    if emotions:
        # Get the dominant emotion
        dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        emotion_score = emotions[0]['emotions'][dominant_emotion]
    else:
        dominant_emotion = "No face detected"
        emotion_score = 0

    return jsonify({
        "caption": caption,
        "sentiment": sentiment,
        "image_emotion": {
            "emotion": dominant_emotion,
            "score": emotion_score
        }
    })

if __name__ == '__main__':
    app.run(debug=True)