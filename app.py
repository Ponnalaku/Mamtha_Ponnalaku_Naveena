from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown
import os
import tensorflow as tf

# Reduce TensorFlow memory usage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

MODEL_PATH = "cnn_model.keras"
FILE_ID = "1JW0Kr9tsI9yLgsmQzc4sAirvRskX7Fu5"


def download_model(file_id, output):
    if not os.path.exists(output):
        print("Downloading model...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)


# Download model if not present
download_model(FILE_ID, MODEL_PATH)

# Load model safely
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None


classes = ["akiec", "bcc", "melanoma"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")  # Important fix
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(img):
    pred = model.predict(img)
    idx = np.argmax(pred)
    conf = np.max(pred)
    return classes[idx], float(conf)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            img = preprocess(path)
            prediction, confidence = predict_image(img)
            image_path = path

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
