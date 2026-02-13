from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown
import os

app = Flask(__name__)


def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)


# Download models if not present
download_model("1JW0Kr9tsI9yLgsmQzc4sAirvRskX7Fu5", "cnn_model.keras")



# Load models
cnn = load_model("cnn_model.keras")

classes = ["akiec", "bcc", "melanoma"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def preprocess(img_path):
    img = Image.open(img_path).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img


def ensemble_predict(img):
    pred = cnn.predict(img)


    idx = np.argmax(pred)
    conf = np.max(pred)

    return classes[idx], conf


@app.route("/", methods=["GET","POST"])
def home():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            img = preprocess(path)
            prediction, confidence = ensemble_predict(img)

            image_path = path

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

