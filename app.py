# app.py
import io
import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib

# TensorFlow preprocessing and model building
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import Model

# ----- CONFIG: update these paths BEFORE running -----
SAVE_DIR = r"Y:\SEMPRNEW\models"   # folder where svm_model.joblib, scaler.joblib, label_encoder.joblib, pca.joblib are stored
IMAGE_SIZE = (224, 224)
TOP_K = 3              # return top-K probabilities
# ----------------------------------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---- load artifacts ----
svm_path = os.path.join(SAVE_DIR, "svm_model.joblib")
scaler_path = os.path.join(SAVE_DIR, "scaler.joblib")
le_path = os.path.join(SAVE_DIR, "label_encoder.joblib")
pca_path = os.path.join(SAVE_DIR, "pca.joblib")  # optional

if not os.path.exists(svm_path):
    raise FileNotFoundError(f"SVM model not found at {svm_path}. Place svm_model.joblib there.")

svm = joblib.load(svm_path)
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
le = joblib.load(le_path)
pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
class_names = list(le.classes_)

# Build EfficientNetB0 feature extractor (same as training)
base = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg",
                      input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base.trainable = False
feature_model = Model(inputs=base.input, outputs=base.output)

def prepare_image(image_bytes):
    """Return a preprocessed numpy array shape (1, H, W, C)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)   # EfficientNet preprocessing
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", classes=class_names)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file part, provide file field named 'file'"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no file selected"}), 400

    try:
        image_bytes = file.read()
        img_arr = prepare_image(image_bytes)

        # extract features
        feats = feature_model.predict(img_arr)  # shape (1, feat_dim)
        # scale
        if scaler is not None:
            feats_scaled = scaler.transform(feats)
        else:
            feats_scaled = feats
        # pca
        if pca is not None:
            feats_final = pca.transform(feats_scaled)
        else:
            feats_final = feats_scaled

        # predict
        probs = svm.predict_proba(feats_final)[0]
        top_idxs = np.argsort(probs)[::-1][:TOP_K]
        top = [{"class": class_names[int(i)], "probability": float(probs[int(i)])} for i in top_idxs]

        best_idx = int(top_idxs[0])
        result = {
            "predicted_class": class_names[best_idx],
            "confidence": float(probs[best_idx]),
            "top_k": top,
            "all_probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: a route to download model artifacts (for debugging)
@app.route("/models/<path:filename>")
def models(filename):
    return send_from_directory(SAVE_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    # In production use a WSGI server (gunicorn) instead of Flask dev server.
    app.run(host="0.0.0.0", port=5000, debug=True)
