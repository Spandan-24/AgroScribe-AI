import os
import traceback
import json
import urllib.parse
import urllib.error
import urllib.request
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# ──── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "agro_cnn.h5")
MODEL_PATH_FIXED = os.path.join(BASE_DIR, "models", "agro_cnn_fixed.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def patch_dtype_policy_and_load(path):
    """
    Workaround for: "'str' object has no attribute 'name'"
    caused by DTypePolicy dict format in newer-Keras-saved models
    being loaded by an older TF/Keras.
    """
    import h5py
    import json
    import shutil
    import tempfile

    with h5py.File(path, "r") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            raise RuntimeError("No model_config found in H5 file.")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        config = json.loads(raw)

    def fix(node):
        if isinstance(node, dict):
            if "dtype" in node and isinstance(node["dtype"], dict):
                inner = node["dtype"].get("config", {})
                node["dtype"] = inner.get("name", "float32")
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for item in node:
                fix(item)
        return node

    fixed_config = fix(config)
    fixed_str = json.dumps(fixed_config)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    shutil.copy(path, tmp_path)

    with h5py.File(tmp_path, "a") as f:
        try:
            f.attrs.modify("model_config", fixed_str)
        except Exception:
            del f.attrs["model_config"]
            f.attrs.create("model_config", fixed_str.encode("utf-8"))

    loaded = load_model(tmp_path, compile=False)
    os.remove(tmp_path)
    return loaded


# ──── Load Model: try multiple strategies ────────────────────────────────
model = None
model_load_error = None
load_strategy_used = None

candidates = [MODEL_PATH_FIXED, MODEL_PATH]
existing_candidates = [p for p in candidates if os.path.exists(p)]

if not existing_candidates:
    model_load_error = (
        f"No model file found. Looked for:\n"
        f"  - {MODEL_PATH_FIXED}\n"
        f"  - {MODEL_PATH}"
    )
    print("WARNING:", model_load_error)
else:
    model_path_to_use = existing_candidates[0]
    print(f"Found model file: {model_path_to_use}")

    # Strategy 1: plain load_model
    try:
        model = load_model(model_path_to_use, compile=False)
        load_strategy_used = "direct load_model()"
    except Exception as e1:
        print("Direct load failed:", e1)

        # Strategy 2: patch DTypePolicy config and retry
        try:
            model = patch_dtype_policy_and_load(model_path_to_use)
            load_strategy_used = "patched DTypePolicy + load_model()"
        except Exception as e2:
            model_load_error = (
                f"Direct load failed: {e1}\n\n"
                f"Patched load also failed: {e2}"
            )
            print("WARNING:", model_load_error)
            traceback.print_exc()

if model is not None:
    print(f"Model loaded successfully via: {load_strategy_used}")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)


# ──── Class Labels ───────────────────────────────────────────────────────
CLASS_LABELS = [
    'Pepper__bell Bacterial_spot',
    'Pepper__bell healthy',
    'Potato Early_blight',
    'Potato Late_blight',
    'Potato healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# ──── Severity & Urgency Map ──────────────────────────────────────────────
DISEASE_META = {
    'Pepper__bell Bacterial_spot':                    ('Medium', 'Medium'),
    'Pepper__bell healthy':                           ('None',   'Low'),
    'Potato Early_blight':                            ('Medium', 'Medium'),
    'Potato Late_blight':                             ('High',   'Critical'),
    'Potato healthy':                                 ('None',   'Low'),
    'Tomato_Bacterial_spot':                          ('Medium', 'High'),
    'Tomato_Early_blight':                            ('Medium', 'Medium'),
    'Tomato_Late_blight':                             ('High',   'Critical'),
    'Tomato_Leaf_Mold':                               ('Low',    'Medium'),
    'Tomato_Septoria_leaf_spot':                      ('Medium', 'Medium'),
    'Tomato_Spider_mites_Two_spotted_spider_mite':    ('Medium', 'High'),
    'Tomato__Target_Spot':                            ('Medium', 'Medium'),
    'Tomato__Tomato_YellowLeaf__Curl_Virus':          ('High',   'High'),
    'Tomato__Tomato_mosaic_virus':                    ('High',   'High'),
    'Tomato_healthy':                                 ('None',   'Low'),
}


# ──── Prediction Helper ────────────────────────────────────────────────────
def predict_image(img_path):
    if model.input_shape and len(model.input_shape) == 4:
        target_h = model.input_shape[1] or 224
        target_w = model.input_shape[2] or 224
    else:
        target_h, target_w = 224, 224

    img = image.load_img(img_path, target_size=(target_h, target_w))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    if class_index >= len(CLASS_LABELS):
        raise ValueError(
            f"Model output has {preds.shape[-1]} classes but CLASS_LABELS "
            f"has only {len(CLASS_LABELS)} entries. They must match exactly "
            f"(same order as used during training)."
        )

    label = CLASS_LABELS[class_index]
    severity, urgency = DISEASE_META.get(label, ('Medium', 'Medium'))
    return label, confidence, severity, urgency


# ──── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/health')
def health():
    """Quick check endpoint to verify model status without uploading a file."""
    return jsonify({
        "model_loaded": model is not None,
        "load_strategy": load_strategy_used,
        "error": model_load_error,
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "input_shape": str(model.input_shape) if model is not None else None,
        "output_shape": str(model.output_shape) if model is not None else None,
        "num_classes_in_model": int(model.output_shape[-1]) if model is not None else None,
        "num_classes_in_labels": len(CLASS_LABELS),
    })


def openweather_request(endpoint, lat, lon):
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return None, ("OPENWEATHERMAP_API_KEY is not configured", 500)

    try:
        lat_value = float(lat)
        lon_value = float(lon)
        if not (-90 <= lat_value <= 90 and -180 <= lon_value <= 180):
            return None, ("Invalid coordinates", 400)
    except (TypeError, ValueError):
        return None, ("Invalid coordinates", 400)

    query = urllib.parse.urlencode({
        "lat": lat_value,
        "lon": lon_value,
        "appid": api_key,
        "units": "metric",
    })
    url = f"https://api.openweathermap.org/data/2.5/{endpoint}?{query}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data, None
    except urllib.error.HTTPError as exc:
        return None, (f"Weather provider returned HTTP {exc.code}", 502)
    except (urllib.error.URLError, TimeoutError):
        return None, ("Could not reach weather provider", 502)
    except Exception:
        traceback.print_exc()
        return None, ("Could not process weather provider response", 502)


@app.route('/api/weather')
def weather():
    data, error = openweather_request("weather", request.args.get("lat"), request.args.get("lon"))
    if error:
        message, status = error
        return jsonify({"error": message}), status
    return jsonify(data)


@app.route('/api/forecast')
def forecast():
    data, error = openweather_request("forecast", request.args.get("lat"), request.args.get("lon"))
    if error:
        message, status = error
        return jsonify({"error": message}), status
    return jsonify(data)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded on server.",
            "details": model_load_error
        }), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        safe_name = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(file_path)

        label, confidence, severity, urgency = predict_image(file_path)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "severity": severity,
            "urgency": urgency,
            "image": "/static/uploads/" + safe_name
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Prediction failed on server.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)