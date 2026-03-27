from flask import Flask, request, jsonify, render_template, g
import pickle
import numpy as np
import logging
import os
import time
import uuid
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load the model, scaler, and metadata
logger.info("Loading model artifact from model.pkl")
with open("model.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    scaler = saved_data["scaler"]
    model_metadata = saved_data.get("metadata", {"version": "unknown"})
logger.info(
    "Model loaded | version=%s hash=%s trained_at=%s train_r2=%s test_r2=%s",
    model_metadata.get("version", "unknown"),
    model_metadata.get("model_hash", "unknown"),
    model_metadata.get("trained_at", "unknown"),
    model_metadata.get("train_r2", "unknown"),
    model_metadata.get("test_r2", "unknown"),
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


def _get_api_key():
    return os.environ.get("API_KEY")


def require_api_key(f):
    """Protect a route with API key auth. Skipped when no API_KEY is configured."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = _get_api_key()
        if api_key is None:
            return f(*args, **kwargs)
        provided = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not provided:
            logger.warning("auth | id=%s error=missing_api_key", g.request_id)
            return jsonify({"error": "Missing API key. Provide X-API-Key header or api_key query param."}), 401
        if provided != api_key:
            logger.warning("auth | id=%s error=invalid_api_key", g.request_id)
            return jsonify({"error": "Invalid API key."}), 403
        return f(*args, **kwargs)
    return decorated


@app.before_request
def before_request():
    g.request_id = uuid.uuid4().hex[:8]
    g.start_time = time.time()
    logger.info(
        "request_start | id=%s method=%s path=%s remote=%s",
        g.request_id,
        request.method,
        request.path,
        request.remote_addr,
    )


@app.after_request
def after_request(response):
    duration_ms = (time.time() - g.start_time) * 1000
    logger.info(
        "request_end | id=%s status=%s duration=%.1fms",
        g.request_id,
        response.status_code,
        duration_ms,
    )
    return response

@app.route("/health", methods=["GET"])
@limiter.exempt
def health_check():
    logger.info("health_check | status=healthy")
    return jsonify({"status": "healthy"})


@app.route("/model/info", methods=["GET"])
@require_api_key
@limiter.limit("30 per minute")
def model_info():
    logger.info("model_info | version=%s", model_metadata.get("version"))
    return jsonify(model_metadata)

@app.route("/", methods=["GET"])
@limiter.exempt
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@require_api_key
@limiter.limit("10 per minute")
def predict():
    data = request.get_json(silent=True)
    if data is None:
        logger.warning("predict | id=%s error=invalid_json", g.request_id)
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    features = data.get("features")
    if not isinstance(features, list):
        logger.warning("predict | id=%s error=features_not_list", g.request_id)
        return jsonify({"error": "'features' must be a list"}), 400

    if len(features) != 4:
        logger.warning("predict | id=%s error=wrong_feature_count count=%d", g.request_id, len(features))
        return jsonify({"error": "'features' must contain exactly 4 values: [sqft, bedrooms, bathrooms, year_built]"}), 400

    for i, val in enumerate(features):
        if not isinstance(val, (int, float)):
            logger.warning("predict | id=%s error=non_numeric_feature index=%d", g.request_id, i)
            return jsonify({"error": f"Feature at index {i} must be a number"}), 400

    try:
        features_arr = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_arr)
        prediction = model.predict(features_scaled).tolist()
        logger.info("predict | id=%s features=%s prediction=%s", g.request_id, features, prediction)
        return jsonify({
            "prediction": prediction,
            "model_version": model_metadata.get("version", "unknown"),
            "model_hash": model_metadata.get("model_hash", "unknown"),
        })
    except Exception as e:
        logger.exception("predict | id=%s error=prediction_failed", g.request_id)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    logger.info("Starting Flask app on 0.0.0.0:5050")
    app.run(host="0.0.0.0", port=5050)
