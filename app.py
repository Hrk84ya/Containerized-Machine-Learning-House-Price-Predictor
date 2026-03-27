from flask import Flask, request, jsonify, render_template_string, g
import pickle
import numpy as np
import logging
import time
import uuid

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
def health_check():
    logger.info("health_check | status=healthy")
    return jsonify({"status": "healthy"})


@app.route("/model/info", methods=["GET"])
def model_info():
    logger.info("model_info | version=%s", model_metadata.get("version"))
    return jsonify(model_metadata)

@app.route("/", methods=["GET"])
def home():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>House Price Predictor | ML Service</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(to right, #f8fafc, #e2e8f0);
                color: #1e293b;
                margin: 0;
                padding: 2rem;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }

            .container {
                background: white;
                border: 1px solid #e2e8f0;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
                border-radius: 12px;
                padding: 2.5rem;
                max-width: 720px;
                width: 100%;
            }

            .header h1 {
                font-size: 2.25rem;
                margin-bottom: 0.5rem;
                color: #0f172a;
            }

            .header p {
                color: #64748b;
                font-size: 1.1rem;
                margin-bottom: 2rem;
            }

            .input-group label {
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #334155;
            }

            .input-group input {
                padding: 0.75rem 1rem;
                border-radius: 8px;
                border: 1px solid #cbd5e1;
                font-size: 1rem;
                background-color: #f1f5f9;
                transition: border-color 0.2s, background-color 0.2s;
            }

            .input-group input:focus {
                border-color: #3b82f6;
                background-color: white;
                outline: none;
            }

            button {
                background-color: #3b82f6;
                color: white;
                padding: 0.75rem 2rem;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: background-color 0.3s;
            }

            button:hover {
                background-color: #2563eb;
            }

            #result.success {
                background-color: #d1fae5;
                color: #065f46;
                border: 1px solid #10b981;
            }

            #result.error {
                background-color: #fee2e2;
                color: #991b1b;
                border: 1px solid #f87171;
            }

            /* Retain other styles unchanged */
            :root {
                --primary-color: #2563eb;
                --primary-hover: #1d4ed8;
                --success-color: #059669;
                --error-color: #dc2626;
                --background-color: #f8fafc;
                --card-background: #ffffff;
                --text-primary: #1e293b;
                --text-secondary: #64748b;
                --border-color: #e2e8f0;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            /* Remove previous body and container styles replaced above */

            .form-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }

            .input-group {
                display: flex;
                flex-direction: column;
            }

            /* Remove previous input-group label and input styles replaced above */

            /* Remove previous button styles replaced above */

            .button-container {
                text-align: center;
            }

            .spinner {
                width: 1.25rem;
                height: 1.25rem;
                border: 2px solid #ffffff;
                border-top: 2px solid transparent;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                display: none;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            #result {
                margin-top: 2rem;
                padding: 1rem;
                border-radius: 0.5rem;
                text-align: center;
                font-weight: 500;
                display: none;
            }

            /* Remove previous success and error styles replaced above */

            @media (max-width: 640px) {
                .container {
                    padding: 1.5rem;
                }

                .header h1 {
                    font-size: 1.5rem;
                }

                .form-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>House Rent Predictor</h1>
                <p>Enter the house details below to get an accurate price prediction</p>
            </div>
            <div class="form-grid">
                <div class="input-group">
                    <label for="sqft">Square Footage</label>
                    <input type="number" id="sqft" placeholder="e.g., 2000" min="1000" max="5000">
                </div>
                <div class="input-group">
                    <label for="bedrooms">Number of Bedrooms</label>
                    <input type="number" id="bedrooms" placeholder="e.g., 3" min="1" max="5">
                </div>
                <div class="input-group">
                    <label for="bathrooms">Number of Bathrooms</label>
                    <input type="number" id="bathrooms" placeholder="e.g., 2" min="1" max="3" step="0.5">
                </div>
                <div class="input-group">
                    <label for="year">Year Built</label>
                    <input type="number" id="year" placeholder="e.g., 2000" min="1960" max="2023">
                </div>
            </div>
            <div class="button-container">
                <button onclick="makePrediction()" id="predictButton">
                    <span class="spinner" id="spinner"></span>
                    <span id="buttonText">Predict Price</span>
                </button>
            </div>
            <div id="result"></div>
        </div>
        <script>
            const predictButton = document.getElementById('predictButton');
            const spinner = document.getElementById('spinner');
            const buttonText = document.getElementById('buttonText');
            const resultDiv = document.getElementById('result');

            async function makePrediction() {
                const sqft = parseFloat(document.getElementById('sqft').value);
                const bedrooms = parseFloat(document.getElementById('bedrooms').value);
                const bathrooms = parseFloat(document.getElementById('bathrooms').value);
                const year = parseFloat(document.getElementById('year').value);
                
                if (!sqft || !bedrooms || !bathrooms || !year) {
                    showResult('Please fill in all fields with valid numbers.', 'error');
                    return;
                }

                // Show loading state
                predictButton.disabled = true;
                spinner.style.display = 'inline-block';
                buttonText.textContent = 'Predicting...';
                resultDiv.style.display = 'none';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            features: [sqft, bedrooms, bathrooms, year]
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Prediction failed');
                    }

                    const data = await response.json();
                    const price = new Intl.NumberFormat('en-US', {
                        style: 'currency',
                        currency: 'USD',
                        maximumFractionDigits: 0
                    }).format(data.prediction[0]);
                    
                    showResult(`Predicted House Price: ${price}`, 'success');
                } catch (error) {
                    showResult('An error occurred while making the prediction. Please try again.', 'error');
                } finally {
                    // Reset button state
                    predictButton.disabled = false;
                    spinner.style.display = 'none';
                    buttonText.textContent = 'Predict Price';
                }
            }

            function showResult(message, type) {
                resultDiv.textContent = message;
                resultDiv.className = type;
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/predict", methods=["POST"])
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
