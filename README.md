# 🏠 Containerized Machine Learning: House Price Predictor

A containerized ML service that predicts house prices using a trained Random Forest model. Built with Flask, packaged with Docker, and wired for CI/CD via GitHub Actions and Jenkins.

## 📦 Project Overview

- **Model Training** — synthetic data generation, feature scaling, and a `RandomForestRegressor` with full metadata/versioning baked into the artifact.
- **Flask API** — serves predictions, model info, and a browser-based UI.
- **Rate Limiting & Auth** — per-endpoint rate limits via Flask-Limiter; optional API key authentication.
- **Structured Logging** — every request is logged with a unique ID, method, path, status, and duration.
- **Graceful Degradation** — the app starts even if the model file is missing or corrupt; health check reports `degraded` status.
- **Containerization** — single Dockerfile for consistent deployment.
- **CI/CD** — GitHub Actions workflow and Jenkinsfile for automated test → build → deploy.

## 🚀 Getting Started

### Prerequisites

- [Python 3.9+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/products/docker-desktop/) (for containerized deployment)
- [Git](https://git-scm.com/downloads)

### Local Setup

```bash
git clone https://github.com/Hrk84ya/Containerized-ML.git
cd Containerized-ML
pip install -r requirements.txt
python train_model.py
python app.py
```

The app will be available at http://localhost:5050.

### Docker

```bash
docker build -t house-price-predictor .
docker run -d -p 5050:5050 house-price-predictor
```

To enable API key authentication:

```bash
docker run -d -p 5050:5050 -e API_KEY=your-secret-key house-price-predictor
```

## 🧠 Model Training

`train_model.py` generates synthetic housing data, trains a `RandomForestRegressor`, and saves a versioned artifact to `model.pkl` containing:

- The trained model and scaler
- Metadata: version, SHA-256 hash, training timestamp, hyperparameters, sklearn/numpy versions, feature names, dataset size, and train/test R² scores

```bash
python train_model.py
```

Sample output:

```
Model training completed and saved as model.pkl
  Version:      1.0.0
  Hash:         3e68dfc69363
  Train R²:     0.9927
  Test R²:      0.9363
```

## 🖥️ API Endpoints

| Endpoint | Method | Auth | Rate Limit | Description |
|---|---|---|---|---|
| `/` | GET | No | Exempt | Browser UI for predictions |
| `/health` | GET | No | Exempt | Health check (returns `degraded` if model failed to load) |
| `/model/info` | GET | Yes | 30/min | Model metadata and performance metrics |
| `/predict` | POST | Yes | 10/min | House price prediction |

### `POST /predict`

**Request:**

```json
{
  "features": [2500, 3, 2, 2005]
}
```

Features are: `[sqft, bedrooms, bathrooms, year_built]`

**Response (200):**

```json
{
  "prediction": [497500.0],
  "model_version": "1.0.0",
  "model_hash": "3e68dfc69363"
}
```

**Error responses:** `400` (validation), `401` (missing API key), `403` (invalid key), `429` (rate limited), `503` (model not loaded).

### Authentication

Set the `API_KEY` environment variable to enable auth. Clients pass the key via:

- `X-API-Key` header, or
- `api_key` query parameter

When `API_KEY` is not set, authentication is disabled (open access).

### Rate Limiting

Global defaults: 200 requests/day, 50/hour per IP. Endpoint-specific limits are listed in the table above. `/` and `/health` are exempt.

## 🧪 Testing

```bash
python -m pytest test_app.py -v
```

The test suite (41 tests) covers:

- All endpoints and response shapes
- Prediction logic: price ranges, monotonic feature relationships, determinism
- Input validation: every error branch (missing body, wrong types, wrong count, etc.)
- API key auth: missing key, wrong key, valid header, valid query param, bypass when unset
- Rate limiting: enforcement on `/predict`, exemption on `/health`
- Model loading failures: missing file, corrupt pickle, missing keys, wrong type, recovery

## 🛠️ Project Structure

```
├── app.py                 # Flask application
├── train_model.py         # Model training script
├── test_app.py            # Test suite (41 tests)
├── model.pkl              # Versioned model artifact
├── templates/
│   └── index.html         # Browser UI template
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build instructions
├── Jenkinsfile            # Jenkins CI/CD pipeline
└── .github/workflows/     # GitHub Actions CI/CD
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | *(unset — auth disabled)* | API key for `/predict` and `/model/info` |
| `MODEL_PATH` | `model.pkl` | Path to the model artifact |

## 🔄 CI/CD

### GitHub Actions

The workflow in `.github/workflows/ci-cd.yml` handles linting, testing, Docker image build, and deployment on every push.

### Jenkins

The `Jenkinsfile` defines a pipeline with stages: Checkout → Setup → Test → Build → Deploy → Post Actions. Requires Docker and Pipeline plugins.

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙌 Acknowledgements

- [Scikit-learn](https://scikit-learn.org/) — ML algorithms
- [Flask](https://flask.palletsprojects.com/) — web framework
- [Flask-Limiter](https://flask-limiter.readthedocs.io/) — rate limiting
- [Docker](https://www.docker.com/) — containerization
- [Jenkins](https://www.jenkins.io/) / [GitHub Actions](https://github.com/features/actions) — CI/CD
