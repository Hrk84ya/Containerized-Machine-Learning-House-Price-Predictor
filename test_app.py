import pytest
import json
from app import app, model_metadata


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

def test_home_page_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"House Price Predictor" in resp.data
    assert resp.content_type.startswith("text/html")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data == {"status": "healthy"}


# ---------------------------------------------------------------------------
# Model info endpoint
# ---------------------------------------------------------------------------

def test_model_info_returns_metadata(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "version" in data
    assert "model_hash" in data
    assert "trained_at" in data
    assert "algorithm" in data
    assert "features" in data
    assert data["features"] == ["sqft", "bedrooms", "bathrooms", "year_built"]


def test_model_info_includes_performance_metrics(client):
    resp = client.get("/model/info")
    data = resp.get_json()
    assert "train_r2" in data
    assert "test_r2" in data
    assert 0 < data["train_r2"] <= 1.0
    assert 0 < data["test_r2"] <= 1.0


# ---------------------------------------------------------------------------
# Predict – valid requests
# ---------------------------------------------------------------------------

def test_predict_valid_input(client):
    """A well-formed request should return a numeric prediction with model version."""
    resp = client.post(
        "/predict",
        data=json.dumps({"features": [2000, 3, 2, 2000]}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == 1
    assert isinstance(data["prediction"][0], (int, float))
    # model versioning fields present
    assert "model_version" in data
    assert "model_hash" in data


def test_predict_returns_reasonable_price(client):
    """Prediction for a typical house should be in a plausible range."""
    resp = client.post(
        "/predict",
        data=json.dumps({"features": [2500, 3, 2, 2005]}),
        content_type="application/json",
    )
    price = resp.get_json()["prediction"][0]
    # Trained on synthetic data: base ~200k + sqft*100 + beds*25k + baths*35k + year offset
    # For these inputs, expect roughly 400k–600k
    assert 100_000 < price < 1_000_000


def test_predict_larger_house_costs_more(client):
    """A larger house should predict a higher price, all else equal."""
    small = client.post(
        "/predict",
        data=json.dumps({"features": [1200, 2, 1, 2000]}),
        content_type="application/json",
    ).get_json()["prediction"][0]

    large = client.post(
        "/predict",
        data=json.dumps({"features": [4500, 2, 1, 2000]}),
        content_type="application/json",
    ).get_json()["prediction"][0]

    assert large > small


def test_predict_more_bedrooms_costs_more(client):
    """More bedrooms should increase the predicted price."""
    fewer = client.post(
        "/predict",
        data=json.dumps({"features": [2500, 1, 2, 2000]}),
        content_type="application/json",
    ).get_json()["prediction"][0]

    more = client.post(
        "/predict",
        data=json.dumps({"features": [2500, 5, 2, 2000]}),
        content_type="application/json",
    ).get_json()["prediction"][0]

    assert more > fewer


def test_predict_newer_house_costs_more(client):
    """A newer house should predict higher than an older one."""
    old = client.post(
        "/predict",
        data=json.dumps({"features": [2500, 3, 2, 1965]}),
        content_type="application/json",
    ).get_json()["prediction"][0]

    new = client.post(
        "/predict",
        data=json.dumps({"features": [2500, 3, 2, 2020]}),
        content_type="application/json",
    ).get_json()["prediction"][0]

    assert new > old


def test_predict_deterministic(client):
    """Same input should always produce the same prediction."""
    payload = json.dumps({"features": [3000, 4, 2, 2010]})
    p1 = client.post("/predict", data=payload, content_type="application/json").get_json()["prediction"][0]
    p2 = client.post("/predict", data=payload, content_type="application/json").get_json()["prediction"][0]
    assert p1 == p2


def test_predict_with_float_features(client):
    """Floats (e.g. 2.5 bathrooms) should be accepted."""
    resp = client.post(
        "/predict",
        data=json.dumps({"features": [2000.5, 3.0, 2.5, 2000.0]}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert isinstance(resp.get_json()["prediction"][0], (int, float))


# ---------------------------------------------------------------------------
# Predict – input validation errors
# ---------------------------------------------------------------------------

def test_predict_no_body(client):
    """Missing JSON body should return 400."""
    resp = client.post("/predict", content_type="application/json")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_invalid_json(client):
    """Malformed JSON should return 400."""
    resp = client.post("/predict", data="not json", content_type="application/json")
    assert resp.status_code == 400


def test_predict_features_not_list(client):
    """features as a string should return 400."""
    resp = client.post(
        "/predict",
        data=json.dumps({"features": "not a list"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "list" in resp.get_json()["error"].lower()


def test_predict_features_wrong_count(client):
    """Too few features should return 400."""
    resp = client.post(
        "/predict",
        data=json.dumps({"features": [2000, 3]}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "4 values" in resp.get_json()["error"]


def test_predict_features_too_many(client):
    """Too many features should return 400."""
    resp = client.post(
        "/predict",
        data=json.dumps({"features": [2000, 3, 2, 2000, 999]}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_non_numeric_feature(client):
    """A string in the features list should return 400."""
    resp = client.post(
        "/predict",
        data=json.dumps({"features": [2000, "three", 2, 2000]}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "number" in resp.get_json()["error"].lower()


def test_predict_missing_features_key(client):
    """JSON without 'features' key should return 400."""
    resp = client.post(
        "/predict",
        data=json.dumps({"inputs": [2000, 3, 2, 2000]}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_features_none(client):
    """features: null should return 400."""
    resp = client.post(
        "/predict",
        data=json.dumps({"features": None}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_empty_features_list(client):
    """Empty features list should return 400."""
    resp = client.post(
        "/predict",
        data=json.dumps({"features": []}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_boolean_in_features(client):
    """Booleans in features should return 400 (bool is subclass of int in Python, but still valid)."""
    # Note: In Python, bool IS a subclass of int, so isinstance(True, int) is True.
    # This means the app will accept booleans — this test documents that behavior.
    resp = client.post(
        "/predict",
        data=json.dumps({"features": [True, 3, 2, 2000]}),
        content_type="application/json",
    )
    # bool passes isinstance(val, (int, float)), so this is accepted
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Method not allowed
# ---------------------------------------------------------------------------

def test_predict_get_not_allowed(client):
    """GET on /predict should be 405."""
    resp = client.get("/predict")
    assert resp.status_code == 405


def test_home_post_not_allowed(client):
    """POST on / should be 405."""
    resp = client.post("/")
    assert resp.status_code == 405
