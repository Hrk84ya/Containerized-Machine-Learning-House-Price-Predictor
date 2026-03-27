import pickle
import hashlib
import datetime
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic house data
np.random.seed(42)
n_samples = 1000

# Features: [square footage, number of bedrooms, number of bathrooms, year built]
X = np.array([
    np.random.uniform(1000, 5000, n_samples),  
    np.random.randint(1, 6, n_samples),        
    np.random.randint(1, 4, n_samples),        
    np.random.randint(1960, 2023, n_samples)
])

# Generate target prices with some noise
base_price = 200000 
price_per_sqft = 100  
price_per_bedroom = 25000  
price_per_bathroom = 35000  
price_per_year = 1000  

y = (base_price + 
     X[0] * price_per_sqft + 
     X[1] * price_per_bedroom + 
     X[2] * price_per_bathroom + 
     (X[3] - 1960) * price_per_year +
     np.random.normal(0, 25000, n_samples))

# Reshape X for sklearn
X = X.T

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model performance
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

# Build versioned artifact with metadata
model_data = pickle.dumps({"model": model, "scaler": scaler})
model_hash = hashlib.sha256(model_data).hexdigest()[:12]

artifact = {
    "model": model,
    "scaler": scaler,
    "metadata": {
        "version": "1.0.0",
        "model_hash": model_hash,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "algorithm": "RandomForestRegressor",
        "n_estimators": 100,
        "random_state": 42,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "features": ["sqft", "bedrooms", "bathrooms", "year_built"],
        "n_samples": n_samples,
        "test_size": 0.2,
        "train_r2": round(train_score, 4),
        "test_r2": round(test_score, 4),
    },
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print(f"Model training completed and saved as model.pkl")
print(f"  Version:      {artifact['metadata']['version']}")
print(f"  Hash:         {model_hash}")
print(f"  Trained at:   {artifact['metadata']['trained_at']}")
print(f"  Train R²:     {train_score:.4f}")
print(f"  Test R²:      {test_score:.4f}")
print(f"  sklearn:      {sklearn.__version__}")
print(f"  Features:     {artifact['metadata']['features']}")
