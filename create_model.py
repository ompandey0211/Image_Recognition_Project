import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_model.pkl")
IMG_SIZE = 64

rng = np.random.RandomState(42)
features = rng.rand(300, IMG_SIZE * IMG_SIZE * 3)
labels = (features[:, 0] > 0.58).astype(int)

model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(features, labels)

joblib.dump(model, MODEL_PATH)
print(f"✓ Model saved successfully to {MODEL_PATH}")
print(f"Model type: {type(model).__name__}")
