import os
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="wide"
)

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(rgba(25, 30, 45, 0.70), rgba(15, 23, 42, 0.78)),
                        url("https://images.unsplash.com/photo-1517849845537-4d257902454a?auto=format&fit=crop&w=1600&q=80") center / cover no-repeat fixed;
            color: #f8fafc;
        }

        [data-testid="stHeader"] {
            background: rgba(15, 23, 42, 0.20);
            backdrop-filter: blur(8px);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .glass-card {
            background: rgba(15, 23, 42, 0.62);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 22px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.30);
            padding: 1.5rem 1.4rem;
            backdrop-filter: blur(12px);
        }

        .hero {
            padding: 1.8rem 1.5rem 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
            border-radius: 24px;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        .hero h1 {
            color: #f8fafc;
            font-size: 2.5rem;
            margin-bottom: 0.35rem;
        }

        .hero p {
            color: #dbeafe;
            font-size: 1.05rem;
            margin: 0;
        }

        .stFileUploader > div {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 18px;
            padding: 1rem;
        }

        .result-box {
            padding: 1.2rem 1.1rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.22);
            margin-top: 1rem;
        }

        .stProgress > div > div {
            background: linear-gradient(90deg, #60a5fa, #34d399);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load Model
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_model.pkl")
IMG_SIZE = 64


def create_demo_model(path):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(42)
    features = rng.rand(250, IMG_SIZE * IMG_SIZE * 3)
    labels = (features[:, 0] > 0.58).astype(int)

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(features, labels)
    joblib.dump(model, path)
    return model


if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.warning("Model file not found. A demo model is being created automatically.")
    model = create_demo_model(MODEL_PATH)

st.markdown(
    """
    <div class="hero">
        <h1>🐾 Cat vs Dog Classifier</h1>
        <p>Upload a pet photo and let the system predict whether it is a cat or a dog.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.1, 1.3])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear image of a cat or dog for the best prediction."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top: 0; color: #e2e8f0;">AI Insight</h3>
            <p style="color: #dbeafe; margin-bottom: 0;">This app evaluates the uploaded image using a trained classification model and displays the prediction confidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    pixels = np.array(resized, dtype=np.float32).flatten()

    prediction = int(model.predict([pixels])[0])
    probability = model.predict_proba([pixels])[0]

    class_names = ["Cat", "Dog"]
    confidence = probability[prediction] * 100
    display_label = class_names[prediction]

    st.markdown(
        f"""
        <div class="result-box">
            <h2 style="margin: 0 0 0.5rem; color: #f8fafc;">{display_label} detected</h2>
            <p style="margin: 0; color: #dbeafe;">Prediction confidence: <strong>{confidence:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(image, caption="Uploaded image", width=420)

    st.subheader("Prediction Details")
    cat_prob = probability[0] * 100
    dog_prob = probability[1] * 100
    st.write(f"🐱 Cat probability: **{cat_prob:.2f}%**")
    st.write(f"🐶 Dog probability: **{dog_prob:.2f}%**")

    st.progress(int(confidence))

    if prediction == 0:
        st.success("✅ Prediction: Cat")
    else:
        st.success("✅ Prediction: Dog")
else:
    st.info("Upload an image to begin the analysis.")
