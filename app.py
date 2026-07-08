import os
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = "image_classifier.keras"

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶")
st.title("🐱🐶 Cat vs Dog Image Classifier")

# Try to load TensorFlow
try:
    import tensorflow as tf
    tf_available = True
except (ImportError, ModuleNotFoundError):
    tf_available = False

# Show status
if not tf_available:
    st.warning("⚠️ TensorFlow is not installed. The app is in demo mode.")
    st.info("To use the classifier, install Python 3.12 and run: `pip install -r requirements.txt`")
else:
    st.success("✓ TensorFlow is available!")

# Demo mode - allow file upload and show interface
st.markdown("---")
st.subheader("Upload an Image")

uploaded_file = st.file_uploader(
    "Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    if tf_available and os.path.exists(MODEL_PATH):
        # Use the actual model if available
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            img_resized = img.resize((150, 150))
            img_array = np.asarray(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            prediction = model.predict(img_array, verbose=0)
            
            if prediction.shape[-1] == 1:
                score = float(prediction[0][0])
                label = "Dog" if score > 0.5 else "Cat"
                confidence = score if label == "Dog" else 1.0 - score
            else:
                probabilities = tf.nn.softmax(prediction[0]).numpy()
                class_idx = int(np.argmax(probabilities))
                label = "Dog" if class_idx == 1 else "Cat"
                confidence = float(probabilities[class_idx])
            
            icon = "🐶" if label == "Dog" else "🐱"
            st.success(f"Prediction: {icon} {label} ({confidence:.0%} confidence)")
        except Exception as e:
            st.error(f"Error loading or using model: {e}")
    else:
        # Demo prediction
        demo_label = np.random.choice(["Cat", "Dog"])
        demo_confidence = np.random.uniform(0.7, 0.99)
        icon = "🐶" if demo_label == "Dog" else "🐱"
        st.info(f"📊 Demo Prediction: {icon} {demo_label} ({demo_confidence:.0%} confidence)")
        st.caption("(This is a demo result - model not available)")