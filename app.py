import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Traffic Sign Classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("traffic_sign_model_fixed.h5")

model = load_model()

IMG_HEIGHT = 224
IMG_WIDTH = 224

# ⚠️ IMPORTANT: use SAME mapping as training
index_to_name = {
    0: "Speed limit (5km/h)",
    1: "Speed limit (15km/h)",
    2: "Speed limit (30km/h)",
    3: "Speed limit (40km/h)",
    4: "Speed limit (50km/h)",
    # 👉 add remaining classes later (works even now)
}

st.title("🚦 Traffic Sign Classification")
st.write("Upload an image of a traffic sign")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.expand_dims(np.array(img), axis=0)
    preds = model.predict(img_array)

    idx = np.argmax(preds)
    confidence = preds[0][idx] * 100

    st.success(f"Prediction: {index_to_name.get(idx, 'Unknown')}")
    st.write(f"Confidence: {confidence:.2f}%")
