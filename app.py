import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Traffic Sign Classifier")

# ================= LOAD TFLITE MODEL =================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="traffic_sign_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Label mapping (partial is fine)
index_to_name = {
    0: "Speed limit (5km/h)",
    1: "Speed limit (15km/h)",
    2: "Speed limit (30km/h)",
    3: "Speed limit (40km/h)",
    4: "Speed limit (50km/h)",
}

# ================= UI =================
st.title("🚦 Traffic Sign Classification")
st.write("Upload an image of a traffic sign")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

# ================= PREDICTION =================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # 1️⃣ Convert image to NumPy
    img_array = np.array(img)

    # 2️⃣ Convert to float32 (🔥 REQUIRED)
    img_array = img_array.astype(np.float32)

    # 3️⃣ Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # 4️⃣ MobileNetV2 preprocessing
    img_array = (img_array / 127.5) - 1.0

    # 5️⃣ TFLite inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    idx = np.argmax(preds)
    confidence = preds[0][idx] * 100

    st.success(f"Prediction: {index_to_name.get(idx, 'Unknown')}")
    st.write(f"Confidence: {confidence:.2f}%")
