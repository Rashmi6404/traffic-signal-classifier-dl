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
 '0': 'Speed limit (5km/h)',
 '1': 'Speed limit (15km/h)',
 '2': 'Speed limit (30km/h)',
 '3': 'Speed limit (40km/h)',
 '4': 'Speed limit (50km/h)',
 '5': 'Speed limit (60km/h)',
 '6': 'Speed limit (70km/h)',
 '7': 'speed limit (80km/h)',
 '8': 'Dont Go straight or left',
 '9': 'Dont Go straight or Right',
 '10': 'Dont Go straight',
 '11': 'Dont Go Left',
 '12': 'Dont Go Left or Right',
 '13': 'Dont Go Right',
 '14': 'Dont overtake from Left',
 '15': 'No Uturn',
 '16': 'No Car',
 '17': 'No horn',
 '18': 'Speed limit (40km/h)',
 '19': 'Speed limit (50km/h)',
 '20': 'Go straight or right',
 '21': 'Go straight',
 '22': 'Go Left',
 '23': 'Go Left or right',
 '24': 'Go Right',
 '25': 'keep Left',
 '26': 'keep Right',
 '27': 'Roundabout mandatory',
 '28': 'watch out for cars',
 '29': 'Horn',
 '30': 'Bicycles crossing',
 '31': 'Uturn',
 '32': 'Road Divider',
 '33': 'Traffic signals',
 '34': 'Danger Ahead',
 '35': 'Zebra Crossing',
 '36': 'Bicycles crossing',
 '37': 'Children crossing',
 '38': 'Dangerous curve to the left',
 '39': 'Dangerous curve to the right',
 '40': 'Unknown1',
 '41': 'Unknown2',
 '42': 'Unknown3',
 '43': 'Go right or straight',
 '44': 'Go left or straight',
 '45': 'Unknown4',
 '46': 'ZigZag Curve',
 '47': 'Train Crossing',
 '48': 'Under Construction',
 '49': 'Unknown5',
 '50': 'Fences',
 '51': 'Heavy Vehicle Accidents',
 '52': 'Unknown6',
 '53': 'Give Way',
 '54': 'No stopping',
 '55': 'No entry',
 '56': 'Unknown7',
 '57': 'Unknown8'
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

