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

# ====================================================
# 🔴 CRITICAL PART: CLASS ORDER (MUST MATCH TRAINING)
# This MUST be the SAME order as:
# train_ds.class_names during training
# (alphabetical folder order)
# ====================================================
class_names = [
    '0','1','10','11','12','13','14','15','16','17','18','19',
    '2','20','21','22','23','24','25','26','27','28','29',
    '3','30','31','32','33','34','35','36','37','38','39',
    '4','40','41','42','43','44','45','46','47','48','49',
    '5','50','51','52','53','54','55','56','57','6','7','8','9'
]

# Folder ID → Human readable label
class_id_to_name = {
    "0": "Speed limit (5km/h)",
    "1": "Speed limit (15km/h)",
    "2": "Speed limit (30km/h)",
    "3": "Speed limit (40km/h)",
    "4": "Speed limit (50km/h)",
    "5": "Speed limit (60km/h)",
    "6": "Speed limit (70km/h)",
    "7": "Speed limit (80km/h)",
    "8": "Dont Go straight or left",
    "9": "Dont Go straight or right",
    "10": "Dont Go straight",
    "11": "Dont Go left",
    "12": "Dont Go left or right",
    "13": "Dont Go right",
    "14": "Dont overtake from left",
    "15": "No U-turn",
    "16": "No car",
    "17": "No horn",
    "18": "Speed limit (40km/h)",
    "19": "Speed limit (50km/h)",
    "20": "Go straight or right",
    "21": "Go straight",
    "22": "Go left",
    "23": "Go left or right",
    "24": "Go right",
    "25": "Keep left",
    "26": "Keep right",
    "27": "Roundabout mandatory",
    "28": "Watch out for cars",
    "29": "Horn",
    "30": "Bicycle crossing",
    "31": "U-turn",
    "32": "Road divider",
    "33": "Traffic signals",
    "34": "Danger ahead",
    "35": "Zebra crossing",
    "36": "Bicycle crossing",
    "37": "Children crossing",
    "38": "Dangerous curve left",
    "39": "Dangerous curve right",
    "40": "Unknown 1",
    "41": "Unknown 2",
    "42": "Unknown 3",
    "43": "Go right or straight",
    "44": "Go left or straight",
    "45": "Unknown 4",
    "46": "Zigzag curve",
    "47": "Train crossing",
    "48": "Under construction",
    "49": "Unknown 5",
    "50": "Fences",
    "51": "Heavy vehicle accidents",
    "52": "Unknown 6",
    "53": "Give way",
    "54": "No stopping",
    "55": "No entry",
    "56": "Unknown 7",
    "57": "Unknown 8"
}

# Build FINAL index → label mapping
index_to_name = {
    i: class_id_to_name[class_names[i]]
    for i in range(len(class_names))
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

    # Convert image to numpy
    img_array = np.array(img).astype(np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 preprocessing
    img_array = (img_array / 127.5) - 1.0

    # TFLite inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    idx = int(np.argmax(preds))
    confidence = preds[0][idx] * 100

    st.success(f"Prediction: {index_to_name[idx]}")
    st.write(f"Confidence: {confidence:.2f}%")
