# Traffic Sign Recognition System

This project implements a deep learning–based traffic sign recognition system that classifies traffic sign images into multiple categories using a convolutional neural network.

The model is trained using transfer learning with MobileNetV2 and optimized for deployment by converting it into a TensorFlow Lite model. The final system is deployed as a Streamlit web application that allows users to upload an image and receive real-time predictions.

---

## Features
- Multi-class traffic sign image classification  
- Transfer learning using MobileNetV2  
- Lightweight and fast inference using TensorFlow Lite  
- Interactive web interface built with Streamlit  
- Real-time prediction with confidence score  

---

## Tech Stack
- Python  
- TensorFlow / TensorFlow Lite  
- Streamlit  
- NumPy  
- Pillow  

---

## Deployment
The model is deployed using **Streamlit Cloud**.  
Users can upload a traffic sign image and get the predicted class instantly.

---

## Model Details
- Architecture: MobileNetV2 (Transfer Learning)  
- Loss Function: Sparse Categorical Cross-Entropy  
- Optimizer: Adam  
- Input Size: 224 × 224 RGB images  

---

## Repository Structure
├── app.py
├── requirements.txt
├── traffic_sign_model.tflite

## Author
Rashmi Patil
