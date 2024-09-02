import streamlit as st
import os
import cv2
import numpy as np
import joblib
from PIL import Image
import requests  # Import to handle API requests

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input size
    img = img.flatten()  # Flatten the image
    img = img.reshape(1, -1)  # Reshape for prediction
    return img

# Function to dynamically get class names from the dataset folder
def get_class_names(dataset_path):
    class_names = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    return class_names

# Function to get disposal suggestions from Gemini AI
def get_disposal_suggestions(predicted_class_name):
    # Replace with the actual endpoint and API key for Gemini AI
    api_url = "https://api.geminiai.com/get-disposal-suggestions"
    headers = {
        "Authorization": "AIzaSyCy4ZTxt1DiSBeySNHw-pYJey70Nc_uQ3I",  # Replace 'YOUR_API_KEY' with the actual API key
        "Content-Type": "application/json"
    }
    data = {
        "class_name": predicted_class_name
    }
    response = requests.post(api_url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json().get('suggestion', "No suggestions available.")
    else:
        return "Error fetching suggestions from Gemini AI."

# Load the trained model
model = joblib.load(r'C:\Users\rathn\OneDrive\Documents\project_HACKHIVE\new_svm_model.pkl')

# Get class names from the dataset folder
dataset_path = r"C:\Users\rathn\OneDrive\Documents\project_HACKHIVE\dataset"
class_names = get_class_names(dataset_path)

# Define a mapping of classes to recyclable or non-recyclable
recyclable_classes = {'paper', 'cardboard', 'plastic'}  # Adjust these class names based on your dataset
non_recyclable_classes = {'metal', 'glass', 'trash'}  # Adjust these class names based on your dataset

# Streamlit app title
st.title("TrashNet Image Classification with Gemini AI")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the image in the app
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = load_and_preprocess_image(image)

    # Make a prediction
    prediction = model.predict(processed_image)

    # Display the prediction with the corresponding class name
    predicted_class_name = prediction[0]  # Directly use the predicted label
    st.write(f"Predicted Class: **{predicted_class_name}**")

    # Determine if the item is recyclable or not
    if predicted_class_name in recyclable_classes:
        st.write("This item is **recyclable**.")
    elif predicted_class_name in non_recyclable_classes:
        st.write("This item is **non-recyclable**.")
    else:
        st.write("Recyclability status is **unknown**.")

    # Get and display disposal suggestions from Gemini AI
    suggestion = get_disposal_suggestions(predicted_class_name)
    st.write(f"Gemini AI Suggestion: **{suggestion}**")
