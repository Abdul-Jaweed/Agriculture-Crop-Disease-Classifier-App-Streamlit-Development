import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Define class labels
class_names = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust"
]

# Define the function for image classification
def classify_image(image):
    # Preprocess the image
    image = image.resize((224, 224))  # Resize the image to match the model's input shape
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize the image
    
    # Reshape the image to match the model's input shape
    image = np.reshape(image, (1, 224, 224, 3))
    
    # Load the model
    model = tf.keras.models.load_model('model.h5')
    
    # Perform prediction
    predictions = model.predict(image)
    
    # Get the top 3 predicted classes and their confidences
    top_indices = np.argsort(predictions[0])[::-1][:3]  # Get the indices of top 3 classes
    top_classes = [class_names[i] for i in top_indices]
    top_confidences = [predictions[0][i] * 100 for i in top_indices]
    
    return top_classes, top_confidences

# Set page title
st.title(":red[Agricultural Crop Disease Classification]")

# Create the file uploader and submit button
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
submit_button = st.button("Predict")

if submit_button and uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    
    # Classify the image
    top_classes, top_confidences = classify_image(image)
    
    # Display the top 3 results
    st.write("Top 3 Predictions:")
    for i in range(len(top_classes)):
        st.write(f"{i+1}. Class: {top_classes[i]}, Confidence: {top_confidences[i]:.2f}%")
