import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os





# Define class names
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
    
    # Load the quantized TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='model_quantized.tflite')
    interpreter.allocate_tensors()
    
    # Get the input and output details of the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    input_index = input_details[0]['index']
    interpreter.set_tensor(input_index, image.astype(np.float32))
    
    # Run the inference
    interpreter.invoke()
    
    # Get the output tensor
    output_index = output_details[0]['index']
    predictions = interpreter.get_tensor(output_index)
    
    # Get the predicted class and confidence
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    confidence = predictions[0][class_index] * 100
    
    return class_name, confidence

# Set page title
st.title(":red[Agricultural Crop Disease Classification]")

# Create the file uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    
    # Classify the image
    class_name, confidence = classify_image(image)
    
    # Display the result
    st.write(f"Class: {class_name}")
    st.write(f"Confidence: {confidence:.2f}%")