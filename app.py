import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cifar10_cnn_model.h5')

# Define the class labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Streamlit webpage title
st.title("CIFAR-10 Image Classification")

# Image upload feature
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    def preprocess_image(image, target_size=(32, 32)):
        # Resize the image to match the model's input size
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        image = np.asarray(image)
        image = (image / 255.0).astype(np.float32)
        image = np.expand_dims(image, 0)  # Add batch dimension
        return image

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    # Display the prediction
    st.write(f"Prediction: {predicted_class}")
