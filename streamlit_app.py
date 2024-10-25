import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3

# Add background image code
st.write('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)

# Load your trained model
model = load_model('Fake-currency.keras')

# Function to preprocess image for the model
def preprocess_image(image):
    # Resize image to the size your model expects
    image = cv2.resize(image, (224, 224))
    # Normalize the image if needed
    image = image / 255.0
    # Expand dimensions to match input shape (1, height, width, channels)
    image = np.expand_dims(image, axis=0)
    return image

# Function for prediction
def predict_currency(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Function to play audio
def play_audio(label, note_value):
    engine = pyttsx3.init()  # Create a new engine instance
    if label == 'Fake Currency':
        engine.say(f"given   {note_value}   rupees   note   fake")
    else:
        engine.say(f"given   {note_value}   rupees   note   real")
    engine.runAndWait()

# Streamlit UI
st.title("Fake Currency Identification")


# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR")

    note_values = ['10', '20', '50', '100', '200', '500', '2000']
    note_value = st.selectbox("Select the note value", note_values)

    if st.button('Detect'):
        st.write('Processing...')
        prediction = predict_currency(image)
        label = 'Fake Currency' if prediction[0][0] > 0.5 else 'Real Currency'
        st.write(f"Prediction: {label}")
        play_audio(label, note_value)