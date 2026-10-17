import streamlit as st
import joblib
import librosa
import numpy as np
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Mental Health Companion",
    page_icon="🧠",
    layout="centered"
)

# Application heading
st.title("🧠 Mental Health Companion")
st.write(
    "Upload a voice recording to explore the emotion "
    "predicted by our machine learning model."
)

st.info(
    "This project demonstrates voice-based emotion recognition. "
    "It is not a medical diagnostic tool."
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("emotion_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Could not load the trained model.")
    st.error(str(e))
    st.stop()


# Extract audio features
def extract_features(file_path):
    audio, sr = librosa.load(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0).reshape(1, -1)


# Upload audio
uploaded_file = st.file_uploader(
    "Choose a voice recording",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file, format="audio/wav")

    if st.button("Analyze Emotion"):

        temp_path = None

        try:
            # Save uploaded audio temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            # Extract features and predict emotion
            features = extract_features(temp_path)
            prediction = model.predict(features)

            st.success("Emotion analysis completed!")

            st.subheader("Predicted Emotion")
            st.write(str(prediction[0]))

            st.caption(
                "This is a machine learning prediction, "
                "not a clinical assessment."
            )

        except Exception as e:
            st.error("Unable to analyze this audio recording.")
            st.error(str(e))

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
