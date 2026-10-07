import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("heart_disease_xgb_pipeline.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")

except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()
