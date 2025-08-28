# app.py
import streamlit as st
from utils import load_audio_for_yamnet
from model import analyze
import numpy as np
import io

st.set_page_config(page_title="🐾 Pet Translator", layout="centered")

st.title("🐾 Pet Translator — Demo (MVP)")
st.markdown("""
Upload a short recording of your pet (wav or mp3).  
This demo uses **YAMNet** to detect sound classes (bark, meow, purr, whine etc.) and maps them by heuristics to simple intents.
""")

uploaded = st.file_uploader("Upload pet audio (wav/mp3)", type=["wav", "mp3", "m4a", "flac"])

if uploaded is None:
    st.info("Try uploading a short bark / meow clip (1–10 seconds). You can also test with your phone recordings.")
    # Optionally provide a sample download link or sample sounds if you want
else:
    # Show audio player
    st.audio(uploaded)
    with st.spinner("Analyzing sound..."):
        # Streamlit's uploaded file is a BytesIO - we need to keep a copy for soundfile
        try:
            # Reset pointer
            uploaded.seek(0)
            waveform = load_audio_for_yamnet(uploaded)
            result = analyze(waveform)
            st.success(f"Prediction: **{result['intent']}**")
            st.write(result['rationale'])
            st.markdown("**Top detected classes (from YAMNet)**:")
            for cls, score in result['top'][:6]:
                st.write(f"- {cls} — {score:.3f}")
            st.markdown("---")
            st.info("This is a heuristic-based demo. For better accuracy, combine sound + motion + heart-rate or fine-tune a model on labelled pet sounds.")
        except Exception as e:
            st.error("Error processing audio: " + str(e))
