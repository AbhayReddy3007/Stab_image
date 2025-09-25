import os
import json
import streamlit as st
from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account

# ---------------- CONFIG ----------------
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]

# Create credentials directly from secrets (no metadata server needed)
credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets["gcp_service_account"])
)

# Init Vertex AI with GLOBAL region for Nano Banana
vertexai.init(project=PROJECT_ID, location="global", credentials=credentials)

# Load Gemini 2.5 Flash Image Preview (Nano Banana)
MODEL = GenerativeModel("gemini-2.5-flash-image-preview")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Nano Banana Image Editor", layout="wide")
st.title("🍌✨ Nano Banana (Gemini 2.5 Flash Image) Editor")

# Session history
if "edited_images" not in st.session_state:
    st.session_state.edited_images = []  # list of {"original", "edited", "prompt"}

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
prompt = st.text_area(
    "✏️ Enter your edit instruction",
    placeholder="e.g. Turn the background into a futuristic neon city"
)

if st.button("🚀 Edit Image"):
    if not uploaded_file or not prompt.strip():
        st.warning("Please upload an image and enter a prompt.")
    else:
        with st.spinner("Editing image with Nano Banana..."):
            try:
                # Read uploaded image
                image_bytes = uploaded_file.read()
                mime_type = "image/" + uploaded_file.type.split("/")[-1]
                input_image = Part.from_data(mime_type=mime_type, data=image_bytes)

                # Call Gemini image model
                resp = MODEL.generate_content([input_image, prompt])

                # Extract edited image (correct way)
                out_bytes = resp.candidates[0].content.parts[0].inline_data.data

                # Show before/after
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_bytes, caption="Original", use_column_width=True)
                with col2:
                    st.image(out_bytes, caption="Edited", use_column_width=True)

                # Download button
                st.download_button(
                    "⬇️ Download Edited Image",
                    data=out_bytes,
                    file_name="edited.png",
                    mime="image/png"
                )

                # Save to session history
                st.session_state.edited_images.append(
                    {"original": image_bytes, "edited": out_bytes, "prompt": prompt}
                )

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# ---------------- HISTORY ----------------
if st.session_state.edited_images:
    st.subheader("📂 Edit History (this session)")
    for i, entry in enumerate(reversed(st.session_state.edited_images[-20:])):  # last 20
        with st.expander(f"{i+1}. Prompt: {entry['prompt']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(entry["original"], caption="Original", use_column_width=True)
            with col2:
                st.image(entry["edited"], caption="Edited", use_column_width=True)

            st.download_button(
                "⬇️ Download Edited Image",
                data=entry["edited"],
                file_name=f"edited_{i}.png",
                mime="image/png",
                key=f"download_history_{i}"
            )
