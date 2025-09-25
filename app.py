import os
import io
import json
import streamlit as st
from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel, Part

# ---------------- CONFIG ----------------
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
REGION = "us-central1"

# Save service account to a temp file (needed for SDK)
creds_path = "/tmp/service_account.json"
with open(creds_path, "w") as f:
    json.dump(dict(st.secrets["gcp_service_account"]), f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# Init Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Load Nano Banana
MODEL = GenerativeModel("gemini-2.5-flash-image-preview")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Nano Banana Image Editor", layout="wide")
st.title("🍌✨ Nano Banana (Gemini 2.5 Flash Image) Editor")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
prompt = st.text_area("Enter your edit instruction", placeholder="e.g. Turn the background into a futuristic neon city")

if st.button("🚀 Edit Image"):
    if not uploaded_file or not prompt.strip():
        st.warning("Please upload an image and enter a prompt.")
    else:
        with st.spinner("Editing image with Nano Banana..."):
            # Read uploaded image
            image_bytes = uploaded_file.read()
            mime_type = "image/" + uploaded_file.type.split("/")[-1]
            input_image = Part.from_data(mime_type=mime_type, data=image_bytes)

            try:
                resp = MODEL.generate_content([input_image, prompt])

                # Extract edited image
                out_bytes = resp.candidates[0].content.parts[0].data

                # Show before/after
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_bytes, caption="Original", use_container_width=True)
                with col2:
                    st.image(out_bytes, caption="Edited", use_container_width=True)

                # Download button
                st.download_button(
                    "⬇️ Download Edited Image",
                    data=out_bytes,
                    file_name="edited.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
