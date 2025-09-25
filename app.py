import os
import re
import datetime
import json
import streamlit as st
from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account

# ---------------- CONFIG ----------------
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]

# Create credentials from secrets
credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets["gcp_service_account"])
)

# Init Vertex AI (global is required for Nano Banana)
vertexai.init(project=PROJECT_ID, location="global", credentials=credentials)

# Models
IMAGE_MODEL = GenerativeModel("gemini-2.5-flash-image-preview")  # Nano Banana
TEXT_MODEL = GenerativeModel("gemini-2.0-flash")  # For prompt refinement

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Image Editor", layout="wide")
st.title("🖼️ AI Image Editor (Nano Banana + Smart Refinement)")

# ---------------- STATE ----------------
if "edited_images" not in st.session_state:
    st.session_state.edited_images = []  # history

# ---------------- UI ----------------
department = st.selectbox(
    "🏢 Select Department",
    options=["Marketing", "Design", "General", "DPEX", "HR", "Business"],
    index=2
)

STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the edit natural.",
    "Smart": "Clean and balanced look, professional but neutral.",
    "Cinematic": "Film-style lighting and composition, storytelling vibe.",
    "Creative": "Playful, imaginative, bold artistic choices.",
    "Minimalist": "Simple, uncluttered, with negative space.",
    "Vibrant": "Bold, saturated colors, energetic tone.",
    "Cyberpunk": "Futuristic neon city vibes, glowing lights, dark tones.",
    "Stock Photo": "Commercial quality, business-friendly aesthetic.",
    "Portrait": "Focus on the subject with natural tones, shallow DOF.",
}

style = st.selectbox(
    "🎨 Choose Style",
    options=list(STYLE_DESCRIPTIONS.keys()),
    index=0
)

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
raw_prompt = st.text_area(
    "✏️ Enter your edit instruction",
    placeholder="e.g. Replace the background with a futuristic neon city"
)

# ---------------- Prompt Templates ----------------
PROMPT_TEMPLATES = {
    "Marketing": """
You are a senior AI prompt engineer creating polished prompts for marketing visuals.
Refine the raw instruction into a campaign-ready, persuasive edit.

User’s raw instruction:
"{USER_PROMPT}"

Refined marketing edit instruction:
""",
    "Design": """
You are a senior AI prompt engineer supporting a design team.
Refine the raw input into a visually inspiring, creative edit prompt.

User’s raw instruction:
"{USER_PROMPT}"

Refined design edit instruction:
""",
    "General": """
You are an expert AI prompt engineer.
Refine the raw instruction into a clear, descriptive edit for image generation.

User’s raw instruction:
"{USER_PROMPT}"

Refined general edit instruction:
""",
    "DPEX": """
You are a senior AI prompt engineer creating IT/tech visuals.
Refine the raw input into a futuristic, technical edit prompt.

User’s raw instruction:
"{USER_PROMPT}"

Refined DPEX edit instruction:
""",
    "HR": """
You are a senior AI prompt engineer creating HR visuals.
Refine the raw input into a workplace/culture-focused edit prompt.

User’s raw instruction:
"{USER_PROMPT}"

Refined HR edit instruction:
""",
    "Business": """
You are a senior AI prompt engineer creating business/corporate visuals.
Refine the raw input into a professional, ambitious edit prompt.

User’s raw instruction:
"{USER_PROMPT}"

Refined business edit instruction:
"""
}

# ---------------- Helpers ----------------
def safe_get_enhanced_text(resp):
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            pass
    return str(resp)

# ---------------- Generation Flow ----------------
if st.button("🚀 Edit Image"):
    if not uploaded_file or not raw_prompt.strip():
        st.warning("Please upload an image and enter an instruction.")
    else:
        with st.spinner("Refining edit instruction with Gemini..."):
            try:
                refinement_prompt = PROMPT_TEMPLATES[department].replace("{USER_PROMPT}", raw_prompt)
                if style != "None":
                    refinement_prompt += f"\n\nApply the visual style: {STYLE_DESCRIPTIONS[style]}"
                text_resp = TEXT_MODEL.generate_content(refinement_prompt)
                enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                st.info(f"🔮 Enhanced Instruction ({department} / {style}):\n\n{enhanced_prompt}")
            except Exception as e:
                st.error(f"⚠️ Gemini refinement error: {e}")
                st.stop()

        with st.spinner("Editing image with Nano Banana..."):
            try:
                # Prepare input image
                image_bytes = uploaded_file.read()
                mime_type = "image/" + uploaded_file.type.split("/")[-1]
                input_image = Part.from_data(mime_type=mime_type, data=image_bytes)

                # Call Gemini image model
                resp = IMAGE_MODEL.generate_content([input_image, enhanced_prompt])

                # Extract edited image
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
                    {"original": image_bytes, "edited": out_bytes, "prompt": enhanced_prompt}
                )

            except Exception as e:
                st.error(f"⚠️ Error editing image: {e}")

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
