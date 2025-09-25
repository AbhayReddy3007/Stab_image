import os
import re
import datetime
import json
from io import BytesIO
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

# Init Vertex AI (global required for Nano Banana)
vertexai.init(project=PROJECT_ID, location="global", credentials=credentials)

# Models
IMAGE_MODEL = GenerativeModel("gemini-2.5-flash-image-preview")  # Nano Banana
TEXT_MODEL = GenerativeModel("gemini-2.0-flash")  # Prompt refinement

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Image Generator + Editor", layout="wide")
st.title("🖼️ AI Image Generator + Editor (Nano Banana + Smart Refinement + WebP + Multi-Edit + Send to Edit)")

# ---------------- STATE ----------------
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []  # [{"filename","content"}]
if "edited_images" not in st.session_state:
    st.session_state.edited_images = []  # [{"original","edited","prompt"}]
if "selected_for_edit" not in st.session_state:
    st.session_state.selected_for_edit = None  # store image bytes from Generate tab

# ---------------- Prompt Templates ----------------
PROMPT_TEMPLATES = {
    "Marketing": """Refine for marketing visuals:
User’s input:
"{USER_PROMPT}"

Refined prompt:
""",
    "Design": """Refine for design visuals:
User’s input:
"{USER_PROMPT}"

Refined prompt:
""",
    "General": """Refine for general visuals:
User’s input:
"{USER_PROMPT}"

Refined prompt:
""",
    "DPEX": """Refine for IT/tech visuals:
User’s input:
"{USER_PROMPT}"

Refined prompt:
""",
    "HR": """Refine for HR/workplace visuals:
User’s input:
"{USER_PROMPT}"

Refined prompt:
""",
    "Business": """Refine for business/corporate visuals:
User’s input:
"{USER_PROMPT}"

Refined prompt:
"""
}

STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep natural.",
    "Smart": "Clean and balanced look, professional but neutral.",
    "Cinematic": "Film-style lighting and composition, storytelling vibe.",
    "Creative": "Playful, imaginative, bold artistic choices.",
    "Minimalist": "Simple, uncluttered, with negative space.",
    "Vibrant": "Bold, saturated colors, energetic tone.",
    "Cyberpunk": "Futuristic neon city vibes, glowing lights, dark tones.",
    "Stock Photo": "Commercial quality, business-friendly aesthetic.",
    "Portrait": "Focus on the subject with natural tones, shallow DOF.",
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


def run_edit_flow(edit_prompt, base_bytes, filename):
    """Run Gemini edit on base_bytes with given edit_prompt, with fallback"""
    input_image = Part.from_data(mime_type="image/png", data=base_bytes)

    # Force Gemini to interpret as edit task
    edit_instruction = f"Edit the provided image as follows: {edit_prompt}. Always return only the edited image as inline PNG."

    resp = IMAGE_MODEL.generate_content([edit_instruction, input_image])

    out_bytes = None
    text_fallback = None

    for part in resp.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data.data:
            out_bytes = part.inline_data.data
        elif hasattr(part, "text") and part.text:
            text_fallback = part.text

    if out_bytes:
        return out_bytes
    else:
        if text_fallback:
            st.warning(f"⚠️ Gemini did not return an image. Response: {text_fallback}")
        return None


# ---------------- TABS ----------------
tab_generate, tab_edit = st.tabs(["✨ Generate Images", "🖌️ Edit Images"])

# ---------------- GENERATE MODE ----------------
with tab_generate:
    st.header("✨ Generate Images from Scratch")

    dept_gen = st.selectbox("🏢 Department", options=list(PROMPT_TEMPLATES.keys()), index=2, key="dept_gen")
    style_gen = st.selectbox("🎨 Style", options=list(STYLE_DESCRIPTIONS.keys()), index=0, key="style_gen")
    raw_prompt_gen = st.text_area("Enter your prompt", height=120, key="prompt_gen")
    num_images = st.slider("🧾 Number of images", 1, 4, 1, key="num_gen")

    if st.button("🚀 Generate", key="gen_btn"):
        if not raw_prompt_gen.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Refining prompt with Gemini..."):
                refinement_prompt = PROMPT_TEMPLATES[dept_gen].replace("{USER_PROMPT}", raw_prompt_gen)
                if style_gen != "None":
                    refinement_prompt += f"\n\nApply the style: {STYLE_DESCRIPTIONS[style_gen]}"
                text_resp = TEXT_MODEL.generate_content(refinement_prompt)
                enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                st.info(f"🔮 Enhanced Prompt:\n\n{enhanced_prompt}")

            with st.spinner("Generating images with Nano Banana..."):
                generated_raws = []
                try:
                    for i in range(num_images):
                        resp = IMAGE_MODEL.generate_content([enhanced_prompt])
                        for part in resp.candidates[0].content.parts:
                            if hasattr(part, "inline_data") and part.inline_data.data:
                                generated_raws.append(part.inline_data.data)
                except Exception as e:
                    st.error(f"⚠️ Image generation error: {e}")

                if generated_raws:
                    for idx, img_bytes in enumerate(generated_raws):
                        filename = f"{dept_gen.lower()}_{style_gen.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
                        st.session_state.generated_images.append({"filename": filename, "content": img_bytes})

                        with st.container():
                            st.image(Image.open(BytesIO(img_bytes)), caption=filename, use_column_width=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button("⬇️ Download", data=img_bytes, file_name=filename, mime="image/png", key=f"dl_{idx}")
                            with col2:
                                if st.button("✏️ Send to Edit tab", key=f"send_edit_{idx}"):
                                    st.session_state.selected_for_edit = {"filename": filename, "content": img_bytes}
                                    st.success("✅ Image sent to Edit tab! Switch to Edit tab to continue.")


# ---------------- EDIT MODE ----------------
with tab_edit:
    st.header("🖌️ Edit Images")

    # Check if image was sent from Generate tab
    if st.session_state.get("selected_for_edit"):
        base_image = st.session_state.selected_for_edit["content"]
        st.image(Image.open(BytesIO(base_image)),
                 caption=f"Image from Generate tab ({st.session_state.selected_for_edit['filename']})",
                 use_column_width=True)
        if st.button("❌ Clear selected image"):
            st.session_state.selected_for_edit = None
            base_image = None
    else:
        uploaded_file = st.file_uploader("📤 Or upload an image", type=["png", "jpg", "jpeg", "webp"])
        base_image = None
        if uploaded_file:
            image_bytes = uploaded_file.read()
            mime_type = "image/" + uploaded_file.type.split("/")[-1]
            if mime_type == "image/webp":  # ✅ Convert WebP → PNG
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                buf = BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
            base_image = image_bytes

    dept_edit = st.selectbox("🏢 Department", options=list(PROMPT_TEMPLATES.keys()), index=2, key="dept_edit")
    style_edit = st.selectbox("🎨 Style", options=list(STYLE_DESCRIPTIONS.keys()), index=0, key="style_edit")
    raw_prompt_edit = st.text_area("Enter your edit instruction", height=120, key="prompt_edit")
    num_edit_images = st.slider("🧾 Number of edited images", 1, 4, 1, key="num_edit")

    if st.button("🚀 Edit Image", key="edit_btn_upload"):
        if not base_image or not raw_prompt_edit.strip():
            st.warning("Please select/upload an image and enter an instruction.")
        else:
            with st.spinner("Refining edit instruction with Gemini..."):
                refinement_prompt = PROMPT_TEMPLATES[dept_edit].replace("{USER_PROMPT}", raw_prompt_edit)
                if style_edit != "None":
                    refinement_prompt += f"\n\nApply the style: {STYLE_DESCRIPTIONS[style_edit]}"
                text_resp = TEXT_MODEL.generate_content(refinement_prompt)
                enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                st.info(f"🔮 Enhanced Instruction:\n\n{enhanced_prompt}")

            with st.spinner("Editing image with Nano Banana..."):
                edited_versions = []
                for i in range(num_edit_images):
                    out_bytes = run_edit_flow(
                        enhanced_prompt,
                        base_image,
                        f"edit_{datetime.datetime.now().strftime('%H%M%S')}_{i}"
                    )
                    if out_bytes:
                        edited_versions.append(out_bytes)

                if edited_versions:
                    cols = st.columns(len(edited_versions))
                    for i, out_bytes in enumerate(edited_versions):
                        with cols[i]:
                            st.image(Image.open(BytesIO(out_bytes)), caption=f"Edited Version {i+1}", use_column_width=True)
                            st.download_button(
                                f"⬇️ Download Edited {i+1}",
                                data=out_bytes,
                                file_name=f"edited_{i}.png",
                                mime="image/png",
                                key=f"edit_download_{i}"
                            )
                            st.session_state.edited_images.append({
                                "original": base_image,
                                "edited": out_bytes,
                                "prompt": enhanced_prompt
                            })

# ---------------- HISTORY ----------------
st.subheader("📂 History")
if st.session_state.generated_images:
    st.markdown("### Generated Images")
    for i, img in enumerate(reversed(st.session_state.generated_images[-20:])):
        with st.expander(f"Generated {i+1}: {img['filename']}"):
            st.image(Image.open(BytesIO(img["content"])), caption=img["filename"], use_column_width=True)
            st.download_button("⬇️ Download Again", data=img["content"], file_name=img["filename"], mime="image/png", key=f"gen_hist_{i}")

if st.session_state.edited_images:
    st.markdown("### Edited Images")
    for i, entry in enumerate(reversed(st.session_state.edited_images[-20:])):
        with st.expander(f"Edited {i+1}: {entry['prompt']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(BytesIO(entry["original"])), caption="Original", use_column_width=True)
            with col2:
                st.image(Image.open(BytesIO(entry["edited"])), caption="Edited", use_column_width=True)
            st.download_button("⬇️ Download Edited", data=entry["edited"], file_name=f"edited_{i}.png", mime="image/png", key=f"edit_hist_{i}")
