import os
import re
import datetime
import json
import io
import streamlit as st
from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel

# ---------------- CONFIG ----------------
PROJECT_ID = "drl-zenai-prod"
REGION = "us-central1"

# Load Google Cloud credentials from Streamlit secrets
creds_path = "/tmp/service_account.json"
service_account_info = dict(st.secrets["gcp_service_account"])
with open(creds_path, "w") as f:
    json.dump(service_account_info, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# Init Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Models
IMAGE_MODEL_NAME = "gemini-2.5-flash-image-preview"
IMAGE_MODEL = GenerativeModel(IMAGE_MODEL_NAME)

TEXT_MODEL_NAME = "gemini-2.0-flash"
TEXT_MODEL = GenerativeModel(TEXT_MODEL_NAME)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Banana Image Generator", layout="wide")
st.title("🖼️ AI Image Generator (Gemini 2.5 Flash Image Preview)")

# ---------------- STATE ----------------
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# ---------------- UI ----------------
department = st.selectbox(
    "🏢 Select Department",
    options=["Marketing", "Design", "General", "DPEX", "HR", "Business"],
    index=2
)

STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the image natural, faithful to the user’s idea.",
    "Smart": "A clean, balanced, and polished look. Professional yet neutral, visually appealing without strong artistic bias.",
    "Cinematic": "Film-style composition with professional lighting. Wide dynamic range, dramatic highlights, storytelling feel.",
    "Creative": "Playful, imaginative, and experimental. Bold artistic choices, unexpected elements, and expressive color use.",
    "Illustration": "Hand-drawn or digitally illustrated style. Clear outlines, stylized shading, expressive and artistic.",
    "3D Render": "Photorealistic or stylized CGI. Crisp geometry, depth, shadows, and reflective surfaces with realistic rendering.",
    "Minimalist": "Simple and uncluttered. Few elements, large negative space, flat or muted color palette, clean composition.",
    "Portrait": "Focus on the subject. Natural skin tones, shallow depth of field, close-up or waist-up framing, studio or natural lighting.",
    "Stock Photo": "Professional, commercial-quality photo. Neutral subject matter, polished composition, business-friendly aesthetic.",
    "Vibrant": "Bold, saturated colors. High contrast, energetic mood, eye-catching and lively presentation.",
    "Fantasy Art": "Epic fantasy scenes. Magical elements, mythical creatures, enchanted landscapes.",
    "Cyberpunk": "Futuristic neon city vibes. High contrast, glowing lights, dark tones, sci-fi feel.",
}

# ---------------- Department-aware Style Suggestions ----------------
DEPARTMENT_STYLE_MAP = {
    "Marketing": ["Fashion", "Vibrant", "Stock Photo", "Cinematic", "Minimalist"],
    "Design": ["Vector", "Creative", "Pop Art", "Illustration", "3D Render"],
    "General": ["Smart", "Cinematic", "Portrait", "Stock Photo"],
    "DPEX": ["Moody", "Cinematic", "3D Render", "Cyberpunk"]
}

def get_styles_for_department(dept):
    base_styles = DEPARTMENT_STYLE_MAP.get(dept, [])
    all_styles = ["None"] + base_styles + [s for s in STYLE_DESCRIPTIONS.keys() if s not in base_styles and s != "None"]
    return all_styles

styles_for_dept = get_styles_for_department(department)

style = st.selectbox(
    "🎨 Choose Style",
    options=styles_for_dept,
    index=0
)

raw_prompt = st.text_area("Enter your prompt to generate an image:", height=120)
num_images = st.slider("🧾 Number of images", 1, 4, 1)

# ---------------- Prompt Templates ----------------
PROMPT_TEMPLATES = {
    "Marketing": """
You are a senior AI prompt engineer creating polished prompts for marketing and advertising visuals.
Expand the raw input into a compelling, professional image prompt with details about background, lighting, style, composition, and branding tone.

User’s raw prompt:
"{USER_PROMPT}"

Refined marketing image prompt:
""",
    "Design": """
You are a senior AI prompt engineer supporting a creative design team.
Expand the raw input into a visually inspiring prompt with details on artistic style, color scheme, composition, and creative tone.

User’s raw prompt:
"{USER_PROMPT}"

Refined design image prompt:
""",
    "General": """
You are an expert AI prompt engineer creating vivid and descriptive image prompts.
Expand the input with details on background, lighting, style, perspective, and mood.

User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
""",
    "DPEX": """
You are a senior AI prompt engineer creating refined prompts for IT and technology visuals.
Expand the raw input into a detailed, futuristic image prompt with servers, code, UIs, and neon tech ambiance.

User’s raw prompt:
"{USER_PROMPT}"

Refined DPEX image prompt:
""",
    "HR": """
You are a senior AI prompt engineer creating refined prompts for HR visuals.
Expand the raw input into a workplace-focused prompt with office settings, teamwork, diversity, and professional tone.

User’s raw prompt:
"{USER_PROMPT}"

Refined HR image prompt:
""",
    "Business": """
You are a senior AI prompt engineer creating refined prompts for business visuals.
Expand the raw input into a corporate, professional prompt with boardrooms, executives, teamwork, and city skylines.

User’s raw prompt:
"{USER_PROMPT}"

Refined business image prompt:
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

def _sanitize_key(s: str):
    return re.sub(r'[^0-9a-zA-Z_-]+', '_', s)

# ---------------- Generation flow ----------------
if st.button("🚀 Generate Image"):
    if not raw_prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        # 1) refine prompt with Gemini text model
        with st.spinner("Refining prompt with Gemini..."):
            try:
                refinement_prompt = PROMPT_TEMPLATES[department].replace("{USER_PROMPT}", raw_prompt)
                if style != "None":
                    refinement_prompt += f"\n\nApply the visual style: {STYLE_DESCRIPTIONS[style]}"
                text_resp = TEXT_MODEL.generate_content(refinement_prompt)
                enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                st.info(f"🔮 Enhanced Prompt ({department} / {style}):\n\n{enhanced_prompt}")
            except Exception as e:
                st.error(f"⚠️ Gemini prompt refinement error: {e}")
                st.stop()

        # 2) generate images with Gemini 2.5 Flash Image
        with st.spinner("Generating image(s) with Gemini 2.5 Flash Image..."):
            generated_raws = []
            try:
                for i in range(num_images):
                    resp = IMAGE_MODEL.generate_content(
                        [enhanced_prompt],
                        generation_config={"response_mime_type": "image/png"}
                    )

                    for part in resp.candidates[0].content.parts:
                        if hasattr(part, "inline_data") and part.inline_data.data:
                            img_bytes = part.inline_data.data
                            generated_raws.append(img_bytes)
            except Exception as e:
                st.error(f"⚠️ Image generation error: {e}")
                st.stop()

            # 3) Show images
            if generated_raws:
                cols = st.columns(len(generated_raws))
                for idx, img_bytes in enumerate(generated_raws):
                    col = cols[idx]
                    filename = f"{department.lower()}_{style.lower()}_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
                    safe_key = _sanitize_key(filename)

                    output_dir = os.path.join(os.path.dirname(__file__), "generated_images")
                    os.makedirs(output_dir, exist_ok=True)
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)

                    with col:
                        st.image(img_bytes, caption=filename, use_column_width=True)
                        st.download_button(
                            "⬇️ Download",
                            data=img_bytes,
                            file_name=filename,
                            mime="image/png",
                            key=f"dl_{safe_key}"
                        )

                    st.session_state.generated_images.append({"filename": filename, "content": img_bytes})
            else:
                st.error("❌ No images produced by the model.")

# ---------------- HISTORY ----------------
if st.session_state.generated_images:
    st.subheader("📂 Past Generated Images")
    for i, img in enumerate(reversed(st.session_state.generated_images[-40:])):
        with st.expander(f"{i+1}. {img['filename']}"):
            st.image(img["content"], caption=img["filename"], use_container_width=True)
            st.download_button(
                "⬇️ Download Again",
                data=img["content"],
                file_name=img["filename"],
                mime="image/png",
                key=f"download_img_{i}"
            )
