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
st.title("🖼️ AI Image Generator + Editor")

# ---------------- STATE ----------------
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []  # [{"filename","content"}]
if "edited_images" not in st.session_state:
    st.session_state.edited_images = []  # [{"original","edited","prompt"}]

# ---------------- Prompt Templates ----------------
PROMPT_TEMPLATES = {
    "Marketing": """
You are a senior AI prompt engineer creating polished prompts for marketing and advertising visuals.

Your job:
- Transform the raw input into a compelling, professional, campaign-ready image prompt.
- Expand with persuasive details about:
  • Background and setting (modern, lifestyle, commercial, aspirational)
  • Lighting and atmosphere (studio lights, golden hour, cinematic)
  • Style (photorealistic, cinematic, product photography, lifestyle branding)
  • Perspective and composition (wide shot, close-up, dramatic angles)
  • Mood, tone, and branding suitability (premium, sleek, aspirational)

Rules:
- Stay faithful to the user’s idea but elevate it for ads, social media, or presentations.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined marketing image prompt:
""",

    "Design": """
You are a senior AI prompt engineer supporting a creative design team.

Your job:
- Expand raw input into a visually inspiring, design-oriented image prompt.
- Add imaginative details about:
  • Artistic styles (minimalist, abstract, futuristic, flat, 3D render, watercolor, digital illustration)
  • Color schemes, palettes, textures, and patterns
  • Composition and balance (symmetry, negative space, creative framing)
  • Lighting and atmosphere (soft glow, vibrant contrast, surreal shading)
  • Perspective (isometric, top-down, wide shot, close-up)

Rules:
- Keep fidelity to the idea but make it highly creative and visually unique.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined design image prompt:
""",

    "General": """
You are an expert AI prompt engineer specialized in creating vivid and descriptive image prompts.

Your job:
- Expand the user’s input into a detailed, clear prompt for an image generation model.
- Add missing details such as:
  • Background and setting
  • Lighting and mood
  • Style and realism level
  • Perspective and composition

Rules:
- Stay true to the user’s intent.
- Keep language concise, descriptive, and expressive.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
""",

    "DPEX": """
You are a senior AI prompt engineer creating refined prompts for IT and technology-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and technology-focused image prompt.
- Expand with contextual details about:
  • Technology environments (server rooms, data centers, cloud systems, coding workspaces)
  • Digital elements (network diagrams, futuristic UIs, holograms, cybersecurity visuals)
  • People in IT roles (developers, engineers, admins, tech support, collaboration)
  • Tone (innovative, technical, futuristic, professional)
  • Composition (screens, servers, code on monitors, abstract digital patterns)
  • Lighting and effects (LED glow, cyberpunk tones, neon highlights, modern tech ambiance)

Rules:
- Ensure images are suitable for IT presentations, product demos, training, technical documentation, and digital transformation campaigns.
- Stay true to the user’s intent but emphasize a technological and innovative look.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined DPEX image prompt:
""",

    "HR": """
You are a senior AI prompt engineer creating refined prompts for human resources and workplace-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and HR-focused image prompt.
- Expand with contextual details about:
  • Workplace settings (modern office, meeting rooms, open workspaces, onboarding sessions)
  • People interactions (interviews, teamwork, training, collaboration, diversity and inclusion)
  • Themes (employee engagement, professional growth, recruitment, performance evaluation)
  • Composition (groups in discussion, managers mentoring, collaborative workshops)
  • Lighting and tone (bright, welcoming, professional, inclusive)

Rules:
- Ensure images are suitable for HR presentations, recruitment campaigns, internal training, or employee engagement material.
- Stay true to the user’s intent but emphasize people, culture, and workplace positivity.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined HR image prompt:
""",

    "Business": """
You are a senior AI prompt engineer creating refined prompts for business and corporate visuals.

Your job:
- Transform the raw input into a detailed, professional, and business-oriented image prompt.
- Expand with contextual details about:
  • Corporate settings (boardrooms, skyscrapers, modern offices, networking events)
  • Business activities (presentations, negotiations, brainstorming sessions, teamwork)
  • People (executives, entrepreneurs, consultants, diverse teams, global collaboration)
  • Tone (professional, ambitious, strategic, innovative)
  • Composition (formal meetings, handshake deals, conference tables, city skyline backgrounds)
  • Lighting and atmosphere (clean, modern, premium, professional)

Rules:
- Ensure images are suitable for corporate branding, investor decks, strategy sessions, or professional reports.
- Stay true to the user’s intent but emphasize professionalism, ambition, and success.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined business image prompt:
"""
}


STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the image natural, faithful to the user’s idea.",
    "Smart": "A clean, balanced, and polished look. Professional yet neutral, visually appealing without strong artistic bias.",
    "Cinematic": "Film-style composition with professional lighting. Wide dynamic range, dramatic highlights, storytelling feel.",
    "Creative": "Playful, imaginative, and experimental. Bold artistic choices, unexpected elements, and expressive color use.",
    "Bokeh": "Photography style with shallow depth of field. Subject in sharp focus with soft, dreamy, blurred backgrounds.",
    "Macro": "Extreme close-up photography. High detail, textures visible, shallow focus highlighting minute features.",
    "Illustration": "Hand-drawn or digitally illustrated style. Clear outlines, stylized shading, expressive and artistic.",
    "3D Render": "Photorealistic or stylized CGI. Crisp geometry, depth, shadows, and reflective surfaces with realistic rendering.",
    "Fashion": "High-end editorial photography. Stylish, glamorous poses, bold makeup, controlled lighting, and modern aesthetic.",
    "Minimalist": "Simple and uncluttered. Few elements, large negative space, flat or muted color palette, clean composition.",
    "Moody": "Dark, atmospheric, and emotional. Strong shadows, high contrast, deep tones, cinematic ambiance.",
    "Portrait": "Focus on the subject. Natural skin tones, shallow depth of field, close-up or waist-up framing, studio or natural lighting.",
    "Stock Photo": "Professional, commercial-quality photo. Neutral subject matter, polished composition, business-friendly aesthetic.",
    "Vibrant": "Bold, saturated colors. High contrast, energetic mood, eye-catching and lively presentation.",
    "Pop Art": "Comic-book and pop-art inspired. Bold outlines, halftone patterns, flat vivid colors, high contrast, playful tone.",
    "Vector": "Flat vector graphics. Smooth shapes, sharp edges, solid fills, and clean scalable style like logos or icons.",

    "Watercolor": "Soft, fluid strokes with delicate blending and washed-out textures. Artistic and dreamy.",
    "Oil Painting": "Rich, textured brushstrokes. Classic fine art look with deep color blending.",
    "Charcoal": "Rough, sketchy textures with dark shading. Artistic, raw, dramatic effect.",
    "Line Art": "Minimal monochrome outlines with clean, bold strokes. No shading, focus on form.",

    "Anime": "Japanese animation style with vibrant colors, clean outlines, expressive features, and stylized proportions.",
    "Cartoon": "Playful, exaggerated features, simplified shapes, bold outlines, and bright colors.",
    "Pixel Art": "Retro digital art style. Small, pixel-based visuals resembling old-school video games.",

    "Fantasy Art": "Epic fantasy scenes. Magical elements, mythical creatures, enchanted landscapes.",
    "Surreal": "Dreamlike, bizarre imagery. Juxtaposes unexpected elements, bending reality.",
    "Concept Art": "Imaginative, detailed artwork for games or films. Often moody and cinematic.",

    "Cyberpunk": "Futuristic neon city vibes. High contrast, glowing lights, dark tones, sci-fi feel.",
    "Steampunk": "Retro-futuristic style with gears, brass, Victorian aesthetics, and industrial design.",
    "Neon Glow": "Bright neon outlines and glowing highlights. Futuristic, nightlife aesthetic.",
    "Low Poly": "Simplified 3D style using flat geometric shapes and polygons.",
    "Isometric": "3D look with isometric perspective. Often used for architecture, games, and diagrams.",

    "Vintage": "Old-school, retro tones. Faded colors, film grain, sepia, or retro print feel.",
    "Graffiti": "Urban street art style with bold colors, spray paint textures, and rebellious tone."
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
    st.header("✨ Generate Images ")

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

                        st.image(Image.open(BytesIO(img_bytes)), caption=filename, use_column_width=True)
                        st.download_button("⬇️ Download", data=img_bytes, file_name=filename, mime="image/png", key=f"dl_{idx}")


# ---------------- EDIT MODE ----------------
with tab_edit:
    st.header("🖌️ Edit Uploaded Images")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg", "webp"])
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
            st.warning("Please upload an image and enter an instruction.")
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
