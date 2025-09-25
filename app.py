import streamlit as st
import requests
import base64
import io
from PIL import Image

# ---------------- CONFIG ----------------
API_KEY = ""
API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-v1-5/image-to-image"

st.set_page_config(page_title="Stable Diffusion Image Editor", layout="wide")
st.title("🖼️ Stable Diffusion Image Editor")

# ---------------- UI ----------------
init_image_file = st.file_uploader("📤 Upload an image to edit", type=["png", "jpg", "jpeg"])
mask_file = st.file_uploader("📤 Upload a mask image (optional, white = editable, black = keep)", type=["png"])

prompt = st.text_area("✨ Enter your edit prompt:", height=120, placeholder="e.g., Replace background with a futuristic neon city")

strength = st.slider("🎚️ Strength (how much to modify the original image)", 0.1, 1.0, 0.75, 0.05)

num_images = st.slider("🧾 Number of images", min_value=1, max_value=4, value=1)

# ---------------- Generation flow ----------------
if st.button("🚀 Generate Edited Image"):
    if not init_image_file or not prompt.strip():
        st.warning("Please upload an image and enter a prompt!")
    else:
        with st.spinner("Editing image with Stable Diffusion..."):
            try:
                # Read images
                init_image_bytes = init_image_file.read()
                files = {
                    "init_image": init_image_bytes
                }

                if mask_file:
                    files["mask_image"] = mask_file.read()

                # Request payload
                data = {
                    "text_prompts[0][text]": prompt,
                    "cfg_scale": 7,          # prompt guidance
                    "clip_guidance_preset": "FAST_BLUE",
                    "samples": num_images,
                    "steps": 50,
                    "init_image_mode": "IMAGE_STRENGTH",
                    "image_strength": strength
                }

                headers = {
                    "Authorization": f"Bearer {API_KEY}"
                }

                # API call
                response = requests.post(API_URL, headers=headers, files=files, data=data)

                if response.status_code != 200:
                    st.error(f"Error: {response.status_code} - {response.text}")
                else:
                    results = response.json()
                    cols = st.columns(len(results["artifacts"]))
                    for i, artifact in enumerate(results["artifacts"]):
                        img_base64 = artifact["base64"]
                        img_bytes = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_bytes))

                        filename = f"edited_image_{i+1}.png"

                        with cols[i]:
                            st.image(img, caption=filename, use_column_width=True)
                            st.download_button(
                                "⬇️ Download",
                                data=img_bytes,
                                file_name=filename,
                                mime="image/png",
                                key=f"dl_{i}"
                            )

            except Exception as e:
                st.error(f"⚠️ Error editing image: {e}")
