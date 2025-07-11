import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Concatenate

# ---------- U-Net Model ----------
def unet_model(input_size=(512, 512, 3)):
    inputs = Input(shape=input_size)

    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    u1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c4)
    u1 = Concatenate()([u1, c3])
    u1 = Dropout(0.2)(u1)
    u1 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)

    u2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u1)
    u2 = Concatenate()([u2, c2])
    u2 = Dropout(0.2)(u2)
    u2 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)

    u3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(u2)
    u3 = Concatenate()([u3, c1])
    u3 = Dropout(0.1)(u3)
    u3 = Conv2D(32, (3, 3), activation='relu', padding='same')(u3)

    output = Conv2D(1, (1, 1), activation='sigmoid')(u3)
    return Model(inputs, output)

@st.cache_resource
def load_model_weights():
    model = unet_model()
    model.load_weights("unet_model.h5")
    return model

def compute_mask_area(mask):
    return np.sum(mask)

def get_average_color(img, mask):
    if np.sum(mask) == 0:
        return np.mean(img.reshape(-1, 3), axis=0)
    return np.mean(img[mask == 1], axis=0)

def are_images_from_same_location(img1, img2, min_matches=20):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return False, 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good) >= min_matches, len(good)

def convert_img_to_bytes(img_array):
    img_pil = Image.fromarray(img_array.astype('uint8'))
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Flood Detection", layout="centered")

if 'show_result' not in st.session_state:
    st.session_state.show_result = False

# Background and style for upload screen
def get_base64_of_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if not st.session_state.show_result:
    bg_image_base64 = get_base64_of_image("background.png")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .main-text-container {{
            padding: 20px;
            border-radius: 12px;
        }}
        .main-text-container h1,
        .main-text-container h2,
        .main-text-container h3,
        .main-text-container label,
        .main-text-container p {{
            color: white !important;
        }}
        .stFileUploader {{
            background-color: rgba(0, 0, 0, 0.6) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

model = load_model_weights()

# Sidebar with threshold setting
st.sidebar.markdown("### ⚙️ Detection Settings")
threshold = st.sidebar.slider("Flood Detection Threshold", 0.3, 0.9, 0.5, step=0.05)

st.markdown('<div class="main-text-container">', unsafe_allow_html=True)

if not st.session_state.show_result:
    st.title("🌊 AI-Based Flood Detection")
    st.subheader("📥 Upload Before and After Images")

    col1, col2 = st.columns(2)
    with col1:
        before_upload = st.file_uploader("Upload BEFORE image", type=["jpg", "png", "jpeg"], key="before")
    with col2:
        after_upload = st.file_uploader("Upload AFTER image", type=["jpg", "png", "jpeg"], key="after")

    if before_upload and after_upload:
        before_img = Image.open(before_upload).convert("RGB").resize((512, 512))
        after_img = Image.open(after_upload).convert("RGB").resize((512, 512))

        st.markdown("### 🖼️ Before Image")
        st.image(before_img, use_container_width=True)

        st.markdown("### 🖼️ After Image")
        st.image(after_img, use_container_width=True)

        if st.button("➡️ Analyze Flood Impact"):
            st.session_state.before_img = before_img
            st.session_state.after_img = after_img
            st.session_state.show_result = True
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Result Page ----------
if st.session_state.show_result:
    before_img = st.session_state.before_img
    after_img = st.session_state.after_img

    before_np = np.array(before_img)
    after_np = np.array(after_img)

    same_location, matches = are_images_from_same_location(before_np, after_np)

    before_input = np.expand_dims(before_np / 255.0, axis=0)
    after_input = np.expand_dims(after_np / 255.0, axis=0)

    before_mask = (model.predict(before_input)[0, :, :, 0] > threshold).astype(np.uint8)
    after_mask = (model.predict(after_input)[0, :, :, 0] > threshold).astype(np.uint8)

    flood_area_diff = np.sum(after_mask) - np.sum(before_mask)
    percent_change = flood_area_diff / (512 * 512)

    avg_color_before = get_average_color(before_np, before_mask)
    avg_color_after = get_average_color(after_np, after_mask)
    color_shift = np.linalg.norm(avg_color_after - avg_color_before)

    st.subheader("📍 Location Match")
    if same_location:
        st.success(f"✅ Images from same spot ({matches} matching features)")
    else:
        st.warning(f"⚠️ Possibly different location ({matches} features only)")

    st.subheader("📊 Flood Detection Result")
    if percent_change > 0.05:
        st.success(f"🌊 Flood Detected! (+{percent_change*100:.2f}% water increase)")
    else:
        st.info("✅ No significant flood detected.")

    if color_shift > 25:
        st.warning("🟡 Water color changed — possibly muddy flood water.")

    st.subheader("🎯 Flood Mask Visuals")
    col1, col2 = st.columns(2)
    col1.markdown("**Before Mask**")
    col1.image(before_mask * 255, width=300)

    col2.markdown("**After Mask**")
    col2.image(after_mask * 255, width=300)

    st.subheader("🖼️ Mask Overlay")
    overlay_image = after_np.copy()
    red_mask = (after_mask == 1)
    overlay_image[red_mask] = [255, 0, 0]
    st.image(overlay_image, caption="Overlay: Red = Flooded Area", use_container_width=True)

    st.button("🔁 Start Over", on_click=lambda: st.session_state.update({"show_result": False}))