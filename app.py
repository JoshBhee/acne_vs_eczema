import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Acne vs Eczema Classifier",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# LOAD 3-CLASS MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "acne_eczema_other_model.h5"
    )


model = load_model()


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Acne",
    "Eczema",
    "Other"
]


# ============================================================
# APP TITLE
# ============================================================

st.title("🩺 Acne vs Eczema Classifier")

st.write(
    "Upload a clear close-up image of the affected skin area."
)

st.info(
    "The model classifies images into Acne, Eczema, "
    "or Other. Images that do not appear suitable "
    "for skin-condition classification may be rejected."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image):

    img = np.array(
        image,
        dtype=np.float32
    )

    brightness = np.mean(img)

    contrast = np.std(img)

    if brightness < 20:

        return False, (
            "The image is too dark. "
            "Please upload an image with better lighting."
        )

    if brightness > 250:

        return False, (
            "The image is too bright. "
            "Please upload an image with better lighting."
        )

    if contrast < 10:

        return False, (
            "The image has very low contrast. "
            "Please upload a clearer image."
        )

    return True, ""


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    img = image.resize(
        (224, 224)
    )

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

# ==============================
# MAKE PREDICTION
# ==============================

prediction = model.predict(
    img_array,
    verbose=0
)

# Get the predicted class
predicted_class = np.argmax(prediction[0])

# Get confidence
confidence = float(
    prediction[0][predicted_class] * 100
)

# ==============================
# CLASS NAMES
# ==============================

class_names = [
    "Acne",
    "Eczema",
    "Other"
]

label = class_names[predicted_class]

# ==============================
# DISPLAY RESULT
# ==============================

st.subheader(
    f"Prediction: {label}"
)

st.write(
    f"Confidence: {confidence:.2f}%"
)

# ==============================
# LOW CONFIDENCE WARNING
# ==============================

if confidence < 60:
    st.warning(
        "⚠️ The model is not very confident in this prediction. "
        "Please upload a clearer image showing the affected skin area."
    )

# ==============================
# MEDICAL DISCLAIMER
# ==============================

st.info(
    "This AI prediction is for educational and research purposes only "
    "and is not a medical diagnosis."
)
