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
# LOAD MODEL
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
    "or Other."
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

    # Image too dark
    if brightness < 20:

        return False, (
            "The image is too dark. "
            "Please upload an image with better lighting."
        )

    # Image too bright
    if brightness > 250:

        return False, (
            "The image is too bright. "
            "Please upload an image with better lighting."
        )

    # Image has very low contrast
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

    # Resize image to model input size
    img = image.resize(
        (224, 224)
    )

    # Convert image to NumPy array
    img_array = np.array(
        img,
        dtype=np.float32
    )

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# MAKE PREDICTION
# ============================================================

def predict_image(image):

    # Preprocess image
    img_array = preprocess_image(
        image
    )

    # Make prediction
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    # Find class with highest probability
    predicted_index = np.argmax(
        predictions
    )

    # Get class name
    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # Get ONLY the confidence of the predicted class
    confidence = float(
        predictions[predicted_index] * 100
    )

    return (
        predicted_class,
        confidence
    )


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ====================================================
        # OPEN IMAGE
        # ====================================================

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ====================================================
        # DISPLAY IMAGE
        # ====================================================

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        # ====================================================
        # CHECK IMAGE QUALITY
        # ====================================================

        quality_ok, quality_message = (
            check_image_quality(
                image
            )
        )


        # ====================================================
        # IF IMAGE QUALITY IS BAD
        # ====================================================

        if not quality_ok:

            st.error(
                "❌ Image Quality Problem"
            )

            st.warning(
                quality_message
            )


        # ====================================================
        # IF IMAGE QUALITY IS GOOD
        # ====================================================

        else:

            # ================================================
            # MAKE PREDICTION
            # ================================================

            predicted_class, confidence = (
                predict_image(
                    image
                )
            )


            # ================================================
            # DISPLAY ONLY ONE PREDICTION AND ONE PERCENTAGE
            # ================================================

            if predicted_class == "Acne":

                st.subheader(
                    "Prediction: Acne"
                )

            elif predicted_class == "Eczema":

                st.subheader(
                    "Prediction: Eczema"
                )

            else:

                st.subheader(
                    "Prediction: Neither Acne nor Eczema"
                )


            # ================================================
            # DISPLAY ONLY THE PREDICTED CLASS CONFIDENCE
            # ================================================

            st.write(
                f"Confidence: {confidence:.2f}%"
            )


            # ================================================
            # LOW CONFIDENCE WARNING
            # ================================================

            if confidence < 60:

                st.warning(
                    "⚠️ The model is uncertain about this prediction. "
                    "Please upload a clearer image showing the affected "
                    "skin area."
                )


            # ================================================
            # MEDICAL DISCLAIMER
            # ================================================

            st.caption(
                "⚠️ This application is for educational "
                "and research purposes only. It does not "
                "provide a medical diagnosis."
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "Something went wrong while processing "
            "the image."
        )

        st.exception(
            e
            )
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
