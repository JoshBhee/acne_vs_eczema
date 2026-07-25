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


# ============================================================
# MAKE PREDICTION
# ============================================================

def predict_image(image):

    img_array = preprocess_image(
        image
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index] * 100
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# DISPLAY RESULTS
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


        if not quality_ok:

            st.error(
                "❌ Image Quality Problem"
            )

            st.warning(
                quality_message
            )


        else:

            # ================================================
            # PREDICT
            # ================================================

            predicted_class, confidence, predictions = (
                predict_image(
                    image
                )
            )


            # ================================================
            # LOW CONFIDENCE = POSSIBLY UNFAMILIAR IMAGE
            # ================================================

            if confidence < 60:

                st.warning(
                    "⚠️ The model is uncertain about this image."
                )

                st.write(
                    "The image may not closely resemble "
                    "the images used during training."
                )


            # ================================================
            # OTHER CLASS
            # ================================================

            if predicted_class == "Other":

                st.subheader(
                    "Prediction: Neither Acne nor Eczema"
                )

                st.write(
                    f"Confidence: {confidence:.2f}%"
                )

                st.info(
                    "The model classified this image as "
                    "Other. It does not appear sufficiently "
                    "similar to the Acne or Eczema classes "
                    "used during training."
                )


            # ================================================
            # ACNE
            # ================================================

            elif predicted_class == "Acne":

                st.subheader(
                    "Prediction: Acne"
                )

                st.write(
                    f"Confidence: {confidence:.2f}%"
                )


            # ================================================
            # ECZEMA
            # ================================================

            elif predicted_class == "Eczema":

                st.subheader(
                    "Prediction: Eczema"
                )

                st.write(
                    f"Confidence: {confidence:.2f}%"
                )


            # ================================================
            # SHOW CLASS PROBABILITIES
            # ================================================

            with st.expander(
                "View model confidence details"
            ):

                st.write(
                    f"Acne: "
                    f"{predictions[0] * 100:.2f}%"
                )

                st.write(
                    f"Eczema: "
                    f"{predictions[1] * 100:.2f}%"
                )

                st.write(
                    f"Other: "
                    f"{predictions[2] * 100:.2f}%"
                )


            # ================================================
            # DISCLAIMER
            # ================================================

            st.caption(
                "⚠️ This application is for educational "
                "and research purposes only. It does not "
                "provide a medical diagnosis."
            )


    except Exception as e:

        st.error(
            "Something went wrong while processing "
            "the image."
        )

        st.exception(
            e
        )
