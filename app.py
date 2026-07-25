import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("acne_vs_eczema_model.h5")


model = load_model()


# ============================================================
# APP TITLE
# ============================================================

st.title("Acne vs Eczema Classifier")

st.write(
    "Upload a clear close-up image of the affected skin area. "
    "The system will first check whether the image appears suitable "
    "for Acne/Eczema classification."
)


# ============================================================
# SETTINGS
# ============================================================

# Minimum confidence required for Acne/Eczema prediction
CONFIDENCE_THRESHOLD = 70

# Minimum percentage of skin-like pixels
SKIN_RATIO_THRESHOLD = 0.10

# Minimum colour diversity
# Helps reject images that are extremely uniform or unusual
COLOUR_VARIATION_THRESHOLD = 15


# ============================================================
# FUNCTION 1: CHECK IF IMAGE LOOKS LIKE SKIN
# ============================================================

def check_skin_image(image):

    # Resize image
    small_image = image.resize((100, 100))

    # Convert to numpy
    img = np.array(
        small_image,
        dtype=np.float32
    )

    # RGB channels
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Basic skin-like pixel detection
    skin_pixels = (
        (r > 40) &
        (g > 20) &
        (b > 10) &
        (r > g) &
        (r > b) &
        ((r - g) > 5)
    )

    # Percentage of skin-like pixels
    skin_ratio = np.mean(skin_pixels)

    # Calculate colour variation
    colour_variation = np.mean(
        np.std(img, axis=(0, 1))
    )

    # Determine whether image passes basic check
    is_skin_like = (
        skin_ratio >= SKIN_RATIO_THRESHOLD
        and
        colour_variation >= COLOUR_VARIATION_THRESHOLD
    )

    return (
        is_skin_like,
        skin_ratio,
        colour_variation
    )


# ============================================================
# FUNCTION 2: CHECK IMAGE QUALITY
# ============================================================

def check_image_quality(image):

    # Convert to numpy
    img = np.array(
        image,
        dtype=np.float32
    )

    # Calculate brightness
    brightness = np.mean(img)

    # Calculate contrast
    contrast = np.std(img)

    # Very dark image
    if brightness < 25:

        return False, "The image is too dark."

    # Very bright image
    if brightness > 245:

        return False, "The image is too bright."

    # Very low contrast
    if contrast < 15:

        return False, "The image has very low contrast."

    return True, ""


# ============================================================
# FUNCTION 3: PREDICT ACNE OR ECZEMA
# ============================================================

def predict_condition(image):

    # Resize to model input size
    img = image.resize((224, 224))

    # Convert to numpy
    img_array = np.array(
        img,
        dtype=np.float32
    ) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Model prediction
    prediction = float(
        model.predict(
            img_array,
            verbose=0
        )[0][0]
    )

    # Your class mapping:
    #
    # 0 = Acne
    # 1 = Eczema

    if prediction >= 0.5:

        label = "Eczema"
        confidence = prediction * 100

    else:

        label = "Acne"
        confidence = (1 - prediction) * 100

    return label, confidence


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        # Open image
        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # Display uploaded image
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        # ====================================================
        # STEP 1: IMAGE QUALITY CHECK
        # ====================================================

        quality_ok, quality_message = check_image_quality(
            image
        )


        if not quality_ok:

            st.error(
                "❌ Image Quality Error"
            )

            st.warning(
                f"{quality_message} "
                "Please upload a clearer image."
            )


        else:

            # =================================================
            # STEP 2: NON-SKIN IMAGE CHECK
            # =================================================

            (
                skin_like,
                skin_ratio,
                colour_variation
            ) = check_skin_image(image)


            if not skin_like:

                st.error(
                    "❌ Invalid Image"
                )

                st.warning(
                    "This image does not appear to contain "
                    "a suitable skin region."
                )

                st.info(
                    "Please upload a clear close-up image "
                    "of the affected skin area."
                )


            else:

                # =============================================
                # STEP 3: ACNE VS ECZEMA PREDICTION
                # =============================================

                label, confidence = predict_condition(
                    image
                )


                # =============================================
                # STEP 4: UNKNOWN / UNFAMILIAR IMAGE CHECK
                # =============================================

                if confidence < CONFIDENCE_THRESHOLD:

                    st.error(
                        "⚠️ Uncertain Classification"
                    )

                    st.warning(
                        "This image may not look sufficiently "
                        "similar to the Acne or Eczema images "
                        "used to train the model."
                    )

                    st.info(
                        "Please upload a clearer close-up image "
                        "showing the affected skin area."
                    )


                else:

                    # =========================================
                    # FINAL RESULT
                    # =========================================

                    st.subheader(
                        f"Prediction: {label}"
                    )

                    st.write(
                        f"Confidence: {confidence:.2f}%"
                    )


                    # =========================================
                    # DISCLAIMER
                    # =========================================

                    st.caption(
                        "This AI prediction is for educational "
                        "and research purposes only and is not "
                        "a medical diagnosis."
                    )


    except Exception as e:

        st.error(
            f"Something went wrong processing this image: {e}"
        )
