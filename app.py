import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("acne_vs_eczema_model.h5")


model = load_model()


# ==============================
# APP TITLE
# ==============================

st.title("Acne vs Eczema Classifier")

st.write(
    "Upload a clear image of the affected skin area "
    "to check whether it is more likely to be Acne or Eczema."
)


# ==============================
# UPLOAD IMAGE
# ==============================

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


# ==============================
# IMAGE VALIDATION
# ==============================

def looks_like_skin(image):
    """
    Basic skin-image validation.
    This is NOT a medical diagnosis.
    It is only used to reject obvious non-skin images.
    """

    # Resize for analysis
    small_image = image.resize((100, 100))

    # Convert to numpy
    img = np.array(small_image).astype(np.float32)

    # Separate RGB channels
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Basic skin-colour pixel detection
    skin_pixels = (
        (r > 60) &
        (g > 40) &
        (b > 20) &
        (r > g) &
        (r > b) &
        ((r - g) > 10)
    )

    # Calculate percentage of pixels that look skin-like
    skin_ratio = np.mean(skin_pixels)

    return skin_ratio > 0.10


# ==============================
# PROCESS IMAGE
# ==============================

if uploaded_file is not None:

    try:

        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        # ==============================
        # VALIDATE IMAGE
        # ==============================

        valid_skin_image = looks_like_skin(image)


        if not valid_skin_image:

            st.warning(
                "⚠️ This image does not appear to contain a clear "
                "skin region."
            )

            st.info(
                "Please upload a clear close-up image of the affected "
                "skin area."
            )


        else:

            # ==============================
            # PREPROCESS IMAGE
            # ==============================

            img = image.resize((224, 224))

            img_array = np.array(
                img,
                dtype=np.float32
            ) / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )


            # ==============================
            # MODEL PREDICTION
            # ==============================

            prediction = float(
                model.predict(
                    img_array,
                    verbose=0
                )[0][0]
            )


            # ==============================
            # CLASSIFICATION
            # ==============================

            # Class mapping:
            # 0 = Acne
            # 1 = Eczema

            if prediction >= 0.5:

                label = "Eczema"
                confidence = prediction * 100

            else:

                label = "Acne"
                confidence = (1 - prediction) * 100


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
            # LOW CONFIDENCE
            # ==============================

            if confidence < 60:

                st.warning(
                    "⚠️ The model is not very confident in this "
                    "prediction. Please upload a clearer image."
                )


    except Exception as e:

        st.error(
            f"Something went wrong processing this image: {e}"
        )
