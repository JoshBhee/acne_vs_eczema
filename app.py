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
# APP
# ==============================

st.title("Acne vs Eczema Classifier")

st.write(
    "Upload a clear skin image to check whether it is more likely "
    "to be Acne or Eczema."
)


# ==============================
# UPLOAD IMAGE
# ==============================

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


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

        raw_prediction = model.predict(
            img_array,
            verbose=0
        )

        # Show raw output for testing
        st.write("Raw model output:", raw_prediction)
        st.write("Output shape:", raw_prediction.shape)


        # ==============================
        # CLASSIFICATION
        # ==============================

        prediction = float(raw_prediction[0][0])

        st.write("Prediction value:", prediction)


        # Your class mapping:
        # 0 = Acne
        # 1 = Eczema

        if prediction >= 0.5:

            label = "Eczema"
            confidence = prediction * 100

        else:

            label = "Acne"
            confidence = (1 - prediction) * 100


        # ==============================
        # RESULT
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
                "⚠️ The model is not very confident in this prediction."
            )


    except Exception as e:

        st.error(
            f"Something went wrong processing this image: {e}"
        )
