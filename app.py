import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


st.set_page_config(
    page_title="Acne vs Eczema Classifier",
    page_icon="🩺"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "acne_eczema_other_model.h5"
    )


model = load_model()


CLASS_NAMES = [
    "Acne",
    "Eczema",
    "Other"
]


st.title("🩺 Acne vs Eczema Classifier")

st.write(
    "Upload a clear close-up image of the affected skin area."
)


uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    resized_image = image.resize(
        (224, 224)
    )

    image_data = np.array(
        resized_image,
        dtype=np.float32
    )

    image_data = image_data / 255.0

    image_data = np.expand_dims(
        image_data,
        axis=0
    )

    prediction = model.predict(
        image_data,
        verbose=0
    )

    predicted_index = int(
        np.argmax(
            prediction[0]
        )
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        prediction[0][predicted_index] * 100
    )

    if predicted_class == "Acne":

        result_text = "Prediction: Acne"

    elif predicted_class == "Eczema":

        result_text = "Prediction: Eczema"

    else:

        result_text = "Prediction: Neither Acne nor Eczema"

    st.subheader(
        result_text
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )

    if confidence < 60:

        st.warning(
            "The model is not very confident about this prediction. "
            "Please upload a clearer image showing the affected "
            "skin area."
        )

    st.info(
        "This AI prediction is for educational and research "
        "purposes only. It is not a medical diagnosis."
    )
