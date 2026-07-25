import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('acne_vs_eczema_model.h5')

model = load_model()

st.title("Acne vs Eczema Classifier")
st.write("Upload a clear skin image to check for Acne or Eczema.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = float(model.predict(img_array, verbose=0)[0][0])
        # class_indices confirmed from training: {'acne': 0, 'eczema': 1}
        confidence_gap = abs(prediction - 0.5)

        if confidence_gap < 0.20:
            st.error(
                "⚠️ This does not appear to be Acne or Eczema.\n\n"
                "Please upload a clearer skin image showing one of these conditions."
            )
        else:
            if prediction > 0.5:
                label = "Eczema"
                confidence = prediction * 100
            else:
                label = "Acne"
                confidence = (1 - prediction) * 100

            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Something went wrong processing this image: {e}")