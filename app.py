import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Acne vs Eczema Classifier", layout="centered")

# EDIT this order if your printed class_indices differs
# Expected: {'acne': 0, 'eczema': 1, 'non_disease': 2, 'other_disease': 3}
class_names = ["Acne", "Eczema", "Non-Disease", "Other Skin Disease"]

CONFIDENCE_THRESHOLD = 0.60

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('acne_eczema_class_model (1).keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Failed to load the model file. Please ensure 'acne_eczema_class_model (1).keras' is present in the repository.\n\nDetails: {e}")
    st.stop()

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

        with st.spinner("Analyzing image..."):
            prediction = model.predict(img_array, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_label = class_names[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

        if predicted_label in ["Non-Disease", "Other Skin Disease"]:
            st.error(
                "⚠️ Invalid Image\n\n"
                "This image does not appear to be either:\n\n"
                "• Acne\n\n"
                "• Eczema\n\n"
                "Please upload a clearer skin image."
            )
        elif confidence < CONFIDENCE_THRESHOLD * 100:
            st.warning(
                "⚠️ Uncertain Result\n\n"
                "The model was unable to confidently classify this image as either Acne or Eczema. "
                "This may be a borderline or unclear case. Please try a clearer image, or consult a dermatologist for an accurate diagnosis."
            )
        else:
            st.subheader(f"Prediction: {predicted_label}")
            st.write(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Something went wrong processing this image: {e}")
