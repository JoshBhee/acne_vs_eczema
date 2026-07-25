import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Acne vs Eczema Classifier",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# LOAD YOUR ACNE / ECZEMA MODEL
# ============================================================

@st.cache_resource
def load_acne_eczema_model():

    return tf.keras.models.load_model(
        "acne_vs_eczema_model.h5"
    )


acne_eczema_model = load_acne_eczema_model()


# ============================================================
# LOAD GENERAL IMAGE MODEL
# ============================================================

@st.cache_resource
def load_general_model():

    # MobileNetV2 is used only as a general image filter.
    # It is NOT the Acne/Eczema model.

    model = tf.keras.applications.MobileNetV2(
        weights="imagenet"
    )

    return model


general_model = load_general_model()


# ============================================================
# APP TITLE
# ============================================================

st.title("🩺 Acne vs Eczema Classifier")

st.write(
    "Upload a clear close-up image of the affected skin area."
)

st.write(
    "The system first checks whether the uploaded image "
    "appears suitable for skin-condition classification."
)

st.info(
    "Images that appear to contain animals, plants, objects, "
    "or other unsuitable content may be rejected before "
    "Acne/Eczema classification."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# FUNCTION 1
# IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image):

    """
    Checks whether the image is extremely dark,
    extremely bright, or has extremely low contrast.
    """

    img = np.array(
        image,
        dtype=np.float32
    )

    # Average brightness
    brightness = np.mean(img)

    # Image contrast
    contrast = np.std(img)


    # Very dark
    if brightness < 25:

        return False, (
            "The image is too dark."
        )


    # Very bright
    if brightness > 245:

        return False, (
            "The image is too bright."
        )


    # Very low contrast
    if contrast < 15:

        return False, (
            "The image has very low contrast."
        )


    return True, ""


# ============================================================
# FUNCTION 2
# GENERAL IMAGE CLASSIFICATION
# ============================================================

def get_general_predictions(image):

    """
    Uses MobileNetV2 to identify the general content
    of the uploaded image.

    This is NOT a medical diagnosis.
    """

    # Resize
    img = image.resize(
        (224, 224)
    )

    # Convert to numpy
    img_array = np.array(
        img,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing
    img_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(
            img_array
        )
    )

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Predict
    predictions = general_model.predict(
        img_array,
        verbose=0
    )

    # Get top 10 predictions
    decoded = (
        tf.keras.applications.mobilenet_v2
        .decode_predictions(
            predictions,
            top=10
        )[0]
    )

    return decoded


# ============================================================
# FUNCTION 3
# CHECK FOR ANIMALS, PLANTS AND OBJECTS
# ============================================================

def check_for_invalid_content(decoded):

    """
    Attempts to reject images containing animals,
    plants, or obvious objects.

    This is a general image filter.
    """

    # ========================================================
    # ANIMAL KEYWORDS
    # ========================================================

    animal_words = [

        # General animal categories
        "animal",
        "mammal",
        "reptile",
        "amphibian",
        "bird",
        "fish",
        "insect",
        "arachnid",

        # Mammals
        "dog",
        "cat",
        "wolf",
        "fox",
        "lion",
        "tiger",
        "leopard",
        "cheetah",
        "bear",
        "panda",
        "elephant",
        "giraffe",
        "zebra",
        "horse",
        "donkey",
        "cow",
        "bull",
        "sheep",
        "goat",
        "pig",
        "rabbit",
        "hare",
        "mouse",
        "rat",
        "hamster",
        "squirrel",
        "monkey",
        "gorilla",
        "chimpanzee",
        "baboon",
        "kangaroo",
        "koala",
        "deer",
        "camel",
        "llama",
        "hippopotamus",
        "rhinoceros",

        # Birds
        "eagle",
        "hawk",
        "owl",
        "parrot",
        "pigeon",
        "crow",
        "raven",
        "sparrow",
        "robin",
        "chicken",
        "hen",
        "rooster",
        "turkey",
        "duck",
        "goose",
        "swan",
        "penguin",
        "flamingo",
        "peacock",
        "ostrich",

        # Reptiles
        "snake",
        "lizard",
        "turtle",
        "tortoise",
        "crocodile",
        "alligator",
        "iguana",
        "chameleon",

        # Amphibians
        "frog",
        "toad",
        "salamander",

        # Fish
        "fish",
        "shark",
        "ray",
        "stingray",
        "eel",
        "goldfish",

        # Insects
        "butterfly",
        "moth",
        "bee",
        "wasp",
        "ant",
        "beetle",
        "grasshopper",
        "cricket",
        "fly",
        "mosquito",
        "dragonfly",

        # Arachnids
        "spider",
        "scorpion",
        "tick"
    ]


    # ========================================================
    # PLANT KEYWORDS
    # ========================================================

    plant_words = [

        "plant",
        "tree",
        "flower",
        "leaf",
        "mushroom",
        "pot",
        "cactus",
        "fern",
        "vine"
    ]


    # ========================================================
    # OBJECT KEYWORDS
    # ========================================================

    object_words = [

        "car",
        "vehicle",
        "bicycle",
        "motorcycle",
        "airplane",
        "boat",

        "chair",
        "table",
        "computer",
        "laptop",
        "keyboard",
        "phone",
        "camera",

        "book",
        "bottle",
        "cup",
        "shoe",
        "bag",
        "backpack",
        "hat",

        "clock",
        "television",
        "guitar",
        "ball",
        "toy",

        "keyboard",
        "microphone",
        "speaker",
        "remote",
        "television",
        "screen"
    ]


    # ========================================================
    # CHECK TOP PREDICTIONS
    # ========================================================

    for _, label, probability in decoded:

        label = label.lower()

        probability = float(
            probability
        )


        # ====================================================
        # CHECK ANIMALS
        # ====================================================

        for animal in animal_words:

            if animal in label:

                if probability >= 0.10:

                    return False, (
                        "animal"
                    )


        # ====================================================
        # CHECK PLANTS
        # ====================================================

        for plant in plant_words:

            if plant in label:

                if probability >= 0.10:

                    return False, (
                        "plant"
                    )


        # ====================================================
        # CHECK OBJECTS
        # ====================================================

        for obj in object_words:

            if obj in label:

                if probability >= 0.10:

                    return False, (
                        "object"
                    )


    # ========================================================
    # NO OBVIOUS INVALID CONTENT FOUND
    # ========================================================

    return True, ""


# ============================================================
# FUNCTION 4
# CHECK FOR HUMAN-RELATED CONTENT
# ============================================================

def check_for_human(decoded):

    """
    Checks whether MobileNetV2 detected a human-related
    concept among its predictions.

    This is only a supporting filter.
    """

    human_words = [

        "person",
        "man",
        "woman",
        "boy",
        "girl",
        "face",
        "head",
        "hand",
        "arm",
        "leg",
        "body",
        "portrait"
    ]


    for _, label, probability in decoded:

        label = label.lower()

        probability = float(
            probability
        )


        for human_word in human_words:

            if human_word in label:

                if probability >= 0.05:

                    return True


    return False


# ============================================================
# FUNCTION 5
# BASIC SKIN PIXEL CHECK
# ============================================================

def check_skin_pixels(image):

    """
    Performs a basic skin-colour check.

    This is NOT a medical diagnosis.
    """

    # Resize
    small_image = image.resize(
        (100, 100)
    )

    # Convert to numpy
    img = np.array(
        small_image,
        dtype=np.float32
    )


    # RGB channels
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]


    # Basic skin-like pixels
    skin_pixels = (

        (r > 40) &

        (g > 20) &

        (b > 10) &

        (r > g) &

        (r > b) &

        ((r - g) > 5)
    )


    # Calculate percentage
    skin_ratio = np.mean(
        skin_pixels
    )


    return skin_ratio


# ============================================================
# FUNCTION 6
# ACNE / ECZEMA MODEL
# ============================================================

def predict_acne_eczema(image):

    """
    Uses the existing Acne/Eczema model.

    Class mapping:

    0 = Acne
    1 = Eczema
    """

    # Resize to model input
    img = image.resize(
        (224, 224)
    )


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


    # Predict
    prediction = float(
        acne_eczema_model.predict(
            img_array,
            verbose=0
        )[0][0]
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if prediction >= 0.5:

        label = "Eczema"

        confidence = (
            prediction * 100
        )

    else:

        label = "Acne"

        confidence = (
            (1 - prediction) * 100
        )


    return label, confidence


# ============================================================
# MAIN APP
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
        # STEP 1
        # IMAGE QUALITY CHECK
        # ====================================================

        quality_ok, quality_message = (
            check_image_quality(
                image
            )
        )


        if not quality_ok:

            st.error(
                "❌ Image Quality Error"
            )

            st.warning(
                quality_message
            )

            st.info(
                "Please upload a clearer image "
                "with good lighting."
            )


        else:

            # =================================================
            # STEP 2
            # GENERAL IMAGE MODEL
            # =================================================

            decoded = (
                get_general_predictions(
                    image
                )
            )


            # =================================================
            # STEP 3
            # CHECK ANIMAL / PLANT / OBJECT
            # =================================================

            valid_content, content_type = (
                check_for_invalid_content(
                    decoded
                )
            )


            if not valid_content:

                st.error(
                    "❌ Invalid Image"
                )


                if content_type == "animal":

                    st.warning(
                        "The uploaded image appears "
                        "to contain an animal."
                    )


                elif content_type == "plant":

                    st.warning(
                        "The uploaded image appears "
                        "to contain a plant or vegetation."
                    )


                elif content_type == "object":

                    st.warning(
                        "The uploaded image appears "
                        "to contain an object rather "
                        "than a suitable skin image."
                    )


                st.info(
                    "Please upload a clear close-up "
                    "image of human skin."
                )


            else:

                # =================================================
                # STEP 4
                # CHECK HUMAN-RELATED CONTENT
                # =================================================

                human_detected = (
                    check_for_human(
                        decoded
                    )
                )


                # =================================================
                # STEP 5
                # SKIN PIXEL CHECK
                # =================================================

                skin_ratio = (
                    check_skin_pixels(
                        image
                    )
                )


                # =================================================
                # NOTE:
                # We do NOT immediately reject based only
                # on human_detected because MobileNetV2
                # may classify a close-up skin image poorly.
                #
                # The skin ratio is therefore used as
                # supporting information.
                # =================================================


                if (
                    not human_detected
                    and
                    skin_ratio < 0.05
                ):

                    st.error(
                        "❌ Invalid Image"
                    )

                    st.warning(
                        "The image does not appear "
                        "to contain a suitable human "
                        "skin region."
                    )

                    st.info(
                        "Please upload a clear close-up "
                        "image of the affected skin area."
                    )


                else:

                    # =============================================
                    # STEP 6
                    # ACNE / ECZEMA PREDICTION
                    # =============================================

                    label, confidence = (
                        predict_acne_eczema(
                            image
                        )
                    )


                    # =============================================
                    # STEP 7
                    # DISPLAY RESULT
                    # =============================================

                    st.subheader(
                        f"Prediction: {label}"
                    )


                    st.write(
                        f"Confidence: "
                        f"{confidence:.2f}%"
                    )


                    # =============================================
                    # STEP 8
                    # LOW CONFIDENCE WARNING
                    # =============================================

                    if confidence < 70:

                        st.warning(
                            "⚠️ The model is uncertain "
                            "about this prediction."
                        )

                        st.info(
                            "The uploaded image may not "
                            "closely resemble the Acne "
                            "or Eczema images used to "
                            "train the model."
                        )


                    # =============================================
                    # DISCLAIMER
                    # =============================================

                    st.caption(
                        "This application is intended "
                        "for educational and research "
                        "purposes only. It does not "
                        "provide a medical diagnosis."
                    )


    except Exception as e:

        st.error(
            "An error occurred while processing "
            "the image."
        )

        st.exception(
            e
        )
