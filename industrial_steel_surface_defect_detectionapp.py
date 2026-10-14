
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# -----------------------------------
# Application configuration
# -----------------------------------
st.set_page_config(
    page_title="Steel Defect Detection",
    page_icon="🏭",
    layout="wide"
)

MODEL_PATH = (
    Path(__file__).parent /
    "resnet50_industrial_defect.keras"
)

CATEGORIES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# -----------------------------------
# Load the trained model
# -----------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# -----------------------------------
# Predict defect
# -----------------------------------
def predict_defect(image, model):
    image = image.convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(
        image, dtype=np.float32
    )

    # ResNet50 preprocessing
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array, verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    return (
        CATEGORIES[predicted_index],
        predictions
    )

# -----------------------------------
# Main interface
# -----------------------------------
st.title("🏭 Industrial Steel Defect Detection")

st.write(
    "Upload a steel-surface image to identify "
    "its defect using a trained ResNet50 model."
)

st.info(
    "Supported classes: Crazing, Inclusion, Patches, "
    "Pitted Surface, Rolled-in Scale and Scratches."
)

uploaded_file = st.file_uploader(
    "Upload a steel defect image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Prediction")

        if st.button("🔍 Detect Steel Defect", type="primary"):
            try:
                if not MODEL_PATH.exists():
                    st.error(
                        f"Model file not found: {MODEL_PATH}"
                    )
                else:
                    model = load_model()

                    defect, probabilities = predict_defect(
                        image, model
                    )

                    confidence = float(
                        probabilities[CATEGORIES.index(defect)]
                    ) * 100

                    st.success(
                        f"Predicted Defect: "
                        f"{defect.replace('_', ' ').title()}"
                    )

                    st.metric(
                        "Prediction Score",
                        f"{confidence:.2f}%"
                    )

                    st.caption(
                        "Scores are model outputs, not "
                        "guaranteed probabilities of correctness."
                    )

                    scores_df = pd.DataFrame({
                        "Defect Class": [
                            name.replace("_", " ").title()
                            for name in CATEGORIES
                        ],
                        "Score (%)": probabilities * 100
                    }).sort_values(
                        "Score (%)", ascending=False
                    )

                    st.subheader("All Class Scores")
                    st.bar_chart(
                        scores_df.set_index("Defect Class")
                    )

                    st.dataframe(
                        scores_df.style.format({
                            "Score (%)": "{:.2f}%"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

st.divider()
st.caption(
    "Industrial Defect Detection | Deep Learning | ResNet50"
)
