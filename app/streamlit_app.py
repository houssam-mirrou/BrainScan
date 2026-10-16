"""Run with: streamlit run app/streamlit_app.py."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import streamlit as st
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.brainscan import CLASS_NAMES, preprocess_image

MODEL_PATH = PROJECT_ROOT / "models/experiments/v3_scheduler/best_cnn.keras"
CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary",
}


@st.cache_resource
def load_model(path: str, modified_ns: int):
    """Reuse the checkpoint until the file changes."""
    model = tf.keras.models.load_model(path, compile=False)
    if model.input_shape != (None, 224, 224, 1):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    if model.output_shape != (None, len(CLASS_NAMES)):
        raise ValueError(f"Unexpected model output shape: {model.output_shape}")
    return model


def main():
    st.set_page_config(page_title="BrainScan", page_icon="🧠", layout="centered")
    st.title("BrainScan")
    st.write("Upload a brain MRI image to classify it with the V3 scheduler model.")
    st.caption("Research demo. Predictions are not a medical diagnosis.")

    with st.sidebar:
        st.header("Model")
        st.write("V3 scheduler · best_cnn.keras")
        st.caption("Input: 224 × 224 grayscale, padded and normalized to [0, 1].")
        report = MODEL_PATH.parent / "test_report.txt"
        if report.is_file():
            with st.expander("Saved evaluation report"):
                st.text(report.read_text())

    if not MODEL_PATH.is_file():
        st.error(f"Model checkpoint not found: {MODEL_PATH}")
        return

    uploaded = st.file_uploader("MRI image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if uploaded is None:
        st.info("Choose an image to get started.")
        return

    try:
        # Use the exact same OpenCV decoder and preprocessing as training.
        with TemporaryDirectory(prefix="brainscan-") as directory:
            image_path = Path(directory) / ("upload" + Path(uploaded.name).suffix.lower())
            image_path.write_bytes(uploaded.getvalue())
            image = preprocess_image(image_path)
    except (ValueError, OSError) as exc:
        st.error(f"Could not process this image: {exc}")
        return

    st.image(image.squeeze(), caption="Model input after preprocessing", clamp=True, width=300)
    if not st.button("Classify image", type="primary"):
        return

    try:
        with st.spinner("Classifying image…"):
            model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
            probabilities = np.asarray(model(image[np.newaxis, ...], training=False))[0]
            if probabilities.shape != (len(CLASS_NAMES),) or not np.isfinite(probabilities).all():
                raise ValueError("The model returned invalid class scores.")
    except Exception as exc:
        st.error(f"Could not run the model: {exc}")
        return

    predicted_id = int(np.argmax(probabilities))
    st.subheader(f"Prediction: {CLASS_LABELS[CLASS_NAMES[predicted_id]]}")
    st.metric("Model confidence", f"{probabilities[predicted_id]:.1%}")
    st.caption("Confidence is the model's class score, not a measure of diagnostic certainty.")
    st.dataframe(
        [
            {"Class": CLASS_LABELS[name], "Probability": float(score)}
            for name, score in zip(CLASS_NAMES, probabilities)
        ],
        column_config={"Probability": st.column_config.ProgressColumn(
            "Probability", min_value=0.0, max_value=1.0, format="percent"
        )},
        hide_index=True,
        width="stretch",
    )


if __name__ == "__main__":
    main()
