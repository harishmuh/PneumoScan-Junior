"""
=============================================================
app.py

Project:
Pediatric Pneumonia Detection using Xception CNN

Application:
PneumoScan Junior

Developer:
Harish
=============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import streamlit as st
from PIL import Image

from config import (
    APP_NAME,
    PAGE_ICON,
    LAYOUT,
    SIDEBAR_STATE,
    SUPPORTED_IMAGE_TYPES,
)

from styles import apply_custom_css

from samples import (
    get_sample_names,
    get_sample,
)

from image_utils import preprocess_image

from model_utils import predict

from explainability import (
    generate_gradcam_visualization,
)

from ui import (
    render_header,
    render_sidebar,
    render_prediction_card,
    render_probability_card,
    render_sample_information,
    render_explainability,
    render_disclaimer,
    render_footer,
    show_error,
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title=APP_NAME,

    page_icon=PAGE_ICON,

    layout=LAYOUT,

    initial_sidebar_state=SIDEBAR_STATE,

)

# ==========================================================
# GLOBAL CSS
# ==========================================================

apply_custom_css()

# ==========================================================
# SIDEBAR
# ==========================================================

render_sidebar()

# ==========================================================
# HEADER
# ==========================================================

render_header()

# ==========================================================
# SESSION STATE
# ==========================================================

if "selected_image" not in st.session_state:

    st.session_state.selected_image = None

if "selected_sample" not in st.session_state:

    st.session_state.selected_sample = None

# ==========================================================
# SAMPLE IMAGES
# ==========================================================

st.markdown("## 📂 Sample Images")

sample_names = get_sample_names()

selected_name = st.selectbox(

    "Choose one of the built-in pediatric chest X-ray images",

    sample_names,

)

col1, col2 = st.columns([1, 1])

with col1:

    if st.button(

        "Load Selected Sample",

        use_container_width=True,

    ):

        sample = get_sample(selected_name)

        st.session_state.selected_sample = sample

        st.session_state.selected_image = Image.open(

            sample["path"]

        ).convert("RGB")

with col2:

    if st.button(

        "Clear Selection",

        use_container_width=True,

    ):

        st.session_state.selected_sample = None

        st.session_state.selected_image = None

# ==========================================================
# FILE UPLOADER
# ==========================================================

st.divider()

st.markdown("## 📤 Upload Chest X-ray")

uploaded_file = st.file_uploader(

    "Upload a pediatric chest X-ray",

    type=SUPPORTED_IMAGE_TYPES,

)

if uploaded_file is not None:

    st.session_state.selected_image = Image.open(

        uploaded_file

    ).convert("RGB")

    st.session_state.selected_sample = None

# ==========================================================
# WAIT FOR IMAGE
# ==========================================================

if st.session_state.selected_image is None:

    st.info(

        """
Please select one of the built-in sample images

or

Upload your own pediatric chest X-ray image.
"""

    )

    render_footer()

    st.stop()

# ==========================================================
# IMAGE PREVIEW
# ==========================================================

st.divider()

left_column, right_column = st.columns(

    [1.2, 1]

)

with left_column:

    st.markdown("## 🖼 Selected Image")

    st.image(

        st.session_state.selected_image,

        use_container_width=True,

    )

    if st.session_state.selected_sample is not None:

        render_sample_information(

            st.session_state.selected_sample

        )

with right_column:

    st.markdown("## 🤖 AI Prediction")

    run_prediction = st.button(

        "Run Prediction",

        type="primary",

        use_container_width=True,

    )

# ==========================================================
# PREDICTION PIPELINE
# ==========================================================

if run_prediction:

    try:

        # --------------------------------------------------
        # PREPROCESS IMAGE
        # --------------------------------------------------

        with st.spinner("Preprocessing image..."):

            preprocessed_image = preprocess_image(
                st.session_state.selected_image
            )

        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        with st.spinner("Running model prediction..."):

            prediction_result = predict(
                preprocessed_image
            )

        # --------------------------------------------------
        # GRAD-CAM
        # --------------------------------------------------

        with st.spinner("Generating Grad-CAM..."):

            gradcam_result = generate_gradcam_visualization(
                original_image=st.session_state.selected_image,
                preprocessed_image=preprocessed_image,
            )

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        st.divider()

        st.markdown("# 🩺 Prediction Results")

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:

            render_prediction_card(
                prediction_result
            )

        with result_col2:

            render_probability_card(
                prediction_result
            )

        # --------------------------------------------------
        # EXPLAINABILITY
        # --------------------------------------------------

        render_explainability(
            original_image=st.session_state.selected_image,
            gradcam_results=gradcam_result,
        )

        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        render_disclaimer()

        # --------------------------------------------------
        # FOOTER
        # --------------------------------------------------

        render_footer()

    except Exception as error:

        show_error(
            "Prediction failed."
        )

        st.exception(error)

# ==========================================================
# INITIAL PAGE
# ==========================================================

else:

    st.divider()

    st.info(
        """
### Welcome to PneumoScan Junior

This application predicts **pediatric pneumonia**
from chest X-ray images using an Xception CNN model.

Workflow

1. Select one of the built-in sample images

or

Upload your own pediatric chest X-ray.

2. Click **Run Prediction**

The application will display

• Prediction

• Model confidence

• Class probabilities

• Grad-CAM attention map

• Overlay visualization

The Grad-CAM visualization highlights image regions
that contributed most strongly to the model prediction.

It does **not** represent a confirmed localization
of disease.
"""
    )

    render_disclaimer()

    render_footer()