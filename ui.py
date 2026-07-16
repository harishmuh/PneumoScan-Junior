"""
=============================================================
ui.py

Project:
Pediatric Pneumonia Detection using Xception CNN

Application:
PneumoScan Junior

Description:
Reusable Streamlit UI components.

Developer:
Harish
=============================================================
"""

import streamlit as st

from config import (
    APP_NAME,
    PROJECT_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
    DEVELOPER,
    MODEL_NAME,
    FRAMEWORK,
    FRAMEWORK_VERSION,
    DATASET_NAME,
    DISCLAIMER,
    FOOTER,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
)


# ==========================================================
# HEADER
# ==========================================================

def render_header():
    """
    Render application header.
    """

    st.markdown(
        f"""
        <div class="main-title">
            {APP_NAME}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="project-title">
            {PROJECT_TITLE}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="subtitle">
            {APP_SUBTITLE}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

def render_sidebar():
    """
    Render application sidebar.
    """

    with st.sidebar:

        st.markdown("## 🫁 PneumoScan Junior")

        st.caption(f"Version {APP_VERSION}")

        st.divider()

        st.markdown("### 🧠 Model")

        st.write(f"**Architecture**")
        st.write(MODEL_NAME)

        st.write(f"**Framework**")
        st.write(f"{FRAMEWORK} {FRAMEWORK_VERSION}")

        st.divider()

        st.markdown("### 📚 Dataset")

        st.write(DATASET_NAME)

        st.divider()

        st.markdown("### 👨‍💻 Developer")

        st.write(DEVELOPER)

        st.divider()

        st.info(
            """
This application is intended for

• Research

• Education

• AI Demonstration

It is **NOT** intended to replace
professional medical judgement.
"""
        )


# ==========================================================
# SAMPLE INFORMATION
# ==========================================================

def render_sample_information(sample):
    """
    Display information about the selected
    sample image.
    """

    st.markdown("### 📖 Sample Information")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Reference Diagnosis**")

        st.write(
            sample["reference_diagnosis"]
        )

    with col2:

        st.markdown("**Dataset**")

        st.write(
            sample["dataset"]
        )

    st.info(
        sample["description"]
    )


# ==========================================================
# PREDICTION CARD
# ==========================================================

def render_prediction_card(result):
    """
    Display prediction card.
    """

    predicted_class = result["predicted_class"]

    confidence = (
        result["confidence"] * 100
    )

    if predicted_class == NEGATIVE_CLASS:

        emoji = "🟢"

        card_color = "#2E8B57"

    else:

        emoji = "🔴"

        card_color = "#C0392B"

    st.markdown(
        f"""
<div style="
background:#F8F9FA;
padding:20px;
border-radius:12px;
border-left:8px solid {card_color};
box-shadow:0px 1px 6px rgba(0,0,0,0.08);
">

<h3 style="margin-bottom:10px;">
Prediction
</h3>

<h2 style="
color:{card_color};
margin-top:0px;
margin-bottom:20px;
">
{emoji} {predicted_class}
</h2>

<b>Model Confidence</b>

<h2 style="margin-top:5px;">
{confidence:.2f}%
</h2>

</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# PROBABILITY CARD
# ==========================================================

def render_probability_card(result):
    """
    Display prediction probabilities.
    """

    st.markdown(
        "### 📊 Prediction Probability"
    )

    normal_prob = (
        result["normal_probability"]
    )

    pneumonia_prob = (
        result["pneumonia_probability"]
    )

    st.write("Normal")

    st.progress(normal_prob)

    st.caption(
        f"{normal_prob*100:.2f}%"
    )

    st.write("")

    st.write("Pneumonia")

    st.progress(
        pneumonia_prob
    )

    st.caption(
        f"{pneumonia_prob*100:.2f}%"
    )

    st.write("")

    st.info(
        """
The probabilities represent the
model's confidence for each class.

Higher confidence does not
guarantee a correct prediction.
"""
    )

# ==========================================================
# EXPLAINABILITY (Grad-CAM)
# ==========================================================

def render_explainability(
    original_image,
    gradcam_results,
):
    """
    Render Grad-CAM explainability section.

    Parameters
    ----------
    original_image : PIL.Image

    gradcam_results : dict

    Returns
    -------
    None
    """

    st.divider()

    st.markdown(
        """
        ## 🔬 Explainability (Grad-CAM)

        The visualizations below illustrate the image regions
        that contributed most strongly to the model prediction.
        """
    )

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------

    with col1:

        st.markdown("#### Original Chest X-ray")

        st.image(
            original_image,
            use_container_width=True,
        )

    # ------------------------------------------------------

    with col2:

        st.markdown("#### AI Attention Map")

        st.image(
            gradcam_results["colored_heatmap"],
            use_container_width=True,
            clamp=True,
        )

    # ------------------------------------------------------

    with col3:

        st.markdown("#### Grad-CAM Overlay")

        st.image(
            gradcam_results["overlay"],
            use_container_width=True,
            clamp=True,
        )

    st.markdown("")

    st.info(
        """
### AI Interpretation

The highlighted regions indicate the image areas that
contributed most strongly to the model prediction.

Grad-CAM visualizes **model attention** rather than
confirmed anatomical abnormalities.

The highlighted regions should therefore **not**
be interpreted as a definitive localization of disease.
"""
    )


# ==========================================================
# DISCLAIMER
# ==========================================================

def render_disclaimer():
    """
    Render clinical disclaimer.
    """

    st.divider()

    st.warning(DISCLAIMER)


# ==========================================================
# FOOTER
# ==========================================================

def render_footer():
    """
    Render application footer.
    """

    st.divider()

    st.markdown(
        f"""
<div style="text-align:center;
font-size:0.90rem;
color:gray;
padding-top:5px;
padding-bottom:10px;">

<b>{FOOTER}</b>

</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# ERROR MESSAGE
# ==========================================================

def show_error(message):
    """
    Display an error message.
    """

    st.error(message)


# ==========================================================
# SUCCESS MESSAGE
# ==========================================================

def show_success(message):
    """
    Display a success message.
    """

    st.success(message)


# ==========================================================
# INFORMATION MESSAGE
# ==========================================================

def show_information(message):
    """
    Display an information message.
    """

    st.info(message)
