"""
=============================================================
styles.py

Project:
Pediatric Pneumonia Detection using Xception CNN

Application:
PneumoScan Junior

Developer:
Harish

Description:
Global CSS styling for the Streamlit application.

This file contains ONLY CSS styling.
=============================================================
"""

import streamlit as st

from config import (
    PRIMARY_COLOR,
    BACKGROUND_COLOR,
    CARD_BACKGROUND,
    BORDER_COLOR,
    TEXT_COLOR,
    SUBTEXT_COLOR,
)


def apply_custom_css():
    """
    Apply global CSS styling.
    """

    st.markdown(
        f"""
<style>

/* ==========================================================
GENERAL PAGE
========================================================== */

.main .block-container {{
    max-width: 1300px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}}

body {{
    background-color: {BACKGROUND_COLOR};
    color: {TEXT_COLOR};
}}


/* ==========================================================
HEADERS
========================================================== */

.main-title {{
    font-size: 42px;
    font-weight: 700;
    color: {PRIMARY_COLOR};
    margin-bottom: 0rem;
}}

.project-title {{
    font-size: 26px;
    font-weight: 600;
    color: {TEXT_COLOR};
    margin-top: 0.25rem;
}}

.subtitle {{
    font-size: 17px;
    color: {SUBTEXT_COLOR};
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}}


/* ==========================================================
CARDS
========================================================== */

.card {{
    background: {CARD_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 18px;
    margin-top: 10px;
    margin-bottom: 15px;
}}


/* ==========================================================
SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {{
    background-color: #F8FAFC;
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {PRIMARY_COLOR};
}}


/* ==========================================================
BUTTONS
========================================================== */

div.stButton > button {{

    width: 100%;

    border-radius: 8px;

    border: none;

    font-weight: 600;

    padding: 0.60rem;

    transition: 0.2s;

}}

div.stButton > button:hover {{

    transform: scale(1.01);

}}


/* ==========================================================
SELECTBOX
========================================================== */

div[data-baseweb="select"] {{
    border-radius: 8px;
}}


/* ==========================================================
FILE UPLOADER
========================================================== */

section[data-testid="stFileUploader"] {{
    border-radius: 10px;
}}


/* ==========================================================
IMAGES
========================================================== */

img {{
    border-radius: 10px;
}}


/* ==========================================================
METRICS
========================================================== */

div[data-testid="metric-container"] {{

    background: white;

    border: 1px solid {BORDER_COLOR};

    padding: 15px;

    border-radius: 10px;

}}


/* ==========================================================
FOOTER
========================================================== */

.footer {{
    text-align: center;
    font-size: 13px;
    color: {SUBTEXT_COLOR};
    margin-top: 20px;
}}

</style>
        """,
        unsafe_allow_html=True,
    )