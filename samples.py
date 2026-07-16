"""
=============================================================
samples.py

Project:
Pediatric Pneumonia Detection using Xception CNN

Application:
PneumoScan Junior

Description:
Metadata for all built-in sample images.

This file stores ONLY sample image information.
=============================================================
"""

from config import (
    NORMAL_SAMPLE_DIR,
    PNEUMONIA_SAMPLE_DIR,
    DATASET_NAME,
    DATASET_DESCRIPTION,
)

# ==========================================================
# SAMPLE IMAGE DATABASE
# ==========================================================

SAMPLES = {

    # ======================================================
    # NORMAL CASES
    # ======================================================

    "🟢 Normal Case A": {

        "path": NORMAL_SAMPLE_DIR / "normal_A.JPEG",

        "class": "Normal",

        "reference_diagnosis": "Normal",

        "dataset": DATASET_NAME,

        "description": (
            "Representative pediatric chest X-ray "
            "without radiographic evidence of pneumonia."
        )

    },

    "🟢 Normal Case B": {

        "path": NORMAL_SAMPLE_DIR / "normal_B.JPEG",

        "class": "Normal",

        "reference_diagnosis": "Normal",

        "dataset": DATASET_NAME,

        "description": (
            "Representative pediatric normal chest X-ray."
        )

    },

    "🟢 Normal Case C": {

        "path": NORMAL_SAMPLE_DIR / "normal_C.JPEG",

        "class": "Normal",

        "reference_diagnosis": "Normal",

        "dataset": DATASET_NAME,

        "description": (
            "Representative pediatric normal chest X-ray."
        )

    },

    # ======================================================
    # PNEUMONIA CASES
    # ======================================================

    "🔴 Pneumonia Case A": {

        "path": PNEUMONIA_SAMPLE_DIR / "pneumonia_A.JPEG",

        "class": "Pneumonia",

        "reference_diagnosis": "Pneumonia",

        "dataset": DATASET_NAME,

        "description": (
            "Representative pediatric pneumonia chest X-ray."
        )

    },

    "🔴 Pneumonia Case B": {

        "path": PNEUMONIA_SAMPLE_DIR / "pneumonia_B.JPEG",

        "class": "Pneumonia",

        "reference_diagnosis": "Pneumonia",

        "dataset": DATASET_NAME,

        "description": (
            "Representative pediatric pneumonia chest X-ray."
        )

    },

    "🔴 Pneumonia Case C": {

        "path": PNEUMONIA_SAMPLE_DIR / "pneumonia_C.JPEG",

        "class": "Pneumonia",

        "reference_diagnosis": "Pneumonia",

        "dataset": DATASET_NAME,

        "description": (
            "Representative pediatric pneumonia chest X-ray."
        )

    }

}

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def get_sample_names():
    """
    Return a list of all sample names.
    """
    return list(SAMPLES.keys())


def get_sample(sample_name):
    """
    Return metadata for a selected sample.

    Parameters
    ----------
    sample_name : str

    Returns
    -------
    dict
    """
    return SAMPLES[sample_name]


def get_sample_path(sample_name):
    """
    Return the pathlib.Path of a sample image.
    """
    return SAMPLES[sample_name]["path"]


def get_reference_diagnosis(sample_name):
    """
    Return the reference diagnosis.
    """
    return SAMPLES[sample_name]["reference_diagnosis"]


def get_dataset_name(sample_name):
    """
    Return dataset name.
    """
    return SAMPLES[sample_name]["dataset"]