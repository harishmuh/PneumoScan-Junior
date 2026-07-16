"""
=============================================================
model_utils.py

Project:
Pediatric Pneumonia Detection using Xception CNN

Application:
PneumoScan Junior

Developer:
Harish

Description:
Utility functions for loading the trained model
and performing predictions.

This module contains NO Streamlit code.
=============================================================
"""

from functools import lru_cache

import tensorflow as tf

from config import (
    MODEL_PATH,
    MODEL_NAME,
    FRAMEWORK,
    FRAMEWORK_VERSION,
    CLASS_NAMES,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    PREDICTION_THRESHOLD,
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained Keras model.

    Returns
    -------
    tensorflow.keras.Model
    """

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ==========================================================
# PREDICTION
# ==========================================================

def predict(preprocessed_image):
    """
    Predict pneumonia probability.

    Parameters
    ----------
    preprocessed_image : numpy.ndarray
        Image already resized, normalized and
        expanded to (1,224,224,3).

    Returns
    -------
    dict
    """

    model = load_model()

    probability = float(
        model.predict(
            preprocessed_image,
            verbose=0
        )[0][0]
    )

    normal_probability = 1.0 - probability

    pneumonia_probability = probability

    if probability >= PREDICTION_THRESHOLD:

        predicted_class = POSITIVE_CLASS

        confidence = pneumonia_probability

    else:

        predicted_class = NEGATIVE_CLASS

        confidence = normal_probability

    return {

        "predicted_class": predicted_class,

        "confidence": confidence,

        "normal_probability": normal_probability,

        "pneumonia_probability": pneumonia_probability,

        "raw_probability": probability

    }


# ==========================================================
# MODEL INFORMATION
# ==========================================================

def get_model_information():
    """
    Return model information.
    """

    return {

        "model_name": MODEL_NAME,

        "framework": FRAMEWORK,

        "framework_version": FRAMEWORK_VERSION,

        "classes": CLASS_NAMES,

        "threshold": PREDICTION_THRESHOLD

    }


# ==========================================================
# MODEL STATUS
# ==========================================================

def model_is_loaded():
    """
    Check whether the model
    has already been loaded.

    Returns
    -------
    bool
    """

    return load_model.cache_info().currsize > 0


# ==========================================================
# CLEAR CACHE
# ==========================================================

def clear_model_cache():
    """
    Clear cached model.

    Mainly useful during development.
    """

    load_model.cache_clear()