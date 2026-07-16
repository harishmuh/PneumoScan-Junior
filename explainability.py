"""
=============================================================
explainability.py

Project:
Pediatric Pneumonia Detection using Xception CNN

Application:
PneumoScan Junior

Developer:
Harish

Description:
Explainability utilities for the Xception CNN.

This module implements Grad-CAM without relying on any
third-party Grad-CAM libraries.

Current implementation:
    • Xception
    • Binary Classification
    • Logit-based Grad-CAM

TensorFlow Version:
    2.20

Keras Version:
    3.x
=============================================================
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from model_utils import load_model


# ==========================================================
# CONSTANTS
# ==========================================================

BACKBONE_NAME = "xception"

LAST_CONV_LAYER = "block14_sepconv2_act"


# ==========================================================
# BACKBONE
# ==========================================================

def get_backbone(
    model: tf.keras.Model
) -> tf.keras.Model:
    """
    Return the Xception backbone.

    Parameters
    ----------
    model : tf.keras.Model

    Returns
    -------
    tf.keras.Model
    """

    return model.get_layer(BACKBONE_NAME)


# ==========================================================
# FEATURE EXTRACTOR
# ==========================================================

def build_feature_extractor(
    backbone: tf.keras.Model
) -> tf.keras.Model:
    """
    Create a model that outputs the activation maps
    from the final convolution layer.

    Returns
    -------
    tf.keras.Model
    """

    return tf.keras.Model(

        inputs=backbone.inputs,

        outputs=backbone.get_layer(
            LAST_CONV_LAYER
        ).output

    )


# ==========================================================
# CLASSIFIER
# ==========================================================

def get_classifier_layers(
    model: tf.keras.Model
):
    """
    Return all classifier layers after the backbone.

    Returns
    -------
    list
    """

    return model.layers[1:]


# ==========================================================
# FORWARD PASS
# ==========================================================

def classifier_forward_pass(
    feature_maps: tf.Tensor,
    classifier_layers: list,
):
    """
    Forward pass through the classifier head.

    Uses logits instead of sigmoid probabilities.

    Parameters
    ----------
    feature_maps : tf.Tensor

    classifier_layers : list

    Returns
    -------
    logits : tf.Tensor

    probabilities : tf.Tensor
    """

    features = feature_maps

    for layer in classifier_layers[:-1]:

        if isinstance(

            layer,

            (
                tf.keras.layers.BatchNormalization,
                tf.keras.layers.Dropout,
            ),

        ):

            features = layer(

                features,

                training=False,

            )

        else:

            features = layer(features)

    final_dense = classifier_layers[-1]

    logits = (

        tf.matmul(

            features,

            final_dense.kernel,

        )

        + final_dense.bias

    )

    probabilities = tf.sigmoid(logits)

    return logits, probabilities


# ==========================================================
# GRADIENT
# ==========================================================

def compute_gradients(
    image: np.ndarray,
):
    """
    Compute Grad-CAM gradients.

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    feature_maps : tf.Tensor

    gradients : tf.Tensor

    probability : float
    """

    model = load_model()

    backbone = get_backbone(model)

    feature_extractor = build_feature_extractor(
        backbone
    )

    classifier_layers = get_classifier_layers(
        model
    )

    with tf.GradientTape() as tape:

        feature_maps = feature_extractor(

            image,

            training=False,

        )

        tape.watch(feature_maps)

        logits, probabilities = classifier_forward_pass(

            feature_maps,

            classifier_layers,

        )

        target_logit = logits[:, 0]

    gradients = tape.gradient(

        target_logit,

        feature_maps,

    )

    probability = float(

        probabilities[0][0]

    )

    return (

        feature_maps,

        gradients,

        probability,

    )


# ==========================================================
# GRAD-CAM
# ==========================================================

def generate_gradcam(
    image: np.ndarray,
):
    """
    Generate a normalized Grad-CAM heatmap.

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    heatmap : np.ndarray

    probability : float
    """

    feature_maps, gradients, probability = compute_gradients(
        image
    )

    pooled_gradients = tf.reduce_mean(

        gradients,

        axis=(0, 1, 2),

    )

    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(

        feature_maps * pooled_gradients,

        axis=-1,

    )

    heatmap = tf.maximum(

        heatmap,

        0,

    )

    max_value = tf.reduce_max(

        heatmap

    )

    if max_value > 0:

        heatmap /= max_value

    heatmap = heatmap.numpy()

    return heatmap, probability

# ==========================================================
# IMPORTS
# ==========================================================

import cv2


# ==========================================================
# HEATMAP RESIZE
# ==========================================================

def resize_heatmap(
    heatmap: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize heatmap to the original image size.

    Parameters
    ----------
    heatmap : np.ndarray

    image_size : tuple
        (width, height)

    Returns
    -------
    np.ndarray
    """

    return cv2.resize(
        heatmap,
        image_size,
        interpolation=cv2.INTER_CUBIC,
    )


# ==========================================================
# COLORIZE HEATMAP
# ==========================================================

def apply_colormap(
    heatmap: np.ndarray,
) -> np.ndarray:
    """
    Apply OpenCV JET colormap.

    Parameters
    ----------
    heatmap : np.ndarray

    Returns
    -------
    np.ndarray
    """

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    return colored_heatmap


# ==========================================================
# OVERLAY
# ==========================================================

def create_overlay(
    original_image: np.ndarray,
    colored_heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Overlay Grad-CAM on the original image.

    Parameters
    ----------
    original_image : np.ndarray

    colored_heatmap : np.ndarray

    alpha : float

    Returns
    -------
    np.ndarray
    """

    if original_image.dtype != np.uint8:

        original_image = (
            original_image * 255
        ).astype(np.uint8)

    overlay = cv2.addWeighted(

        original_image,

        1.0 - alpha,

        colored_heatmap,

        alpha,

        0,

    )

    return overlay


# ==========================================================
# COMPLETE VISUALIZATION
# ==========================================================

def generate_gradcam_visualization(
    original_image,
    preprocessed_image,
):
    """
    Generate all Grad-CAM visualizations.

    Parameters
    ----------
    original_image : PIL.Image

    preprocessed_image : np.ndarray

    Returns
    -------
    dict
    """

    heatmap, probability = generate_gradcam(
        preprocessed_image
    )

    width, height = original_image.size

    resized_heatmap = resize_heatmap(

        heatmap,

        (width, height),

    )

    colored_heatmap = apply_colormap(
        resized_heatmap
    )

    original_np = np.array(
        original_image
    )

    overlay = create_overlay(

        original_np,

        colored_heatmap,

    )

    return {

        "probability": probability,

        "heatmap": heatmap,

        "resized_heatmap": resized_heatmap,

        "colored_heatmap": colored_heatmap,

        "overlay": overlay,

    }