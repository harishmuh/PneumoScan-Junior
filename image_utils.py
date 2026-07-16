"""
=============================================================
image_utils.py

Project:
Pediatric Pneumonia Detection using Xception CNN

Application:
PneumoScan Junior

Developer:
Harish

Description:
Utility functions for loading and preprocessing images.

=============================================================
"""

from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    IMAGE_SIZE,
)

# ==========================================================
# IMAGE LOADING
# ==========================================================

def load_image(image_source):
    """
    Load an image.

    Parameters
    ----------
    image_source :
        pathlib.Path
        str
        UploadedFile
        PIL.Image.Image

    Returns
    -------
    PIL.Image.Image
    """

    if isinstance(image_source, Image.Image):
        return image_source

    return Image.open(image_source)


# ==========================================================
# IMAGE CONVERSION
# ==========================================================

def convert_to_rgb(image):
    """
    Convert image to RGB.
    """

    return image.convert("RGB")


# ==========================================================
# IMAGE RESIZE
# ==========================================================

def resize_image(image):
    """
    Resize image to model input size.
    """

    return image.resize(IMAGE_SIZE)


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_image(image):
    """
    Convert image to float32 and normalize to [0,1].
    """

    image = np.asarray(image, dtype=np.float32)

    image = image / 255.0

    return image


# ==========================================================
# ADD BATCH DIMENSION
# ==========================================================

def expand_dimension(image):
    """
    Add batch dimension.

    Shape:
        (224,224,3)

    becomes

        (1,224,224,3)
    """

    return np.expand_dims(image, axis=0)


# ==========================================================
# COMPLETE PREPROCESSING
# ==========================================================

def preprocess_image(image_source):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    image_source

    Returns
    -------
    numpy.ndarray
        Ready for prediction.
    """

    image = load_image(image_source)

    image = convert_to_rgb(image)

    image = resize_image(image)

    image = normalize_image(image)

    image = expand_dimension(image)

    return image


# ==========================================================
# IMAGE INFORMATION
# ==========================================================

def get_image_size(image):
    """
    Return original image size.
    """

    return image.size


def get_image_mode(image):
    """
    Return image mode.

    Example:
        RGB
        L
    """

    return image.mode


# ==========================================================
# IMAGE VALIDATION
# ==========================================================

def is_rgb(image):
    """
    Check whether image is RGB.
    """

    return image.mode == "RGB"


def validate_image(image):
    """
    Validate image.

    Returns
    -------
    bool
    """

    return image is not None


# ==========================================================
# IMAGE SUMMARY
# ==========================================================

def get_image_information(image):
    """
    Return image metadata.
    """

    return {

        "width": image.width,

        "height": image.height,

        "mode": image.mode

    }