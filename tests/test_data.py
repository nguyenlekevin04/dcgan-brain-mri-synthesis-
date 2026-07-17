"""
Unit tests for the data module (data.py).
"""
from src.data import normalize_images, preprocess_images, load_images, resize_images, convert_to_grayscale, to_numpy_array
from pathlib import Path
import numpy as np

def test_load_images(dummy_image_dir: Path):
    """
    Test loading images from a directory. It checks if the correct number of images are loaded and handles corrupt images gracefully.
    """
    images = load_images(dummy_image_dir)
    assert len(images) == 5

def test_load_images_with_corrupt_image(dummy_dir_with_corrupt_image: Path):
    """
    Test loading images from a directory containing a corrupt image. It checks if the correct number of valid images are loaded.
    """
    images = load_images(dummy_dir_with_corrupt_image)
    assert len(images) == 1

def test_resize_images(dummy_rgb_image):
    """
    Test resizing images. It checks if the resized image has the expected dimensions.
    """
    resized_images = resize_images([dummy_rgb_image], size=(32, 32))
    assert resized_images[0].size == (32, 32)

def test_normalize_images():
    """
    Test normalizing images. It checks if the normalized values are within the expected range.
    """
    dummy_array = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    normalized_array = normalize_images(dummy_array)
    assert np.isclose(normalized_array.min(), -1.0)
    assert np.isclose(normalized_array.max(), 1.0)

def test_convert_to_grayscale(dummy_rgb_image):
    """
    Test converting images to grayscale. It checks if the converted image has the expected mode.
    """
    grayscale_images = convert_to_grayscale([dummy_rgb_image])
    assert grayscale_images[0].mode == "L"