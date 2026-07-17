"""
This module provides functions to load and preprocess images from a specified directory.
"""
from pathlib import Path
from PIL import Image
import numpy as np

def load_images(path: Path) -> list:
    """
    Load images from the specified directory and return a list of PIL Image objects.
    """
    images = []
    for img_path in path.glob("*.jpg"):
        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images

def resize_images(images: list, size=(128, 128)) -> list:
    """
    Resize a list of PIL Image objects to the specified size.
    """
    resized_images = []
    for img in images:
        resized_img = img.resize(size)
        resized_images.append(resized_img)
    return resized_images

def convert_to_grayscale(images: list) -> list:
    """
    Convert a list of PIL Image objects to grayscale.
    """
    grayscale_images = []
    for img in images:
        grayscale_img = img.convert("L")
        grayscale_images.append(grayscale_img)
    return grayscale_images

def to_numpy_array(images: list) -> np.ndarray:
    """
    Convert a list of PIL Image objects to a NumPy array.
    """
    return np.array([np.array(img) for img in images])

def normalize_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize a NumPy array of image data to have values in the range [-1, 1].
    """
    return (images.astype(np.float32) / 127.5) - 1.0

def preprocess_images(path: Path, size=(128, 128)) -> np.ndarray:
    """
    Load, resize, convert to grayscale, normalize, and convert images to a NumPy array.
    """
    images = load_images(path)
    images = resize_images(images, size)
    images = convert_to_grayscale(images)
    images = to_numpy_array(images)
    return normalize_images(images)

result = preprocess_images(Path("data/yes"), size=(128, 128))
print(result.shape)