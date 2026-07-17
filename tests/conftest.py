"""
This file contains pytest fixtures for testing purposes.
"""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

@pytest.fixture
def dummy_rgb_image():
    """Fixture to create a dummy RGB image."""
    array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(array)

@pytest.fixture
def dummy_grayscale_image():
    """Fixture to create a dummy grayscale image."""
    array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    return Image.fromarray(array)

@pytest.fixture
def dummy_image_list():
    """Fixture to create a list of dummy images."""
    return [dummy_rgb_image for _ in range(5)]

@pytest.fixture
def dummy_image_dir(tmp_path: Path):
    """Fixture to create a temporary directory with dummy images."""
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        img.save(tmp_path / f"image_{i}.jpg")
    return tmp_path

@pytest.fixture
def dummy_dir_with_corrupt_image(tmp_path: Path, dummy_rgb_image: Image):
    """Fixture to create a temporary directory with one valid and one corrupt image."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    dummy_rgb_image.save(img_dir / "valid.jpg")

    corrupt_path = img_dir / "corrupt.jpg"
    corrupt_path.write_text("das ist kein bild, nur text")

    return img_dir