import cv2
import numpy as np


def load_image(image_file):
    """
    Load an uploaded image and convert it to RGB.
    """
    file_bytes = np.asarray(
        bytearray(image_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_images(before, after):
    """
    Resize the AFTER image to match the BEFORE image dimensions.
    """

    height, width = before.shape[:2]

    after_resized = cv2.resize(
        after,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    return before, after_resized


def calculate_change_map(before, after, threshold=30):
    """
    Calculate a pixel-level temporal change map.

    Returns:
        change_mask
        change_percentage
        difference_image
    """

    # Convert RGB images to grayscale
    before_gray = cv2.cvtColor(
        before,
        cv2.COLOR_RGB2GRAY
    )

    after_gray = cv2.cvtColor(
        after,
        cv2.COLOR_RGB2GRAY
    )

    # Absolute temporal difference
    difference = cv2.absdiff(
        before_gray,
        after_gray
    )

    # Threshold the difference
    _, change_mask = cv2.threshold(
        difference,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Calculate percentage of changed pixels
    changed_pixels = np.count_nonzero(change_mask)

    total_pixels = change_mask.shape[0] * change_mask.shape[1]

    change_percentage = (
        changed_pixels / total_pixels
    ) * 100

    return (
        change_mask,
        change_percentage,
        difference
    )


def create_overlay(image, change_mask):
    """
    Highlight detected changes over the original image.
    """

    overlay = image.copy()

    # Create red highlight
    highlight = np.zeros_like(image)
    highlight[:, :, 0] = 255

    changed_area = change_mask > 0

    overlay[changed_area] = (
        0.55 * overlay[changed_area]
        + 0.45 * highlight[changed_area]
    ).astype(np.uint8)

    return overlay