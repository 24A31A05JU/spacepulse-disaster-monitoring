import cv2
import numpy as np


def _prepare_image(image):
    """
    Convert an image into a grayscale float representation.

    The current SpacePulse baseline operates on RGB imagery.
    This function does not assume multispectral information.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    image = np.asarray(image)

    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    elif image.ndim == 2:

        gray = image.copy()

    else:

        raise ValueError(
            "Image must be a 2D grayscale or 3D RGB image."
        )

    gray = gray.astype(np.float32)

    return gray


def _normalize_difference(before_gray, after_gray):
    """
    Calculate normalized absolute pixel difference.
    """

    difference = cv2.absdiff(
        before_gray,
        after_gray
    )

    max_difference = np.max(
        difference
    )

    if max_difference > 0:

        normalized = (
            difference
            / max_difference
            * 255.0
        )

    else:

        normalized = np.zeros_like(
            difference
        )

    return normalized.astype(
        np.float32
    )


def _smooth_difference(difference, kernel_size):
    """
    Apply Gaussian smoothing at a particular spatial scale.
    """

    return cv2.GaussianBlur(
        difference,
        (
            kernel_size,
            kernel_size
        ),
        0
    )


def _create_binary_mask(
    difference,
    threshold
):
    """
    Convert a difference image into a binary change mask.
    """

    mask = (
        difference >= threshold
    ).astype(
        np.uint8
    )

    return mask


def _clean_mask(
    mask,
    kernel_size=5
):
    """
    Remove isolated noise and connect nearby changed regions.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size
        )
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    return cleaned


def _apply_valid_mask(
    change_mask,
    valid_mask
):
    """
    Restrict detected change to the valid registered overlap.
    """

    if valid_mask is None:

        return change_mask

    valid = (
        np.asarray(valid_mask) > 0
    ).astype(
        np.uint8
    )

    return (
        change_mask
        * valid
    ).astype(
        np.uint8
    )


def _calculate_scale_statistics(
    mask,
    valid_mask
):
    """
    Calculate measurable statistics for one scale.
    """

    if valid_mask is None:

        valid = np.ones_like(
            mask,
            dtype=np.uint8
        )

    else:

        valid = (
            np.asarray(valid_mask) > 0
        ).astype(
            np.uint8
        )

    valid_pixels = np.count_nonzero(
        valid
    )

    changed_pixels = np.count_nonzero(
        mask * valid
    )

    if valid_pixels > 0:

        change_percentage = (
            changed_pixels
            / valid_pixels
            * 100.0
        )

    else:

        change_percentage = 0.0

    return {
        "changed_pixels":
            int(changed_pixels),

        "valid_pixels":
            int(valid_pixels),

        "change_percentage":
            float(change_percentage)
    }


def _calculate_consistency(
    masks,
    valid_mask
):
    """
    Measure how consistently a pixel is classified as changed
    across all spatial scales.

    Returns:
        consistency_map
        consistency_score
    """

    if len(masks) == 0:

        raise ValueError(
            "At least one scale mask is required."
        )

    stack = np.stack(
        masks,
        axis=0
    ).astype(
        np.float32
    )

    consistency_map = (
        np.mean(
            stack,
            axis=0
        )
    )

    if valid_mask is not None:

        valid = (
            np.asarray(valid_mask) > 0
        )

    else:

        valid = np.ones(
            consistency_map.shape,
            dtype=bool
        )

    valid_consistency = (
        consistency_map[valid]
    )

    if valid_consistency.size == 0:

        consistency_score = 0.0

    else:

        # Percentage of valid pixels that remain classified
        # as changed in at least half of the scales.
        stable_pixels = np.count_nonzero(
            valid_consistency >= 0.5
        )

        consistency_score = (
            stable_pixels
            / valid_consistency.size
            * 100.0
        )

    return (
        consistency_map,
        float(consistency_score)
    )


def detect_multiscale_change(
    before_image,
    after_image,
    valid_mask=None,
    threshold=30,
    scales=None
):
    """
    Perform multi-scale temporal change detection.

    Pipeline:

        BEFORE + AFTER
              ↓
        Pixel difference
              ↓
        Fine-scale analysis
              ↓
        Medium-scale analysis
              ↓
        Coarse-scale analysis
              ↓
        Cross-scale consistency
              ↓
        Stable change map

    Parameters
    ----------
    before_image:
        Reference BEFORE RGB image.

    after_image:
        Registered AFTER RGB image.

    valid_mask:
        Valid overlap mask produced by image registration.

    threshold:
        Pixel-difference threshold from 0-255.

    scales:
        Gaussian kernel sizes.

    Returns
    -------
    dictionary containing:

        difference
        scale_masks
        scale_statistics
        consistency_map
        stable_change_mask
        stable_change_percentage
        consistency_score
    """

    if before_image is None:

        raise ValueError(
            "BEFORE image is required."
        )

    if after_image is None:

        raise ValueError(
            "AFTER image is required."
        )

    # ---------------------------------------------------------
    # DEFAULT SPATIAL SCALES
    # ---------------------------------------------------------

    if scales is None:

        scales = [
            3,
            7,
            15
        ]

    # Make sure kernel sizes are odd.
    scales = [
        int(scale)
        if int(scale) % 2 == 1
        else int(scale) + 1
        for scale in scales
    ]

    # ---------------------------------------------------------
    # PREPARE IMAGES
    # ---------------------------------------------------------

    before_gray = _prepare_image(
        before_image
    )

    after_gray = _prepare_image(
        after_image
    )

    if before_gray.shape != after_gray.shape:

        raise ValueError(
            "BEFORE and AFTER images must have "
            "the same dimensions."
        )

    # ---------------------------------------------------------
    # BASE PIXEL DIFFERENCE
    # ---------------------------------------------------------

    difference = _normalize_difference(
        before_gray,
        after_gray
    )

    # ---------------------------------------------------------
    # MULTI-SCALE ANALYSIS
    # ---------------------------------------------------------

    scale_masks = []

    scale_statistics = []

    scale_differences = []

    for scale in scales:

        smoothed_difference = (
            _smooth_difference(
                difference,
                scale
            )
        )

        scale_differences.append(
            smoothed_difference
        )

        mask = _create_binary_mask(
            smoothed_difference,
            threshold
        )

        mask = _clean_mask(
            mask,
            kernel_size=5
        )

        mask = _apply_valid_mask(
            mask,
            valid_mask
        )

        statistics = (
            _calculate_scale_statistics(
                mask,
                valid_mask
            )
        )

        statistics["scale"] = scale

        scale_masks.append(
            mask
        )

        scale_statistics.append(
            statistics
        )

    # ---------------------------------------------------------
    # CROSS-SCALE CONSISTENCY
    # ---------------------------------------------------------

    (
        consistency_map,
        consistency_score
    ) = _calculate_consistency(
        scale_masks,
        valid_mask
    )

    # ---------------------------------------------------------
    # STABLE CHANGE MAP
    # ---------------------------------------------------------
    # A pixel is considered stable change when it is detected
    # at least half of the spatial scales.

    stable_change_mask = (
        consistency_map >= 0.5
    ).astype(
        np.uint8
    )

    stable_change_mask = _apply_valid_mask(
        stable_change_mask,
        valid_mask
    )

    # ---------------------------------------------------------
    # STABLE CHANGE PERCENTAGE
    # ---------------------------------------------------------

    if valid_mask is None:

        valid = np.ones_like(
            stable_change_mask,
            dtype=np.uint8
        )

    else:

        valid = (
            np.asarray(valid_mask) > 0
        ).astype(
            np.uint8
        )

    valid_pixels = np.count_nonzero(
        valid
    )

    stable_pixels = np.count_nonzero(
        stable_change_mask
        * valid
    )

    if valid_pixels > 0:

        stable_change_percentage = (
            stable_pixels
            / valid_pixels
            * 100.0
        )

    else:

        stable_change_percentage = 0.0

    # ---------------------------------------------------------
    # SCALE RANGE
    # ---------------------------------------------------------

    if scale_statistics:

        percentages = [
            item["change_percentage"]
            for item in scale_statistics
        ]

        minimum_change = float(
            np.min(percentages)
        )

        maximum_change = float(
            np.max(percentages)
        )

    else:

        minimum_change = 0.0
        maximum_change = 0.0

    scale_spread = (
        maximum_change
        - minimum_change
    )

    # ---------------------------------------------------------
    # INTERPRETATION
    # ---------------------------------------------------------

    if consistency_score >= 70:

        consistency_label = "HIGH"

        interpretation = (
            "The detected temporal change remains "
            "relatively stable across spatial scales."
        )

    elif consistency_score >= 40:

        consistency_label = "MODERATE"

        interpretation = (
            "The detected temporal change shows "
            "partial stability across spatial scales."
        )

    else:

        consistency_label = "LOW"

        interpretation = (
            "The detected temporal change is not "
            "consistently observed across spatial scales."
        )

    # ---------------------------------------------------------
    # RETURN RESULTS
    # ---------------------------------------------------------

    return {

        "difference":
            difference,

        "scales":
            scales,

        "scale_masks":
            scale_masks,

        "scale_differences":
            scale_differences,

        "scale_statistics":
            scale_statistics,

        "consistency_map":
            consistency_map,

        "stable_change_mask":
            stable_change_mask,

        "stable_change_percentage":
            float(
                stable_change_percentage
            ),

        "consistency_score":
            float(
                consistency_score
            ),

        "consistency_label":
            consistency_label,

        "minimum_change":
            minimum_change,

        "maximum_change":
            maximum_change,

        "scale_spread":
            float(
                scale_spread
            ),

        "interpretation":
            interpretation
    }