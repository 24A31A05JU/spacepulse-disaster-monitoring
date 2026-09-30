import cv2
import numpy as np


def _normalize_difference(difference):
    """
    Convert the difference output into a single 2D float array.
    """

    if difference is None:
        return None

    diff = np.asarray(difference)

    if diff.ndim == 3:
        diff = np.mean(diff, axis=2)

    diff = diff.astype(np.float32)

    if diff.size == 0:
        return None

    return diff


def calculate_change_quality(
    change_mask,
    difference,
    valid_mask,
    registration_info
):
    """
    Calculate measurable evidence properties for detected change.

    This is a transparent heuristic research score.
    It is NOT a probability that a disaster occurred.
    """

    mask = (
        np.asarray(change_mask) > 0
    ).astype(np.uint8)

    height, width = mask.shape

    total_pixels = mask.size

    # ---------------------------------------------------------
    # VALID REGION
    # ---------------------------------------------------------

    if valid_mask is not None:

        valid = (
            np.asarray(valid_mask) > 0
        ).astype(np.uint8)

    else:

        valid = np.ones_like(
            mask,
            dtype=np.uint8
        )

    valid_pixels = np.count_nonzero(
        valid
    )

    changed_pixels = np.count_nonzero(
        mask * valid
    )

    # ---------------------------------------------------------
    # CHANGE COVERAGE
    # ---------------------------------------------------------

    if valid_pixels > 0:

        change_percentage = (
            changed_pixels
            / valid_pixels
            * 100
        )

    else:

        change_percentage = 0.0

    # ---------------------------------------------------------
    # CONNECTED COMPONENT ANALYSIS
    # ---------------------------------------------------------

    binary = (
        mask * valid
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if num_labels > 1:

        areas = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        areas = areas[
            areas > 0
        ]

    else:

        areas = np.array(
            [],
            dtype=np.float32
        )

    components = len(areas)

    if components > 0:

        largest_area = float(
            np.max(areas)
        )

        largest_component_ratio = (
            largest_area
            / changed_pixels
            * 100
            if changed_pixels > 0
            else 0.0
        )

    else:

        largest_component_ratio = 0.0

    # ---------------------------------------------------------
    # FRAGMENTATION
    # ---------------------------------------------------------

    if changed_pixels > 0:

        small_threshold = max(
            10,
            int(
                total_pixels * 0.00005
            )
        )

        small_components = np.sum(
            areas < small_threshold
        )

        small_area = np.sum(
            areas[
                areas < small_threshold
            ]
        )

        fragmentation_ratio = (
            small_area
            / changed_pixels
            * 100
        )

    else:

        small_components = 0

        fragmentation_ratio = 0.0

    # ---------------------------------------------------------
    # BORDER EFFECT
    # ---------------------------------------------------------

    border_size = max(
        2,
        int(
            min(height, width) * 0.02
        )
    )

    border_mask = np.zeros_like(
        binary,
        dtype=np.uint8
    )

    border_mask[
        :border_size,
        :
    ] = 1

    border_mask[
        -border_size:,
        :
    ] = 1

    border_mask[
        :,
        :border_size
    ] = 1

    border_mask[
        :,
        -border_size:
    ] = 1

    border_changed = np.count_nonzero(
        binary * border_mask
    )

    border_change_ratio = (
        border_changed
        / changed_pixels
        * 100
        if changed_pixels > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # PIXEL DIFFERENCE STRENGTH
    # ---------------------------------------------------------

    diff = _normalize_difference(
        difference
    )

    mean_changed_difference = 0.0
    mean_unchanged_difference = 0.0
    difference_separation = 0.0

    if diff is not None:

        changed_region = (
            (binary > 0)
            & (valid > 0)
        )

        unchanged_region = (
            (binary == 0)
            & (valid > 0)
        )

        if np.any(changed_region):

            mean_changed_difference = float(
                np.mean(
                    diff[changed_region]
                )
            )

        if np.any(unchanged_region):

            mean_unchanged_difference = float(
                np.mean(
                    diff[unchanged_region]
                )
            )

        denominator = (
            mean_changed_difference
            + mean_unchanged_difference
        )

        if denominator > 0:

            difference_separation = (
                mean_changed_difference
                / denominator
                * 100
            )

    # ---------------------------------------------------------
    # REGISTRATION INFORMATION
    # ---------------------------------------------------------

    registration_accepted = (
        registration_info.get(
            "accepted",
            False
        )
    )

    registration_quality = (
        registration_info.get(
            "quality",
            "UNKNOWN"
        )
    )

    inlier_ratio = float(
        registration_info.get(
            "inlier_ratio",
            0.0
        )
    )

    # IMPORTANT:
    # overlap_ratio is stored as a percentage when registration
    # succeeds. If registration fails, the registration module
    # falls back to the original AFTER image, so the comparison
    # covers the complete original scene.

    if registration_accepted:

        overlap_ratio = float(
            registration_info.get(
                "overlap_ratio",
                100.0
            )
        )

    else:

        overlap_ratio = 100.0

    reprojection_error = (
        registration_info.get(
            "reprojection_error"
        )
    )

    # ---------------------------------------------------------
    # EVIDENCE COMPONENT SCORES
    # ---------------------------------------------------------

    scores = {}

    # Registration
    if (
        registration_accepted
        and inlier_ratio >= 0.60
        and overlap_ratio >= 85
    ):

        scores["registration"] = 1.0

    elif (
        registration_accepted
        and inlier_ratio >= 0.40
        and overlap_ratio >= 75
    ):

        scores["registration"] = 0.75

    elif registration_accepted:

        scores["registration"] = 0.50

    else:

        scores["registration"] = 0.0

    # Spatial coherence
    if largest_component_ratio >= 50:

        scores["spatial_coherence"] = 1.0

    elif largest_component_ratio >= 25:

        scores["spatial_coherence"] = 0.75

    elif largest_component_ratio >= 10:

        scores["spatial_coherence"] = 0.50

    else:

        scores["spatial_coherence"] = 0.25

    # Difference separation
    if difference_separation >= 80:

        scores["difference_separation"] = 1.0

    elif difference_separation >= 65:

        scores["difference_separation"] = 0.75

    elif difference_separation >= 50:

        scores["difference_separation"] = 0.50

    else:

        scores["difference_separation"] = 0.25

    # Fragmentation
    if fragmentation_ratio <= 10:

        scores["fragmentation"] = 1.0

    elif fragmentation_ratio <= 25:

        scores["fragmentation"] = 0.75

    elif fragmentation_ratio <= 50:

        scores["fragmentation"] = 0.50

    else:

        scores["fragmentation"] = 0.25

    # Boundary reliability
    if border_change_ratio <= 10:

        scores["border_reliability"] = 1.0

    elif border_change_ratio <= 25:

        scores["border_reliability"] = 0.75

    elif border_change_ratio <= 50:

        scores["border_reliability"] = 0.50

    else:

        scores["border_reliability"] = 0.25

    # ---------------------------------------------------------
    # BASE HEURISTIC SCORE
    # ---------------------------------------------------------

    overall_score = (
        scores["registration"] * 0.35
        + scores["spatial_coherence"] * 0.20
        + scores["difference_separation"] * 0.20
        + scores["fragmentation"] * 0.15
        + scores["border_reliability"] * 0.10
    )

    evidence_score = (
        overall_score * 100
    )

    # ---------------------------------------------------------
    # IMPORTANT SAFETY RULE
    # ---------------------------------------------------------
    # If registration fails, the evidence chain should not be
    # presented as moderate/strong merely because the pixel
    # change happens to have a coherent shape.

    if not registration_accepted:

        evidence_score = min(
            evidence_score,
            39.9
        )

    # ---------------------------------------------------------
    # EVIDENCE STRENGTH
    # ---------------------------------------------------------

    if evidence_score >= 80:

        evidence_strength = "STRONG"

    elif evidence_score >= 60:

        evidence_strength = "MODERATE"

    elif evidence_score >= 40:

        evidence_strength = "LIMITED"

    else:

        evidence_strength = "WEAK"

    # ---------------------------------------------------------
    # FALSE-POSITIVE INDICATORS
    # ---------------------------------------------------------

    warnings = []

    if not registration_accepted:

        warnings.append(
            "Image registration was not accepted. "
            "The current change result should therefore "
            "be treated cautiously."
        )

    if border_change_ratio > 25:

        warnings.append(
            "A substantial portion of detected change "
            "occurs near image boundaries."
        )

    if fragmentation_ratio > 40:

        warnings.append(
            "Detected change is highly fragmented."
        )

    if difference_separation < 50:

        warnings.append(
            "Changed and unchanged regions have "
            "weak difference separation."
        )

    if change_percentage > 50:

        warnings.append(
            "A very large portion of the valid scene "
            "is classified as changed."
        )

    if (
        registration_accepted
        and reprojection_error is not None
        and reprojection_error > 8
    ):

        warnings.append(
            "Registration reprojection error is relatively high."
        )

    if not warnings:

        warnings.append(
            "No major false-positive indicator was detected "
            "by the current baseline checks."
        )

    # ---------------------------------------------------------
    # INTERPRETATION
    # ---------------------------------------------------------

    if not registration_accepted:

        interpretation = (
            "Registration was not accepted, so the current "
            "temporal change result should be treated as a "
            "preliminary visual-change signal rather than "
            "strong event evidence. Additional aligned "
            "observations are recommended."
        )

    elif evidence_strength == "STRONG":

        interpretation = (
            "The detected change has relatively strong "
            "support from the current registration, spatial "
            "and pixel-level evidence. It should still be "
            "validated with independent Earth-observation data."
        )

    elif evidence_strength == "MODERATE":

        interpretation = (
            "The detected change has moderate support from "
            "the current evidence chain. Additional temporal, "
            "spectral or radar observations are recommended."
        )

    elif evidence_strength == "LIMITED":

        interpretation = (
            "The detected change has limited evidence support. "
            "The pattern should be treated cautiously and "
            "validated with additional observations."
        )

    else:

        interpretation = (
            "The current evidence is weak. The detected change "
            "should not be interpreted as an event signal without "
            "additional validation."
        )

    return {

        "change_percentage":
            change_percentage,

        "components":
            components,

        "largest_component_ratio":
            largest_component_ratio,

        "fragmentation_ratio":
            fragmentation_ratio,

        "small_components":
            int(
                small_components
            ),

        "border_change_ratio":
            border_change_ratio,

        "mean_changed_difference":
            mean_changed_difference,

        "mean_unchanged_difference":
            mean_unchanged_difference,

        "difference_separation":
            difference_separation,

        "registration_quality":
            registration_quality,

        "registration_accepted":
            registration_accepted,

        "inlier_ratio":
            inlier_ratio,

        "overlap_ratio":
            overlap_ratio,

        "reprojection_error":
            reprojection_error,

        "scores":
            scores,

        "evidence_score":
            evidence_score,

        "evidence_strength":
            evidence_strength,

        "warnings":
            warnings,

        "interpretation":
            interpretation
    }