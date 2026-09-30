import cv2
import numpy as np


def _to_gray(image):
    """
    Convert RGB image to grayscale and improve local contrast.
    """
    if image is None:
        raise ValueError("Image cannot be None.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    return gray


def _calculate_reprojection_error(
    source_points,
    destination_points,
    homography,
    inlier_mask
):
    """
    Calculate average reprojection error for RANSAC inliers.
    """

    if homography is None or inlier_mask is None:
        return None

    inliers = inlier_mask.ravel().astype(bool)

    if np.sum(inliers) < 1:
        return None

    projected = cv2.perspectiveTransform(
        source_points,
        homography
    )

    errors = np.linalg.norm(
        projected[inliers] -
        destination_points[inliers],
        axis=2
    )

    if len(errors) == 0:
        return None

    return float(np.mean(errors))


def register_images(before_image, after_image):
    """
    Align the AFTER image to the BEFORE image.

    Method:
        1. ORB feature detection
        2. Hamming descriptor matching
        3. Lowe-style ratio filtering
        4. RANSAC homography estimation
        5. Perspective warping

    Returns:
        aligned_after
        valid_mask
        registration_info
    """

    if before_image is None or after_image is None:
        raise ValueError("Both images are required for registration.")

    # Make sure both images have the same spatial dimensions.
    height, width = before_image.shape[:2]

    after_resized = cv2.resize(
        after_image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    before_gray = _to_gray(before_image)
    after_gray = _to_gray(after_resized)

    # ---------------------------------------------------------
    # 1. Detect ORB features
    # ---------------------------------------------------------

    orb = cv2.ORB_create(
        nfeatures=4000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=12
    )

    keypoints_before, descriptors_before = orb.detectAndCompute(
        before_gray,
        None
    )

    keypoints_after, descriptors_after = orb.detectAndCompute(
        after_gray,
        None
    )

    info = {
        "method": "ORB + RANSAC Homography",
        "keypoints_before": 0 if keypoints_before is None else len(keypoints_before),
        "keypoints_after": 0 if keypoints_after is None else len(keypoints_after),
        "candidate_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reprojection_error": None,
        "overlap_ratio": 1.0,
        "quality": "NOT ACCEPTED",
        "accepted": False,
        "reason": ""
    }

    # ---------------------------------------------------------
    # 2. Validate feature detection
    # ---------------------------------------------------------

    if descriptors_before is None or descriptors_after is None:

        info["reason"] = (
            "Insufficient visual features were detected. "
            "The original image pair was retained."
        )

        return (
            after_resized,
            np.ones(
                (height, width),
                dtype=np.uint8
            ),
            info
        )

    if len(descriptors_before) < 10 or len(descriptors_after) < 10:

        info["reason"] = (
            "Too few features were detected for reliable registration."
        )

        return (
            after_resized,
            np.ones(
                (height, width),
                dtype=np.uint8
            ),
            info
        )

    # ---------------------------------------------------------
    # 3. Feature matching
    # ---------------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    knn_matches = matcher.knnMatch(
        descriptors_after,
        descriptors_before,
        k=2
    )

    info["candidate_matches"] = len(knn_matches)

    good_matches = []

    # Lowe ratio test
    for pair in knn_matches:

        if len(pair) < 2:
            continue

        match_1, match_2 = pair

        if match_1.distance < 0.75 * match_2.distance:
            good_matches.append(match_1)

    info["good_matches"] = len(good_matches)

    # ---------------------------------------------------------
    # 4. Estimate homography
    # ---------------------------------------------------------

    if len(good_matches) < 12:

        info["reason"] = (
            f"Only {len(good_matches)} reliable feature matches "
            "were found. Registration was not accepted."
        )

        return (
            after_resized,
            np.ones(
                (height, width),
                dtype=np.uint8
            ),
            info
        )

    source_points = np.float32(
        [
            keypoints_after[m.queryIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    destination_points = np.float32(
        [
            keypoints_before[m.trainIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0
    )

    if homography is None or inlier_mask is None:

        info["reason"] = (
            "A reliable geometric transformation could not "
            "be estimated."
        )

        return (
            after_resized,
            np.ones(
                (height, width),
                dtype=np.uint8
            ),
            info
        )

    inliers = int(
        np.sum(inlier_mask)
    )

    inlier_ratio = (
        inliers / len(good_matches)
        if len(good_matches) > 0
        else 0
    )

    reprojection_error = _calculate_reprojection_error(
        source_points,
        destination_points,
        homography,
        inlier_mask
    )

    info["inliers"] = inliers
    info["inlier_ratio"] = inlier_ratio
    info["reprojection_error"] = reprojection_error

    # ---------------------------------------------------------
    # 5. Registration quality
    # ---------------------------------------------------------

    # Conservative acceptance criteria.
    accepted = (
        inliers >= 8
        and inlier_ratio >= 0.25
        and (
            reprojection_error is None
            or reprojection_error <= 8.0
        )
    )

    if not accepted:

        info["reason"] = (
            "Feature matches were found, but the geometric "
            "agreement was not strong enough for reliable "
            "registration."
        )

        if inlier_ratio >= 0.50:
            info["quality"] = "MODERATE"
        else:
            info["quality"] = "LOW"

        return (
            after_resized,
            np.ones(
                (height, width),
                dtype=np.uint8
            ),
            info
        )

    # ---------------------------------------------------------
    # 6. Warp AFTER image onto BEFORE geometry
    # ---------------------------------------------------------

    aligned_after = cv2.warpPerspective(
        after_resized,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # ---------------------------------------------------------
    # 7. Create valid overlap mask
    # ---------------------------------------------------------

    source_valid = np.ones(
        (height, width),
        dtype=np.uint8
    ) * 255

    valid_mask = cv2.warpPerspective(
        source_valid,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    valid_mask = (
        valid_mask > 0
    ).astype(np.uint8)

    overlap_ratio = (
        np.count_nonzero(valid_mask)
        / valid_mask.size
        * 100
    )

    info["overlap_ratio"] = overlap_ratio

    # ---------------------------------------------------------
    # 8. Final quality label
    # ---------------------------------------------------------

    if (
        inlier_ratio >= 0.60
        and (
            reprojection_error is None
            or reprojection_error <= 3.0
        )
    ):
        quality = "HIGH"

    elif inlier_ratio >= 0.40:
        quality = "MODERATE"

    else:
        quality = "ACCEPTED"

    info["quality"] = quality
    info["accepted"] = True

    info["reason"] = (
        "The AFTER observation was geometrically aligned "
        "to the BEFORE observation using reliable feature "
        "correspondences."
    )

    return (
        aligned_after,
        valid_mask,
        info
    )