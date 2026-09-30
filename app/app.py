import sys
import os

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


OBSERVATION_IMAGE_WIDTH = 450
RESULT_IMAGE_WIDTH = 450
IMPACT_GRAPH_WIDTH = 550


def display_image(image, caption=None, width=450):
    if image is None:
        return
    st.image(image, caption=caption, width=width)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT MODULES
# ============================================================

from src.change_detection import (
    load_image,
    resize_images,
    calculate_change_map,
    create_overlay
)

from src.intelligence import (
    calculate_change_concentration,
    create_hotspot_map,
    classify_change_level,
    generate_evidence_chain,
    create_research_interpretation,
    generate_event_hypotheses
)

from src.registration import (
    register_images
)

from src.evidence import (
    calculate_change_quality
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SpacePulse",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛰️ SpacePulse")

st.subheader(
    "Explainable Earth Observation Intelligence"
)

st.write(
    """
    SpacePulse is a research-oriented Earth observation
    prototype for analyzing temporal changes in satellite
    imagery and building an interpretable evidence chain.
    """
)

st.caption(
    "Research Prototype • Earth Observation • Computer Vision • AI"
)

st.divider()


# ============================================================
# 1. OBSERVE
# ============================================================

st.header("🛰️ 1. OBSERVE")

st.write(
    """
    Upload two observations of the same geographical region
    captured at different points in time.
    """
)

col1, col2 = st.columns(2)


with col1:

    st.subheader("Before Observation")

    before_file = st.file_uploader(
        "Upload the BEFORE satellite image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff"
        ],
        key="before"
    )


with col2:

    st.subheader("After Observation")

    after_file = st.file_uploader(
        "Upload the AFTER satellite image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff"
        ],
        key="after"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

if before_file and after_file:

    try:

        # ====================================================
        # LOAD IMAGES
        # ====================================================

        before_image = load_image(
            before_file
        )

        after_image = load_image(
            after_file
        )

        before_image, after_image = resize_images(
            before_image,
            after_image
        )

        st.success(
            "✅ Both Earth observations loaded successfully."
        )

        st.divider()


        # ====================================================
        # EARTH OBSERVATION PAIR
        # ====================================================

        st.header(
            "🌍 Earth Observation Pair"
        )

        col1, col2 = st.columns(2)

        with col1:

            display_image(
                before_image,
                "Before observation",
                OBSERVATION_IMAGE_WIDTH
            )

        with col2:

            display_image(
                after_image,
                "After observation",
                OBSERVATION_IMAGE_WIDTH
            )


        st.divider()


        # ====================================================
        # 2. ALIGN
        # ====================================================

        st.header(
            "🧭 2. ALIGN"
        )

        st.write(
            """
            Before detecting temporal change, SpacePulse attempts
            to align the AFTER observation with the BEFORE
            observation. This reduces false changes caused by
            geometric misalignment.
            """
        )

        register_button = st.button(
            "🧭 Register Earth Observations",
            type="secondary"
        )


        if register_button:

            with st.spinner(
                "Matching satellite features and estimating alignment..."
            ):

                (
                    aligned_after,
                    valid_mask,
                    registration_info
                ) = register_images(
                    before_image,
                    after_image
                )

            st.session_state[
                "aligned_after"
            ] = aligned_after

            st.session_state[
                "valid_mask"
            ] = valid_mask

            st.session_state[
                "registration_info"
            ] = registration_info


        # ====================================================
        # REGISTRATION RESULT
        # ====================================================

        if (
            "aligned_after" in st.session_state
            and
            "registration_info" in st.session_state
        ):

            aligned_after = (
                st.session_state[
                    "aligned_after"
                ]
            )

            valid_mask = (
                st.session_state[
                    "valid_mask"
                ]
            )

            registration_info = (
                st.session_state[
                    "registration_info"
                ]
            )


            st.subheader(
                "🔬 Registration Quality"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "BEFORE Features",
                    registration_info[
                        "keypoints_before"
                    ]
                )


            with col2:

                st.metric(
                    "AFTER Features",
                    registration_info[
                        "keypoints_after"
                    ]
                )


            with col3:

                st.metric(
                    "Reliable Matches",
                    registration_info[
                        "good_matches"
                    ]
                )


            with col4:

                st.metric(
                    "RANSAC Inliers",
                    registration_info[
                        "inliers"
                    ]
                )


            if registration_info[
                "accepted"
            ]:

                st.success(
                    f"""
                    ✅ Registration ACCEPTED

                    **Quality:** {registration_info["quality"]}

                    **Inlier ratio:** \
                    {registration_info["inlier_ratio"] * 100:.1f}%

                    **Valid overlap:** \
                    {registration_info["overlap_ratio"]:.1f}%
                    """
                )

            else:

                st.warning(
                    f"""
                    ⚠️ Registration NOT ACCEPTED

                    {registration_info["reason"]}

                    The original AFTER observation is being
                    retained instead of applying an unreliable
                    transformation.
                    """
                )


            if registration_info[
                "reprojection_error"
            ] is not None:

                st.caption(
                    "Average reprojection error: "
                    f"{registration_info['reprojection_error']:.2f} pixels"
                )


            st.subheader(
                "🛰️ Registration Result"
            )

            col1, col2 = st.columns(2)

            with col1:

                display_image(
                    before_image,
                    "Reference — BEFORE",
                    RESULT_IMAGE_WIDTH
                )

            with col2:

                display_image(
                    aligned_after,
                    "Registered — AFTER",
                    RESULT_IMAGE_WIDTH
                )


            st.divider()


            # =================================================
            # 3. DETECT
            # =================================================

            st.header(
                "🔬 3. DETECT"
            )

            st.write(
                """
                SpacePulse compares the aligned observations
                to identify temporal visual changes.
                """
            )


            threshold = st.slider(
                "Change sensitivity",
                min_value=5,
                max_value=100,
                value=30,
                step=5
            )


            analyze = st.button(
                "🔎 Analyze Earth Change",
                type="primary"
            )


            if analyze:

                # =============================================
                # CHANGE DETECTION
                # =============================================

                (
                    change_mask,
                    raw_change_percentage,
                    difference
                ) = calculate_change_map(
                    before_image,
                    aligned_after,
                    threshold
                )


                # =============================================
                # VALID OVERLAP
                # =============================================

                if valid_mask is not None:

                    change_mask = np.where(
                        valid_mask > 0,
                        change_mask,
                        0
                    ).astype(
                        change_mask.dtype
                    )


                valid_pixels = np.count_nonzero(
                    valid_mask
                )

                changed_pixels = np.count_nonzero(
                    change_mask
                )


                if valid_pixels > 0:

                    change_percentage = (
                        changed_pixels
                        / valid_pixels
                        * 100
                    )

                else:

                    change_percentage = 0.0


                # =============================================
                # OVERLAY
                # =============================================

                overlay = create_overlay(
                    aligned_after,
                    change_mask
                )


                # =============================================
                # SPATIAL INTELLIGENCE
                # =============================================

                concentration_data = (
                    calculate_change_concentration(
                        change_mask
                    )
                )

                concentration = (
                    concentration_data[
                        "concentration"
                    ]
                )

                components = (
                    concentration_data[
                        "components"
                    ]
                )


                change_level = classify_change_level(
                    change_percentage
                )


                # =============================================
                # EVIDENCE CHAIN
                # =============================================

                evidence = generate_evidence_chain(
                    change_percentage,
                    concentration
                )


                interpretation = (
                    create_research_interpretation(
                        change_percentage,
                        concentration
                    )
                )


                hypotheses = (
                    generate_event_hypotheses(
                        change_percentage,
                        concentration,
                        components
                    )
                )


                # =============================================
                # NEW: EVIDENCE VALIDATION
                # =============================================

                evidence_analysis = (
                    calculate_change_quality(
                        change_mask=change_mask,
                        difference=difference,
                        valid_mask=valid_mask,
                        registration_info=registration_info
                    )
                )


                # =================================================
                # DETECTED CHANGE
                # =================================================

                st.header(
                    "📊 Detected Change"
                )


                col1, col2, col3, col4 = st.columns(4)


                with col1:

                    st.metric(
                        "Changed Area",
                        f"{change_percentage:.2f}%"
                    )


                with col2:

                    st.metric(
                        "Temporal Change Intensity",
                        change_level
                    )


                with col3:

                    st.metric(
                        "Change Regions",
                        components
                    )


                with col4:

                    st.metric(
                        "Spatial Concentration",
                        f"{concentration:.1f}%"
                    )


                st.caption(
                    f"""
                    Analysis performed using sensitivity threshold
                    **{threshold}** on the registered observation pair.
                    """
                )


                st.divider()


                # =================================================
                # CHANGE SIGNATURE
                # =================================================

                st.header(
                    "🗺️ Change Signature"
                )


                col1, col2 = st.columns(2)


                with col1:

                    display_image(
                        change_mask,
                        "Registered binary change mask",
                        RESULT_IMAGE_WIDTH
                    )


                with col2:

                    display_image(
                        overlay,
                        "Detected changes on registered AFTER observation",
                        RESULT_IMAGE_WIDTH
                    )


                st.divider()


                # =================================================
                # 4. EVIDENCE VALIDATION
                # =================================================

                st.header(
                    "🧠 4. EVIDENCE VALIDATION"
                )

                st.write(
                    """
                    SpacePulse evaluates whether the detected
                    temporal change has supporting evidence from
                    registration quality, spatial structure,
                    pixel-level difference and image boundaries.
                    """
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Evidence Strength",
                        evidence_analysis[
                            "evidence_strength"
                        ]
                    )


                with col2:

                    st.metric(
                        "Heuristic Evidence Score",
                        f"{evidence_analysis['evidence_score']:.1f}/100"
                    )


                with col3:

                    st.metric(
                        "Registration Quality",
                        evidence_analysis[
                            "registration_quality"
                        ]
                    )


                st.caption(
                    """
                    The evidence score is a transparent heuristic
                    based on measurable image properties. It is not
                    a probability that a disaster occurred.
                    """
                )


                # -------------------------------------------------
                # Evidence components
                # -------------------------------------------------

                st.subheader(
                    "🔬 Evidence Components"
                )


                evidence_rows = [

                    {
                        "Evidence": "Image Registration",
                        "Result": (
                            evidence_analysis[
                                "registration_quality"
                            ]
                        )
                    },

                    {
                        "Evidence": "Spatial Coherence",
                        "Result": (
                            f"{evidence_analysis['largest_component_ratio']:.1f}% "
                            "largest-component share"
                        )
                    },

                    {
                        "Evidence": "Difference Separation",
                        "Result": (
                            f"{evidence_analysis['difference_separation']:.1f}%"
                        )
                    },

                    {
                        "Evidence": "Fragmentation",
                        "Result": (
                            f"{evidence_analysis['fragmentation_ratio']:.1f}% "
                            "small-component area"
                        )
                    },

                    {
                        "Evidence": "Boundary Effect",
                        "Result": (
                            f"{evidence_analysis['border_change_ratio']:.1f}% "
                            "of changed pixels"
                        )
                    }

                ]


                st.dataframe(
                    evidence_rows,
                    width="stretch",
                    hide_index=True
                )


                # -------------------------------------------------
                # False-positive analysis
                # -------------------------------------------------

                st.subheader(
                    "⚠️ False-Positive Analysis"
                )


                warnings = (
                    evidence_analysis[
                        "warnings"
                    ]
                )


                for warning in warnings:

                    if (
                        "No major" in warning
                    ):

                        st.success(
                            f"✓ {warning}"
                        )

                    else:

                        st.warning(
                            f"⚠️ {warning}"
                        )


                st.info(
                    evidence_analysis[
                        "interpretation"
                    ]
                )


                st.divider()


                # =================================================
                # 5. REASON
                # =================================================

                st.header(
                    "🧠 5. REASON"
                )


                st.write(
                    """
                    SpacePulse does not immediately assign a
                    disaster label. It combines temporal change,
                    spatial structure and evidence-quality signals
                    before presenting competing hypotheses.
                    """
                )


                st.subheader(
                    "🔬 Evidence Chain"
                )


                for item in evidence:

                    st.write(
                        f"✓ {item}"
                    )


                st.subheader(
                    "🔎 Competing Event Hypotheses"
                )


                st.caption(
                    """
                    These are evidence-based research hypotheses,
                    not confirmed disaster classifications.
                    """
                )


                for hypothesis in hypotheses:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### 🔎 {hypothesis['event']}"
                        )

                        st.write(
                            hypothesis["reason"]
                        )

                        st.caption(
                            f"Status: {hypothesis['status']}"
                        )


                st.divider()


                # =================================================
                # 6. IMPACT
                # =================================================

                st.header(
                    "🌍 6. IMPACT"
                )


                st.write(
                    """
                    The registered observation is divided into
                    spatial regions to identify areas where change
                    is concentrated.
                    """
                )


                hotspot_map = create_hotspot_map(
                    change_mask,
                    grid_size=6
                )


                fig, ax = plt.subplots(
                    figsize=(7, 4.5)
                )


                image = ax.imshow(
                    hotspot_map,
                    interpolation="nearest"
                )


                ax.set_title(
                    "Spatial Change Hotspot Map",
                    fontsize=12
                )

                ax.set_xlabel(
                    "Spatial Region",
                    fontsize=9
                )

                ax.set_ylabel(
                    "Spatial Region",
                    fontsize=9
                )


                ax.tick_params(axis="both", labelsize=8)

                fig.colorbar(
                    image,
                    ax=ax,
                    label="Changed Area (%)",
                    fraction=0.046,
                    pad=0.04
                )

                fig.tight_layout()


                st.pyplot(
                    fig,
                    width=IMPACT_GRAPH_WIDTH
                )


                plt.close(fig)


                # -------------------------------------------------
                # Highest change regions
                # -------------------------------------------------

                st.subheader(
                    "🔥 Highest Change Regions"
                )


                flat_values = (
                    hotspot_map.flatten()
                )


                top_indices = np.argsort(
                    flat_values
                )[::-1][:5]


                hotspot_rows = []


                for index in top_indices:

                    row = (
                        index
                        // hotspot_map.shape[1]
                    )

                    col = (
                        index
                        % hotspot_map.shape[1]
                    )


                    hotspot_rows.append(
                        {
                            "Region": (
                                f"R{row + 1}-C{col + 1}"
                            ),
                            "Changed Area (%)": round(
                                hotspot_map[
                                    row,
                                    col
                                ],
                                2
                            )
                        }
                    )


                st.dataframe(
                    hotspot_rows,
                    width="stretch",
                    hide_index=True
                )


                st.divider()


                # =================================================
                # 7. EXPLAIN
                # =================================================

                st.header(
                    "🔎 7. EXPLAIN"
                )


                st.subheader(
                    "Why was this region flagged?"
                )


                st.markdown(
                    f"""
                    **Temporal change**

                    `{change_percentage:.2f}%` of the valid
                    registered observation area was identified
                    as changed.

                    **Spatial evidence**

                    The largest connected change structure represents
                    approximately `{concentration:.1f}%` of the detected
                    changed area.

                    **Registration evidence**

                    - Reliable matches: `{registration_info["good_matches"]}`
                    - RANSAC inliers: `{registration_info["inliers"]}`
                    - Inlier ratio: `{registration_info["inlier_ratio"] * 100:.1f}%`
                    - Registration quality: `{registration_info["quality"]}`

                    **Evidence validation**

                    - Evidence strength: `{evidence_analysis["evidence_strength"]}`
                    - Heuristic evidence score: `{evidence_analysis["evidence_score"]:.1f}/100`
                    - Boundary change ratio: `{evidence_analysis["border_change_ratio"]:.1f}%`
                    - Fragmentation ratio: `{evidence_analysis["fragmentation_ratio"]:.1f}%`

                    **Interpretation**

                    {evidence_analysis["interpretation"]}
                    """
                )


                # =================================================
                # RESEARCH LIMITATION
                # =================================================

                st.warning(
                    """
                    ⚠️ **Research limitation**

                    The current SpacePulse baseline detects visual
                    temporal change. It does not establish that a
                    disaster occurred.

                    Image registration and evidence validation reduce
                    important sources of false change, but reliable
                    disaster assessment still requires validated
                    Earth-observation datasets, suitable spectral or
                    radar data, ground truth, and independent
                    verification.
                    """
                )


                st.divider()


                # =================================================
                # RESEARCH ROADMAP
                # =================================================

                st.header(
                    "🔬 Research Roadmap"
                )


                st.write(
                    """
                    SpacePulse now combines geometric registration,
                    temporal change detection, spatial reasoning and
                    evidence-quality analysis.
                    """
                )


                st.code(
                    """
RGB Satellite Images
        ↓
Preprocessing
        ↓
Feature Matching
        ↓
RANSAC Image Registration
        ↓
Aligned Image Pair
        ↓
Pixel Difference Baseline
        ↓
Spatial Change Intelligence
        ↓
Evidence Validation
        ↓
False-Positive Analysis
        ↓
Evidence-Based Hypotheses
        ↓
Sentinel-2 Multispectral Analysis
        ↓
NDVI / NDWI / NBR
        ↓
Sentinel-1 SAR
        ↓
Multimodal Fusion
        ↓
Deep Learning Change Detection
        ↓
Explainable Disaster Intelligence
                """,
                    language="text"
                )


    except Exception as error:

        st.error(
            f"Analysis failed: {error}"
        )

        st.exception(error)


else:

    st.info(
        "Upload both BEFORE and AFTER observations to begin."
    )