import cv2
import numpy as np


def calculate_change_concentration(change_mask):

    binary = (
        change_mask > 0
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if num_labels <= 1:

        return {
            "components": 0,
            "largest_component_percentage": 0.0,
            "concentration": 0.0
        }

    areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    total_area = np.sum(areas)

    largest_area = np.max(areas)

    concentration = (
        largest_area /
        total_area *
        100
        if total_area > 0
        else 0
    )

    return {
        "components": len(areas),
        "largest_component_percentage": concentration,
        "concentration": concentration
    }


def create_hotspot_map(
    change_mask,
    grid_size=6
):

    height, width = (
        change_mask.shape
    )

    hotspot_map = []

    cell_height = (
        height // grid_size
    )

    cell_width = (
        width // grid_size
    )

    for row in range(grid_size):

        row_values = []

        for col in range(grid_size):

            y1 = (
                row *
                cell_height
            )

            y2 = (
                (row + 1) *
                cell_height
                if row < grid_size - 1
                else height
            )

            x1 = (
                col *
                cell_width
            )

            x2 = (
                (col + 1) *
                cell_width
                if col < grid_size - 1
                else width
            )

            cell = change_mask[
                y1:y2,
                x1:x2
            ]

            changed = np.count_nonzero(
                cell
            )

            total = cell.size

            percentage = (
                changed /
                total *
                100
                if total > 0
                else 0
            )

            row_values.append(
                percentage
            )

        hotspot_map.append(
            row_values
        )

    return np.array(
        hotspot_map
    )


def classify_change_level(
    change_percentage
):

    if change_percentage < 5:

        return "LOW"

    elif change_percentage < 20:

        return "MODERATE"

    elif change_percentage < 40:

        return "HIGH"

    else:

        return "VERY HIGH"


def generate_evidence_chain(
    change_percentage,
    concentration
):

    evidence = []

    if change_percentage >= 20:

        evidence.append(
            "A substantial portion of the observation changed."
        )

    elif change_percentage >= 5:

        evidence.append(
            "A measurable amount of temporal change was detected."
        )

    else:

        evidence.append(
            "Only limited temporal change was detected."
        )

    if concentration >= 50:

        evidence.append(
            "A large fraction of detected change "
            "is spatially concentrated."
        )

    elif concentration >= 20:

        evidence.append(
            "Some detected changes form "
            "spatially concentrated regions."
        )

    else:

        evidence.append(
            "Detected changes are distributed "
            "across multiple regions."
        )

    return evidence


def create_research_interpretation(
    change_percentage,
    concentration
):

    if (
        change_percentage >= 30
        and concentration >= 50
    ):

        return (
            "Significant and spatially concentrated "
            "surface transformation detected. "
            "This pattern may warrant investigation "
            "as a potential environmental or "
            "disaster-related event."
        )

    elif change_percentage >= 20:

        return (
            "Significant temporal transformation detected. "
            "Further analysis using semantic and "
            "multispectral Earth-observation data "
            "is recommended."
        )

    elif change_percentage >= 5:

        return (
            "Moderate temporal transformation detected. "
            "Additional evidence is required to "
            "determine the cause of the observed change."
        )

    else:

        return (
            "Limited temporal transformation detected. "
            "No strong event-level signal is indicated "
            "by the current baseline analysis."
        )


def generate_event_hypotheses(
    change_percentage,
    concentration,
    components
):

    hypotheses = []

    if (
        change_percentage >= 20
        and concentration >= 20
    ):

        hypotheses.append({

            "event":
                "Flood-like surface transformation",

            "reason":
                "Large temporal change combined "
                "with spatially concentrated regions.",

            "status":
                "Investigate"
        })

    if components >= 50:

        hypotheses.append({

            "event":
                "Distributed environmental change",

            "reason":
                "Change is distributed across "
                "many spatial regions.",

            "status":
                "Investigate"
        })

    if concentration >= 50:

        hypotheses.append({

            "event":
                "Localized high-impact transformation",

            "reason":
                "A substantial proportion of change "
                "is concentrated in a limited region.",

            "status":
                "Investigate"
        })

    if not hypotheses:

        hypotheses.append({

            "event":
                "Low-intensity surface transformation",

            "reason":
                "Current evidence does not show "
                "a strong event-level spatial signal.",

            "status":
                "Monitor"
        })

    return hypotheses