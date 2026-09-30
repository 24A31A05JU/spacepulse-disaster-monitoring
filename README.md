# 🛰️ SpacePulse — Explainable Earth Observation Intelligence

SpacePulse is a research-oriented computer-vision prototype for analyzing temporal visual change between two satellite observations of the same geographic region.

## Pipeline

BEFORE + AFTER RGB imagery → preprocessing → ORB feature matching → RANSAC registration → aligned pair → pixel-difference change detection → spatial concentration → evidence validation → false-positive analysis → event hypotheses → hotspot impact analysis → explainable output.

## Current capabilities

- BEFORE/AFTER satellite image upload
- Geometric image registration with ORB + RANSAC homography
- Registration acceptance/rejection based on measurable quality signals
- Pixel-level temporal change detection
- Valid-overlap masking after registration
- Spatial change concentration and connected-component analysis
- Transparent heuristic evidence validation
- False-positive indicators
- Competing event hypotheses instead of automatic disaster claims
- Spatial hotspot analysis
- Explainable research interpretation

## Scientific scope

The current system is an RGB visual temporal-change baseline. It does not directly calculate NDVI, NDWI or NBR because those require appropriate multispectral bands. The evidence score is a heuristic and is not a probability that a disaster occurred.

## Run

```bash
py -m pip install -r requirements.txt
py -m streamlit run app/app.py
```

If your machine uses `python` instead of `py`, replace `py` with `python`.

## Structure

```text
SpacePulse/
├── app/app.py
├── src/
│   ├── __init__.py
│   ├── change_detection.py
│   ├── disaster_analysis.py
│   ├── evidence.py
│   ├── explainability.py
│   ├── intelligence.py
│   ├── preprocessing.py
│   └── registration.py
├── data/
├── models/
├── notebooks/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## Research roadmap

- Sentinel-2 multispectral indices: NDVI, NDWI, NBR
- Sentinel-1 SAR change detection
- Multi-scale temporal change detection
- Optical + SAR fusion
- Deep-learning change detection
- Ground-truth evaluation with precision, recall, F1 and IoU
- Geospatial metadata and event localization

## Important limitation

A detected visual change is not automatically a confirmed disaster. Independent Earth-observation evidence, suitable spectral/radar data, ground truth and validation are required for disaster assessment.
