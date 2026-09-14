# TERRA / VISION — Geospatial Earth Observation Machine Intelligence

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![Sentinel-2](https://img.shields.io/badge/Satellite-Sentinel--2%20MSI-0072bc.svg)](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)
[![EuroSAT](https://img.shields.io/badge/Dataset-EuroSAT%20(13--Band)-orange.svg)](https://github.com/phelber/eurosat)

**TERRA / VISION** is a production inference platform and machine learning classification application for Earth Observation satellite imagery. The models ingest all **13 multispectral reflectance bands** acquired by the ESA Copernicus Sentinel-2 MultiSpectral Instrument (MSI) to classify scenes into 10 distinct Land Use and Land Cover (LULC) categories.

---

## Key Features

- **13-Band Multispectral Pipeline**: Full-spectral ingestion of Sentinel-2 bands (Coastal Aerosol, Blue, Green, Red, Vegetation Red Edge 1–3, NIR, Water Vapour, SWIR 1–2).
- **Multiple Trained Backbones**:
  - `ResNet-50 (ImageNet Pretrained Stem Adaptation)` [23.5M parameters]
  - `ResNet-50 (Trained from Scratch)` [23.5M parameters]
  - `MobileNetV3 Large (ImageNet Pretrained Stem Adaptation)` [4.2M parameters]
  - `MobileNetV3 Large (Trained from Scratch)` [4.2M parameters]
  - `Custom CNN (4-Block Multispectral Deep Architecture)` [1.6M parameters]
- **Real-Time Calibration & Normalization**: uint16 reflectance scaling by $\frac{1}{10000.0}$ with per-channel normalization ($\mu=0.5, \sigma=0.5$).
- **True-Color Composite Visualizer**: Dynamically generates calibrated RGB composite previews (Bands 4, 3, 2) from uploaded 13-band GeoTIFFs.
- **Editorial Single-Page Application**: Clean, responsive interface adhering to geospatial research aesthetics (Editorial typography, INK/BEIGE/ORANGE palette, topographic contours, live telemetry, and probability distributions).

---

## Quick Start

### 1. Environment Setup
Activate your virtual environment and verify dependencies:
```bash
# Windows
.\venv\Scripts\activate

# Install requirements if needed
pip install -r requirements.txt
```

### 2. Launch the Application
Run the dedicated launcher:
```bash
python run_app.py
```
Or start via Uvicorn:
```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

---

## API Documentation

The FastAPI service exposes RESTful endpoints with interactive OpenAPI documentation available at `http://127.0.0.1:8000/docs`:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | Hardware discovery, device acceleration (CUDA/CPU), service readiness. |
| `/model-info` | `GET` | Discovered checkpoints, architectures, 13 Sentinel-2 bands, and class labels. |
| `/predict` | `POST` | Upload and classify a 13-band Sentinel-2 GeoTIFF (`.tif` / `.tiff`). |
| `/samples` | `GET` | Discovers representative 13-band scenes across all 10 EuroSAT classes. |
| `/samples/{sample_id}/preview` | `GET` | Generates a true-color (B4-B3-B2) RGB composite PNG preview. |
| `/samples/{sample_id}/predict` | `POST` | Executes inference directly on a dataset sample scene. |

---

## Project Structure

```
LULC_Unified/
├── data/
│   └── dataset_allBands/       # 27,000 13-band Sentinel-2 GeoTIFF scenes (10 classes)
│       ├── label_map.json       # Canonical class index mapping
│       ├── train.csv            # 18,900 training scenes
│       ├── validation.csv       # 5,400 validation scenes
│       └── test.csv             # 2,700 test scenes
├── saved_models/
│   └── allbands/                # Trained PyTorch model checkpoints (.pth)
├── src/
│   ├── app.py                   # Production FastAPI inference engine & static server
│   ├── dataset.py               # Multispectral PyTorch Dataset loader
│   ├── evaluate.py              # Scikit-learn evaluation & confusion matrix harness
│   ├── predict.py               # CLI single-image prediction script
│   ├── train.py                 # Multi-GPU / Multi-worker training harness
│   └── models/
│       ├── custom_cnn.py        # 4-block deep CNN
│       ├── mobilenet.py         # MobileNetV3 Large with 13-band conv1 stem
│       └── resnet.py            # ResNet-50 with 13-band conv1 stem
├── web/
│   ├── index.html               # Semantic HTML5 single-page application
│   ├── style.css                # Visual design system (INK / BEIGE / ORANGE)
│   └── app.js                   # Client-side inference controller & telemetry
├── run_app.py                   # 1-command startup launcher
└── requirements.txt             # Python dependencies
```

---

## Sentinel-2 Spectral Channels (13 Bands)

| Channel | Band Name | Central Wavelength | Spatial Resolution | Primary Geophysical Application |
| :---: | :--- | :---: | :---: | :--- |
| **B01** | Coastal Aerosol | 443 nm | 60 m | Atmospheric correction, coastal bathymetry |
| **B02** | Blue | 490 nm | 10 m | Vegetation discrimination, soil reflectance |
| **B03** | Green | 560 nm | 10 m | Peak vegetation reflectance |
| **B04** | Red | 665 nm | 10 m | Maximum chlorophyll absorption |
| **B05** | Vegetation Red Edge 1 | 705 nm | 20 m | Chlorophyll sensitivity & leaf structure |
| **B06** | Vegetation Red Edge 2 | 740 nm | 20 m | Canopy biomass estimation |
| **B07** | Vegetation Red Edge 3 | 783 nm | 20 m | Leaf Area Index (LAI) retrieval |
| **B08** | NIR (Broad) | 842 nm | 10 m | High-resolution biomass & cell structure |
| **B08A** | NIR (Narrow) | 865 nm | 20 m | Atmospheric water vapor separation |
| **B09** | Water Vapour | 945 nm | 60 m | Atmospheric water column correction |
| **B10** | SWIR – Cirrus | 1375 nm | 60 m | High-altitude cirrus cloud detection |
| **B11** | SWIR 1 | 1610 nm | 20 m | Vegetation moisture, snow/ice separation |
| **B12** | SWIR 2 | 2190 nm | 20 m | Soil mineralogy, urban substrate discrimination |

---

## 10 EuroSAT Land Use / Land Cover Classes

1. **AnnualCrop**: Agricultural croplands with annual seasonal crop rotations.
2. **Forest**: Dense and open coniferous, broadleaved, or mixed forest canopy.
3. **HerbaceousVegetation**: Natural grasslands, meadows, and wild flora.
4. **Highway**: Major road infrastructures, expressways, and transport corridors.
5. **Industrial**: Manufacturing facilities, logistics hubs, and industrial complexes.
6. **Pasture**: Managed grazing meadows and permanent grasslands.
7. **PermanentCrop**: Orchards, vineyards, and perennial agriculture.
8. **Residential**: Dense and suburban urban settlements and housing.
9. **River**: Inland fluvial watercourses, canals, and river systems.
10. **SeaLake**: Open marine waters, natural inland lakes, and reservoirs.
