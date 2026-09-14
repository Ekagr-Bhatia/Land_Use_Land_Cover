"""
TERRA / VISION — Geospatial Inference Engine
Production FastAPI inference service for 13-band Sentinel-2 EuroSAT checkpoints.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import rasterio
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from torchvision import transforms

# Ensure src directory is in sys.path so model imports work cleanly
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.custom_cnn import build_custom_cnn
from models.mobilenet import build_mobilenet
from models.resnet import build_resnet

ROOT = SRC_DIR.parent
DATA = ROOT / "data" / "dataset_allBands"
WEIGHTS = ROOT / "saved_models" / "allbands"
WEB = ROOT / "web"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL = "resnet_pretrained_best"

MODEL_SPECS = {
    "resnet_pretrained_best": {
        "name": "ResNet-50",
        "type": "resnet",
        "training": "ImageNet Pretrained Stem Adaptation",
        "description": "Deep 50-layer residual network with modified 13-channel stem initialized from RGB channel weights.",
        "params": "23.5M parameters"
    },
    "resnet_scratch_best": {
        "name": "ResNet-50",
        "type": "resnet",
        "training": "Trained from Scratch",
        "description": "50-layer residual network initialized with random weights, trained directly on 13-band reflectance.",
        "params": "23.5M parameters"
    },
    "mobilenet_pretrained_best": {
        "name": "MobileNetV3 Large",
        "type": "mobilenet",
        "training": "ImageNet Pretrained Stem Adaptation",
        "description": "Efficient lightweight architecture with inverted residual blocks and squeeze-and-excitation attention.",
        "params": "4.2M parameters"
    },
    "mobilenet_scratch_best": {
        "name": "MobileNetV3 Large",
        "type": "mobilenet",
        "training": "Trained from Scratch",
        "description": "Lightweight MobileNetV3 Large trained from scratch on 13-band Sentinel-2 reflectance.",
        "params": "4.2M parameters"
    },
    "custom_scratch_best": {
        "name": "Custom CNN",
        "type": "custom",
        "training": "Trained from Scratch",
        "description": "4-block convolutional network with batch normalization, max pooling, and adaptive average pooling.",
        "params": "1.6M parameters"
    },
}

SENTINEL2_BANDS = [
    {"index": 1, "name": "B01", "desc": "Coastal Aerosol (443 nm)"},
    {"index": 2, "name": "B02", "desc": "Blue (490 nm)"},
    {"index": 3, "name": "B03", "desc": "Green (560 nm)"},
    {"index": 4, "name": "B04", "desc": "Red (665 nm)"},
    {"index": 5, "name": "B05", "desc": "Vegetation Red Edge 1 (705 nm)"},
    {"index": 6, "name": "B06", "desc": "Vegetation Red Edge 2 (740 nm)"},
    {"index": 7, "name": "B07", "desc": "Vegetation Red Edge 3 (783 nm)"},
    {"index": 8, "name": "B08", "desc": "NIR Broad (842 nm)"},
    {"index": 9, "name": "B08A", "desc": "NIR Narrow (865 nm)"},
    {"index": 10, "name": "B09", "desc": "Water Vapour (945 nm)"},
    {"index": 11, "name": "B10", "desc": "SWIR – Cirrus (1375 nm)"},
    {"index": 12, "name": "B11", "desc": "SWIR 1 (1610 nm)"},
    {"index": 13, "name": "B12", "desc": "SWIR 2 (2190 nm)"},
]


def get_class_labels() -> List[str]:
    """Loads true class mapping from dataset label_map.json."""
    map_file = DATA / "label_map.json"
    if not map_file.is_file():
        return [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
            "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
        ]
    with open(map_file, encoding="utf-8") as f:
        mapping = json.load(f)
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Returns available checkpoints discovered in saved_models/allbands."""
    available = {}
    for key, spec in MODEL_SPECS.items():
        weight_file = WEIGHTS / f"{key}.pth"
        if weight_file.is_file():
            available[key] = {
                "name": spec["name"],
                "type": spec["type"],
                "training": spec["training"],
                "description": spec["description"],
                "params": spec["params"],
                "file": weight_file.name,
                "size_mb": round(weight_file.stat().st_size / (1024 * 1024), 2)
            }
    return available


@lru_cache(maxsize=5)
def load_model(model_key: str):
    """Loads and caches trained PyTorch checkpoint in evaluation mode."""
    available = get_available_models()
    if model_key not in available:
        raise KeyError(f"Model checkpoint '{model_key}' is not available.")

    model_type = MODEL_SPECS[model_key]["type"]
    num_classes = len(get_class_labels())

    if model_type == "resnet":
        model = build_resnet(in_channels=13, num_classes=num_classes, pretrained=False)
    elif model_type == "mobilenet":
        model = build_mobilenet(in_channels=13, num_classes=num_classes, pretrained=False)
    elif model_type == "custom":
        model = build_custom_cnn(in_channels=13, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")

    checkpoint_path = WEIGHTS / f"{model_key}.pth"
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def make_rgb_composite(bands_13_array: np.ndarray) -> str:
    """
    Creates a true-color RGB composite image from 13-band Sentinel-2 data.
    Sentinel-2 bands: Band 4 = Red, Band 3 = Green, Band 2 = Blue (1-indexed).
    Array is shape (H, W, 13) with values scaled by 10000.
    Returns base64 encoded PNG data URI.
    """
    try:
        # 0-indexed: B4 is index 3, B3 is index 2, B2 is index 1
        rgb = bands_13_array[:, :, [3, 2, 1]].copy()
        # Scale for visual display
        rgb_stretched = np.clip(rgb * 2.5, 0.0, 1.0) * 255.0
        im = Image.fromarray(rgb_stretched.astype(np.uint8)).resize((256, 256), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception:
        return ""


def run_inference(raw_bytes: bytes, model_key: str) -> Dict[str, Any]:
    """
    Executes production inference pipeline on raw GeoTIFF bytes.
    Validates 13 bands, scales reflectance, normalizes, and computes softmax.
    """
    start_time = time.perf_counter()

    try:
        with rasterio.MemoryFile(raw_bytes) as mem_file:
            with mem_file.open() as src:
                if src.count != 13:
                    raise ValueError(
                        f"Expected 13 Sentinel-2 multispectral bands, but received {src.count} band(s). "
                        "Please upload a 13-band EuroSAT GeoTIFF (.tif/.tiff)."
                    )
                orig_h, orig_w = src.height, src.width
                raw_data = src.read()
                # Transpose from (13, H, W) to (H, W, 13)
                arr = np.transpose(raw_data.astype(np.float32) / 10000.0, (1, 2, 0))
    except rasterio.errors.RasterioError as e:
        raise ValueError(f"Failed to read GeoTIFF: {str(e)}") from e

    # Generate RGB visualization preview
    preview_data_url = make_rgb_composite(arr)

    # Ensure 64x64 dimensions matching EuroSAT training pipeline
    if arr.shape[:2] != (64, 64):
        arr = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_LINEAR)

    # Apply training normalization: mean=[0.5]*13, std=[0.5]*13
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (13, 64, 64)
    normalizer = transforms.Normalize(mean=[0.5] * 13, std=[0.5] * 13)
    input_tensor = normalizer(tensor).unsqueeze(0).to(DEVICE)

    # Model inference
    model = load_model(model_key)
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits[0], dim=0).cpu().tolist()

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    labels = get_class_labels()
    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)

    return {
        "model": model_key,
        "model_name": MODEL_SPECS.get(model_key, {}).get("name", model_key),
        "predicted_class": ranked[0][0],
        "confidence": round(ranked[0][1] * 100, 2),
        "probabilities": [
            {
                "label": label,
                "percentage": round(prob * 100, 2),
                "probability": prob
            }
            for label, prob in ranked
        ],
        "preview": preview_data_url,
        "metadata": {
            "bands": 13,
            "dimensions": [orig_w, orig_h],
            "input_shape": [13, 64, 64],
            "scaling": "reflectance / 10000.0",
            "latency_ms": latency_ms,
            "device": str(DEVICE)
        }
    }


def resolve_sample_path(sample_id: str) -> Path:
    """Safely resolves sample file path within DATA directory."""
    path = (DATA / sample_id).resolve()
    if DATA.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Sample scene not found.")
    return path


def discover_sample_scenes() -> List[Dict[str, Any]]:
    """Discovers representative test samples for each class in dataset_allBands."""
    samples = []
    for label in get_class_labels():
        class_dir = DATA / label
        if class_dir.is_dir():
            tifs = sorted(class_dir.glob("*.tif"))
            if tifs:
                sample_file = tifs[0]
                rel_path = sample_file.relative_to(DATA).as_posix()
                samples.append({
                    "id": rel_path,
                    "label": label,
                    "filename": sample_file.name,
                    "preview": f"/samples/{rel_path}/preview"
                })
    return samples


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warms default model on startup and clears cache on shutdown."""
    try:
        load_model(DEFAULT_MODEL)
    except Exception as e:
        print(f"Warning: Could not pre-warm default model: {e}")
    yield
    load_model.cache_clear()


app = FastAPI(
    title="TERRA / VISION — Geospatial Inference Engine",
    description="Production PyTorch inference API for 13-band Sentinel-2 EuroSAT Land Use / Land Cover classification.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """System health check and hardware discovery."""
    return {
        "status": "online",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "default_model": DEFAULT_MODEL,
        "available_models": list(get_available_models().keys())
    }


@app.get("/model-info")
async def model_info():
    """Returns technical metadata, architectures, input shapes, and class list."""
    return {
        "system": "TERRA / VISION",
        "subtitle": "Understanding the surface of our planet through machine vision.",
        "framework": f"PyTorch {torch.__version__}",
        "device": str(DEVICE),
        "default_model": DEFAULT_MODEL,
        "models": get_available_models(),
        "input": {
            "bands": 13,
            "size": [64, 64],
            "format": "GeoTIFF (13-band Sentinel-2 MSI)",
            "scaling": "uint16 reflectance / 10000.0",
            "normalization": {
                "mean": [0.5] * 13,
                "std": [0.5] * 13
            }
        },
        "bands": SENTINEL2_BANDS,
        "classes": get_class_labels(),
        "dataset": "EuroSAT Sentinel-2 Multispectral Dataset (13 Spectral Bands)"
    }


@app.post("/predict")
async def predict_geotiff(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL, description="Checkpoint key to use for inference")
):
    """Upload and classify a 13-band Sentinel-2 GeoTIFF (.tif/.tiff)."""
    available = get_available_models()
    if model not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model checkpoint '{model}'. Available: {list(available.keys())}"
        )

    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=415,
            detail="Unsupported format. Please upload a 13-band Sentinel-2 GeoTIFF (.tif or .tiff)."
        )

    try:
        content = await file.read()
        return run_inference(content, model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}") from e


@app.get("/samples")
async def list_samples():
    """Lists representative samples across all 10 EuroSAT classes."""
    return {"samples": discover_sample_scenes()}


@app.post("/samples/{sample_id:path}/predict")
async def predict_sample_scene(
    sample_id: str,
    model: str = Query(DEFAULT_MODEL, description="Checkpoint key to use for inference")
):
    """Runs real inference on a sample scene from the dataset."""
    available = get_available_models()
    if model not in available:
        raise HTTPException(status_code=400, detail=f"Unknown model checkpoint '{model}'.")

    sample_path = resolve_sample_path(sample_id)
    raw_bytes = sample_path.read_bytes()
    return run_inference(raw_bytes, model)


@app.get("/samples/{sample_id:path}/preview")
async def sample_preview(sample_id: str):
    """Generates RGB true-color composite PNG preview for a sample scene."""
    sample_path = resolve_sample_path(sample_id)
    with rasterio.open(sample_path) as src:
        # Read Red (Band 4), Green (Band 3), Blue (Band 2)
        rgb = src.read([4, 3, 2]).astype(np.float32) / 10000.0
        rgb = np.moveaxis(rgb, 0, -1)
        rgb_stretched = np.clip(rgb * 2.5, 0.0, 1.0) * 255.0

    img = Image.fromarray(rgb_stretched.astype(np.uint8)).resize((256, 256), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# Mount static web frontend assets
if (WEB / "index.html").is_file():
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/style.css", include_in_schema=False)
    async def style_css():
        return FileResponse(WEB / "style.css", media_type="text/css")

    @app.get("/app.js", include_in_schema=False)
    async def app_js():
        return FileResponse(WEB / "app.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  TERRA / VISION — Earth Observation Machine Intelligence")
    print(f"  Device: {DEVICE}")
    print("  Server launching on: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
