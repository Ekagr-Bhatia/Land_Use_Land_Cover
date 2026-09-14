"""
TERRA / VISION — Launcher Script
Launches the FastAPI backend and serves the web interface on http://127.0.0.1:8000
"""

import os
import sys
from pathlib import Path

# Add src to sys.path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main():
    print("=" * 65)
    print("  TERRA / VISION")
    print("  Geospatial Land Use & Land Cover (LULC) Classification")
    print("  Sentinel-2 · 13 Multispectral Bands · EuroSAT PyTorch Engine")
    print("=" * 65)

    # Check model weights
    weights_dir = ROOT / "saved_models" / "allbands"
    models_found = list(weights_dir.glob("*.pth")) if weights_dir.is_dir() else []
    print(f"\n[+] Checkpoint Directory: {weights_dir}")
    print(f"[+] Discovered Models: {len(models_found)} checkpoints")
    for m in models_found:
        print(f"    - {m.name} ({round(m.stat().st_size / (1024*1024), 2)} MB)")

    print("\n[+] Starting FastAPI & Uvicorn Server...")
    print("    URL: http://127.0.0.1:8000")
    print("    Press Ctrl+C to terminate the server.\n")

    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
