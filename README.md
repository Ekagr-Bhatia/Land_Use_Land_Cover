# Unified LULC Classification

This repository contains unified code for training Land Use and Land Cover (LULC) classification models. It supports both standard RGB images and can be adapted for Hyperspectral/Multispectral datasets.

## Models Available
- `custom`: A CNN built from scratch.
- `resnet`: Transfer learning with ResNet50.
- `mobilenet`: Transfer learning with MobileNetV3 (Large).

## Project Structure
- `data/`: Contains the datasets (`dataset_rgb`).
- `src/`: Core Python files.
  - `models/`: Architectures.
  - `train.py`: Unified training script.
  - `dataset.py`: Data loader that handles N-channel images.
  - `evaluate.py`: Evaluation metrics.
- `notebooks/`: Exploratory Data Analysis and legacy `.ipynb` files.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: 
   ```bash
   python src/train.py --model resnet --epochs 10 --batch_size 32 --in_channels 3
   ```
