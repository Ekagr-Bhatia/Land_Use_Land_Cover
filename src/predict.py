import os
import argparse
import torch
import cv2
import rasterio
import numpy as np
from torchvision import transforms
from dataset import LULCDataset
from models.custom_cnn import build_custom_cnn
from models.resnet import build_resnet
from models.mobilenet import build_mobilenet

def load_image(image_path, in_channels):
    """Loads an image for prediction."""
    if in_channels == 3 and not image_path.endswith('.tif'):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read {image_path}")
        image = image[:, :, ::-1] # BGR to RGB
        image = cv2.resize(image, (64, 64))
    else:
        with rasterio.open(image_path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))
            # Cast to float32 and scale, matching training exactly
            image = image.astype(np.float32) / 10000.0
    return image

def predict(image_path, model_path, model_type, in_channels, num_classes, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build Model
    if model_type == 'custom':
        model = build_custom_cnn(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'resnet':
        model = build_resnet(in_channels=in_channels, num_classes=num_classes, pretrained=False)
    elif model_type == 'mobilenet':
        model = build_mobilenet(in_channels=in_channels, num_classes=num_classes, pretrained=False)
    
    # Load Weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Process Image
    image = load_image(image_path, in_channels)
    
    # Use standard transform (make sure this matches training!)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406] if in_channels == 3 else [0.5]*in_channels,
                             std=[0.229, 0.224, 0.225] if in_channels == 3 else [0.5]*in_channels)
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted_class = torch.max(outputs, 1)
        
    pred_idx = predicted_class.item()
    confidence = probabilities[pred_idx].item()
    
    predicted_label = class_names[pred_idx] if class_names else str(pred_idx)
    
    print(f"Predicted Class: {predicted_label} (Confidence: {confidence*100:.2f}%)")
    return predicted_label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict LULC Class for a single image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model weights')
    parser.add_argument('--model_type', type=str, choices=['custom', 'resnet', 'mobilenet'], default='resnet')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=10)
    
    args = parser.parse_args()
    
    # We don't have access to the dataset's class_names mapping here easily without loading it,
    # so we assume standard EuroSAT alphabetical ordering or pass it manually.
    standard_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
                        'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    
    predict(args.image_path, args.model_path, args.model_type, args.in_channels, args.num_classes, standard_classes)
