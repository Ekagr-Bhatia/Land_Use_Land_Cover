import os
import cv2
import torch
import random
import itertools
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
import rasterio
import numpy as np

class LULCDataset(Dataset):
    def __init__(self, root, transform_status=True, in_channels=3):
        """
        Custom Dataset for LULC Classification.
        
        Args:
            root (str): Root directory of the dataset (e.g., 'data/dataset_rgb/train')
            transform_status (bool): Whether to apply transformations
            in_channels (int): Number of input channels (3 for RGB, 13 for Sentinel-2 MSI, etc.)
        """
        self.root = root
        self.in_channels = in_channels
        self.transform_status = transform_status
        
        # We assume folder structure is root/class_name/image_files
        self.images_paths = [glob(os.path.join(root, folder, '*.*')) for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))] 
        self.images_paths = list(itertools.chain.from_iterable(self.images_paths))
        random.shuffle(self.images_paths)

        self.classes_names = {class_name: label for label, class_name in enumerate(sorted(os.listdir(root))) if os.path.isdir(os.path.join(root, class_name))}
        self.labels = [self.classes_names[os.path.basename(os.path.dirname(path))] for path in self.images_paths]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # These are ImageNet stats. Might need to be updated for hyperspectral data.
            transforms.Normalize(mean=[0.485, 0.456, 0.406] if in_channels == 3 else [0.5]*in_channels,
                                 std=[0.229, 0.224, 0.225] if in_channels == 3 else [0.5]*in_channels)
        ])

    def __len__(self):
        return len(self.images_paths)   
        
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        label = self.labels[index]
        
        # Load image
        if self.in_channels == 3 and not image_path.endswith('.tif'):
            # Assume RGB image (jpg/png)
            image = cv2.imread(image_path)
            if image is not None:
                image = image[:, :, ::-1] # BGR to RGB
            else:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Resize for RGB images if they are not already 64x64
            image = cv2.resize(image, (64, 64))
        else:
            # Hyperspectral/Multispectral (.tif)
            with rasterio.open(image_path) as src:
                # rasterio reads as (Channels, H, W). We need (H, W, Channels) for consistent transforms
                image = src.read()
                image = np.transpose(image, (1, 2, 0)) # -> (H, W, Channels)
            
            # EuroSAT Sentinel-2 .tif images are uint16 (reflectance * 10000).
            # We must cast to float32 so ToTensor() doesn't return a uint16 tensor that crashes Normalize().
            image = image.astype(np.float32) / 10000.0
            
        if self.transform_status:
            image = self.transform(image)
        else:
            # If no transform, we still need it as a tensor.
            # Convert to float32 and normalize (RGB 0-255 -> 0-1, TIF depends but we cast to float)
            image = torch.tensor(image, dtype=torch.float32)
            if image.max() > 1.0 and not image_path.endswith('.tif'):
                image = image / 255.0
            image = image.permute(2, 0, 1)

        return image, torch.tensor(label, dtype=torch.long)
