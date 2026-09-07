import torch
import torch.nn as nn
from torchvision import models

def build_mobilenet(in_channels=3, num_classes=10, pretrained=True):
    """
    Builds a MobileNetV3 (Large) model.
    Adjusts the first convolution layer if in_channels != 3.
    """
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    
    if in_channels != 3:
        # Access the first Conv2dNormActivation module
        first_conv_seq = model.features[0]
        orig_conv = first_conv_seq[0]
        
        new_conv = nn.Conv2d(in_channels, orig_conv.out_channels, 
                             kernel_size=orig_conv.kernel_size, 
                             stride=orig_conv.stride, 
                             padding=orig_conv.padding, 
                             bias=False)
        
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :] = orig_conv.weight.mean(dim=1, keepdim=True)
                
        # Replace the conv
        first_conv_seq[0] = new_conv

    # Modify the final classification layer
    # MobileNetV3 uses a classifier block
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    return model
