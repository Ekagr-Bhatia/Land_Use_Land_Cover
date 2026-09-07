import torch.nn as nn
from torchvision import models

def build_resnet(in_channels=3, num_classes=10, pretrained=True):
    """
    Builds a ResNet50 model.
    Adjusts the first convolution layer if in_channels != 3.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Adapt stem for hyperspectral data
    if in_channels != 3:
        # Get the original weights
        orig_conv = model.conv1
        
        # Create a new conv layer
        new_conv = nn.Conv2d(in_channels, orig_conv.out_channels, 
                             kernel_size=orig_conv.kernel_size, 
                             stride=orig_conv.stride, 
                             padding=orig_conv.padding, 
                             bias=False)
        
        if pretrained:
            # Initialize with the mean of RGB channels to leverage pre-trained knowledge somewhat
            with torch.no_grad():
                new_conv.weight[:, :] = orig_conv.weight.mean(dim=1, keepdim=True)
                
        model.conv1 = new_conv

    # Modify the final classification layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
