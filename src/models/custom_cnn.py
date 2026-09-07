import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=1)
    )

class CustomCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 64, 3),
            conv_block(64, 128, 3),
            conv_block(128, 256, 3),
            conv_block(256, 512, 3),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_custom_cnn(in_channels=3, num_classes=10):
    return CustomCNN(in_channels=in_channels, num_classes=num_classes)
