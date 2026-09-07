import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from dataset import LULCDataset
from models.custom_cnn import build_custom_cnn
from models.resnet import build_resnet
from models.mobilenet import build_mobilenet

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, model_name="model", data_type="rgb", is_pretrained=False):
    model.to(device)
    
    best_acc = 0.0
    save_dir = os.path.join("saved_models", data_type)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_acc = train_correct / total_train
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.2f}%")
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            total_val = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
            val_acc = val_correct / total_val
            print(f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc*100:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                
                # Format name based on pretrained status
                weight_type = "pretrained" if is_pretrained else "scratch"
                if model_name == "custom":
                    weight_type = "scratch" # custom is always from scratch
                    
                save_path = os.path.join(save_dir, f"{model_name}_{weight_type}_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LULC Models")
    parser.add_argument('--model', type=str, choices=['custom', 'resnet', 'mobilenet'], default='custom', help='Model to train')
    parser.add_argument('--data_dir', type=str, default='data/dataset_rgb/train', help='Path to training data')
    parser.add_argument('--data_type', type=str, choices=['rgb', 'allbands'], default='rgb', help='Dataset type to determine save directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels (e.g., 3 for RGB, 13 for multispectral)')
    
    # Pretrained flag (defaults to true for ResNet/MobileNet, but can be turned off)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pretrained ImageNet weights')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='Train from scratch')
    parser.set_defaults(pretrained=True)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    train_dataset = LULCDataset(root=args.data_dir, transform_status=True, in_channels=args.in_channels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    num_classes = len(train_dataset.classes_names)
    print(f"Detected {num_classes} classes.")
    
    # Build Model
    if args.model == 'custom':
        model = build_custom_cnn(in_channels=args.in_channels, num_classes=num_classes)
        args.pretrained = False # Custom cannot be pretrained
    elif args.model == 'resnet':
        model = build_resnet(in_channels=args.in_channels, num_classes=num_classes, pretrained=args.pretrained)
    elif args.model == 'mobilenet':
        model = build_mobilenet(in_channels=args.in_channels, num_classes=num_classes, pretrained=args.pretrained)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    train(model, train_loader, val_loader=None, criterion=criterion, optimizer=optimizer, epochs=args.epochs, device=device, model_name=args.model, data_type=args.data_type, is_pretrained=args.pretrained)
