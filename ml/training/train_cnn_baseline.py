#!/usr/bin/env python3
"""
train_cnn_baseline.py
---------------------
PHASE 1: Train a standard, non-spiking CNN to prove the balanced dataset 
is learnable and establish an accuracy ceiling before SNN conversion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Add parent directories to path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from dataset.generate_tactical_dataset import fen_to_tensor

# === CONFIGURATION ===
BATCH_SIZE = 64
EPOCHS = 50  # Increased from 15 to 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChessDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fen = row['fen']
        label = int(row['is_tactical'])
        
        # Get static 12x8x8 tensor and convert to float32
        tensor = torch.from_numpy(fen_to_tensor(fen)).float()
        return tensor, label

class TinyChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv Layer 1: Local pattern detection
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # 8x8 -> 4x4
        
        # Conv Layer 2: Tactical pattern combination
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # 4x4 -> 2x2
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 2 * 2, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 2) # 2 classes: Quiet (0), Tactical (1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    print(f"🚀 PHASE 1: Training Tiny Chess CNN on {DEVICE}")
    
    # 1. Load Data
    dataset_path = Path(__file__).parent.parent / "dataset" / "tactical_dataset_balanced.csv"
    dataset = ChessDataset(dataset_path)
    
    # 80/20 Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    model = TinyChessCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Track best validation accuracy
    best_val_acc = 0.0
    best_epoch = 0
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # Save best model
            save_path = Path(__file__).parent / "tiny_chess_cnn_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% 💾 NEW BEST")
        else:
            print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 4. Final Summary
    print(f"\n{'='*60}")
    print(f"✅ PHASE 1 COMPLETE")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Epoch Accuracy:     {val_acc:.2f}% (Epoch {EPOCHS})")
    print(f"Best Model Saved To:      tiny_chess_cnn_best.pth")
    print(f"{'='*60}")
    print(f"\n🎯 Target for Phase 2 (SNN Conversion): Match this {best_val_acc:.2f}% accuracy.")


if __name__ == "__main__":
    train_model()