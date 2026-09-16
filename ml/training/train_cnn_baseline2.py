#!/usr/bin/env python3
"""
train_cnn_baseline_v2.py
------------------------
PHASE 1 (Enhanced): Deeper network + data augmentation + learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

import sys
sys.path.append(str(Path(__file__).parent.parent))
from dataset.generate_tactical_dataset import fen_to_tensor

# === CONFIGURATION ===
BATCH_SIZE = 64
EPOCHS = 100  # Increased for better convergence
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 15  # Early stopping patience

class ChessDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        self.df = pd.read_csv(csv_path)
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def horizontal_flip(self, tensor):
        """Flip the board horizontally (mirror left-right)"""
        return torch.flip(tensor, [2])  # Flip along width dimension
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fen = row['fen']
        label = int(row['is_tactical'])
        tensor = torch.from_numpy(fen_to_tensor(fen)).float()
        
        # Data augmentation: randomly flip board horizontally
        if self.augment and torch.rand(1).item() > 0.5:
            tensor = self.horizontal_flip(tensor)
        
        return tensor, label

class ChessCNN_Enhanced(nn.Module):
    """Deeper network with BatchNorm for better chess pattern recognition."""
    def __init__(self):
        super().__init__()
        
        # Block 1: Basic piece detection
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        self.dropout1 = nn.Dropout(0.2)
        
        # Block 2: Piece relationships
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 4x4 -> 2x2
        self.dropout2 = nn.Dropout(0.3)
        
        # Block 3: Tactical patterns (no pooling to preserve spatial info)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # Conv blocks
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        
        # FC layers
        x = x.view(x.size(0), -1)
        x = self.dropout4(self.relu4(self.bn4(self.fc1(x))))
        x = self.dropout5(self.relu5(self.bn5(self.fc2(x))))
        x = self.fc3(x)
        return x

def train_model():
    print(f"🚀 PHASE 1 (Enhanced): Training deeper network with augmentation on {DEVICE}")
    
    # Load datasets with augmentation for training
    dataset_path = Path(__file__).parent.parent / "dataset" / "tactical_dataset_balanced.csv"
    train_dataset = ChessDataset(dataset_path, augment=True)
    val_dataset = ChessDataset(dataset_path, augment=False)  # No augmentation for validation
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # For validation, use the same split but without augmentation
    val_indices = list(range(train_size, len(train_dataset)))
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ChessCNN_Enhanced().to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_path = Path(__file__).parent / "chess_cnn_enhanced_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | Loss: {avg_val_loss:.4f} 💾 NEW BEST")
        else:
            epochs_without_improvement += 1
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final detailed evaluation
    print(f"\n{'='*70}")
    print(f"✅ PHASE 1 (Enhanced) COMPLETE")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"{'='*70}")
    
    # Load best model for detailed metrics
    checkpoint = torch.load(Path(__file__).parent / "chess_cnn_enhanced_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print("\n📊 Detailed Classification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=['Quiet', 'Tactical'],
        digits=3
    ))
    
    print(f"{'='*70}")
    print(f"🎯 Model saved to: chess_cnn_enhanced_best.pth")
    print(f"{'='*70}")

if __name__ == "__main__":
    train_model()