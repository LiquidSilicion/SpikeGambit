import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# SpikingJelly imports
from spikingjelly.activation_based import functional

# Add parent directories to path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from dataset.generate_tactical_dataset import fen_to_tensor
from models.scnn import SpikeGambitSCNN

# === CONFIGURATION ===
BATCH_SIZE = 64
EPOCHS = 20          # Increased epochs
LEARNING_RATE = 1e-3
TIME_STEPS = 32      # Increased time steps for better integration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChessTacticalDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fen = row['fen']
        label = int(row['tactical_label'])
        
        # 1. Get static 12x8x8 tensor
        static_tensor = torch.from_numpy(fen_to_tensor(fen))
        
        # 2. IMPROVED ENCODING: 90% spike probability for pieces, 0% for empty
        # This preserves the spatial geometry of the board across time steps
        spike_prob = static_tensor * 0.9 
        spikes = torch.bernoulli(spike_prob.unsqueeze(0).repeat(TIME_STEPS, 1, 1, 1))
        
        return spikes, label

def train_model():
    print(f"🚀 Training SpikeGambit SCNN on {DEVICE}")
    
    # 1. Load Data
    dataset_path = Path(__file__).parent.parent / "dataset" / "tactical_dataset.csv"
    dataset = ChessTacticalDataset(dataset_path)
    
    # --- CRITICAL FIX: Calculate Class Weights ---
    class_counts = dataset.df['tactical_label'].value_counts().sort_index()
    total_samples = len(dataset.df)
    num_classes = len(class_counts)
    # Weight = Total / (Num_Classes * Count_of_Class)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(DEVICE)
    print(f"⚖️  Applying Class Weights to Loss Function: {class_weights.tolist()}")
    
    # 80/20 Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model with Weighted Loss
    model = SpikeGambitSCNN(T=TIME_STEPS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # <-- Weighted Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (spikes, labels) in enumerate(train_loader):
            spikes = spikes.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(spikes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_class_correct = defaultdict(int)
        val_class_total = defaultdict(int)
        
        with torch.no_grad():
            for spikes, labels in val_loader:
                spikes = spikes.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(spikes)
                functional.reset_net(model)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Track per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    val_class_total[label] += 1
                    if predicted[i] == label:
                        val_class_correct[label] += 1
                
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_loss = train_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 4. Final Per-Class Evaluation (The "Is it actually learning?" test)
    print("\n" + "="*60)
    print("📊 FINAL PER-CLASS VALIDATION ACCURACY")
    print("="*60)
    class_names = {0: "Quiet", 1: "Check", 2: "Hanging", 3: "Threat"}
    for cls in sorted(val_class_total.keys()):
        acc = 100. * val_class_correct[cls] / val_class_total[cls] if val_class_total[cls] > 0 else 0
        print(f"Class {cls} ({class_names[cls]:>8}): {acc:5.2f}% ({val_class_correct[cls]}/{val_class_total[cls]})")
    print("="*60)

    # 5. Save Model
    save_path = Path(__file__).parent / "spikegambit_scnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to {save_path}")

if __name__ == "__main__":
    train_model()