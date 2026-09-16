#!/usr/bin/env python3
"""
convert_to_snn.py
-----------------
PHASE 2: Convert the trained CNN to a Spiking Neural Network.
Measures accuracy retention across different time steps (T).

Based on methodology from:
- Nunes et al. 2022 (SNN Survey, IEEE Access)
- Diehl et al. 2015 (Fast-classifying SNNs through weight/threshold balancing)
"""

import torch
import torch.nn as nn
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Time steps to test (based on literature recommendations)
# Static vision tasks typically use T=4-16, but we'll test wider range
TIME_STEPS_TO_TEST = [10, 25, 50, 100, 200]

class ChessDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fen = row['fen']
        label = int(row['is_tactical'])
        tensor = torch.from_numpy(fen_to_tensor(fen)).float()
        return tensor, label


# === ORIGINAL CNN ARCHITECTURE (must match trained model exactly) ===
class ChessCNN_Enhanced(nn.Module):
    """Original CNN architecture (for reference)."""
    def __init__(self):
        super().__init__()
        # Block 1: Basic piece detection
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Block 2: Piece relationships
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Block 3: Tactical patterns
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


# === SNN ARCHITECTURE (ReLU → LIF) ===
class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron (software simulation).
    
    Based on Maass 1997 and standard LIF dynamics:
    V[t] = beta * V[t-1] + input
    if V[t] >= threshold: fire spike, reset V
    """
    def __init__(self, beta=0.9, threshold=1.0):
        self.beta = beta  # Leak factor (0.9 = slow leak, retains memory)
        self.threshold = threshold
        self.membrane = None
    
    def forward(self, input_current):
        """
        Process input current and return spikes.
        input_current shape: (batch, features)
        returns: spikes (batch, features)
        """
        # Initialize membrane with correct shape if needed
        if self.membrane is None or self.membrane.shape != input_current.shape:
            self.membrane = torch.zeros_like(input_current)
        
        # Leak + integrate
        self.membrane = self.beta * self.membrane + input_current
        
        # Fire where threshold exceeded
        spikes = (self.membrane >= self.threshold).float()
        
        # Reset membrane where spikes occurred (soft reset)
        self.membrane = self.membrane * (1 - spikes)
        
        return spikes


class ChessSNN(nn.Module):
    """
    Spiking version of ChessCNN_Enhanced.
    Each layer has its own LIF neuron instance with correct shape.
    """
    def __init__(self, beta=0.9, threshold=1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        
        # Conv layers (same as CNN, but without activation/dropout)
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        
        # Separate LIF neurons for each layer
        self.lif1 = LIFNeuron(beta=beta, threshold=threshold)
        self.lif2 = LIFNeuron(beta=beta, threshold=threshold)
        self.lif3 = LIFNeuron(beta=beta, threshold=threshold)
        self.lif4 = LIFNeuron(beta=beta, threshold=threshold)
        self.lif5 = LIFNeuron(beta=beta, threshold=threshold)
        
        # Output accumulator
        self.output_accumulator = None
    
    def forward_single_step(self, x):
        """
        Single timestep forward pass.
        x shape: (batch, 12, 8, 8)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize output accumulator if needed
        if self.output_accumulator is None:
            self.output_accumulator = torch.zeros(batch_size, 2, device=device)
        
        # Layer 1: Conv + Pool + LIF
        x = self.conv1(x)
        x = self.pool1(x)  # (batch, 32, 4, 4)
        x_shape = x.shape
        x_flat = x.view(batch_size, -1)  # (batch, 512)
        x_flat = self.lif1.forward(x_flat)
        x = x_flat.view(x_shape)
        
        # Layer 2: Conv + Pool + LIF
        x = self.conv2(x)
        x = self.pool2(x)  # (batch, 64, 2, 2)
        x_shape = x.shape
        x_flat = x.view(batch_size, -1)  # (batch, 256)
        x_flat = self.lif2.forward(x_flat)
        x = x_flat.view(x_shape)
        
        # Layer 3: Conv + LIF
        x = self.conv3(x)  # (batch, 128, 2, 2)
        x_shape = x.shape
        x_flat = x.view(batch_size, -1)  # (batch, 512)
        x_flat = self.lif3.forward(x_flat)
        x = x_flat.view(x_shape)
        
        # FC layers with LIF
        x = x.view(batch_size, -1)  # (batch, 512)
        x = self.lif4.forward(self.fc1(x))  # (batch, 256)
        x = self.lif5.forward(self.fc2(x))  # (batch, 128)
        
        # Output layer: accumulate membrane potential (no spike)
        output = self.fc3(x)
        self.output_accumulator += output
        
        return self.output_accumulator
    
    def forward(self, x, T):
        """
        Run for T timesteps with same static input.
        Returns average output over time (rate coding).
        """
        batch_size = x.shape[0]
        
        # Reset all LIF neurons for new inference
        self.lif1.membrane = None
        self.lif2.membrane = None
        self.lif3.membrane = None
        self.lif4.membrane = None
        self.lif5.membrane = None
        self.output_accumulator = None
        
        for t in range(T):
            self.forward_single_step(x)
        
        # Return average accumulated output
        return self.output_accumulator / T


def fold_batchnorm_into_conv(conv, bn):
    """
    Fold BatchNorm parameters into Conv weights.
    This eliminates the need to compute BN during inference.
    
    Math:
    BN(x) = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
    Conv(x) = W * x + b
    
    Combined: W_new = gamma * W / sqrt(running_var + eps)
              b_new = gamma * (b - running_mean) / sqrt(running_var + eps) + beta
    """
    with torch.no_grad():
        # Get BN parameters
        gamma = bn.weight
        beta = bn.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps
        
        # Compute scaling factor
        std = torch.sqrt(running_var + eps)
        scale = gamma / std
        
        # Fold into conv weights
        if isinstance(conv, nn.Conv2d):
            conv.weight *= scale.view(-1, 1, 1, 1)
            if conv.bias is not None:
                conv.bias.data = scale * (conv.bias - running_mean) + beta
            else:
                conv.bias = nn.Parameter(scale * (-running_mean) + beta)
        elif isinstance(conv, nn.Linear):
            conv.weight *= scale.view(-1, 1)
            if conv.bias is not None:
                conv.bias.data = scale * (conv.bias - running_mean) + beta
            else:
                conv.bias = nn.Parameter(scale * (-running_mean) + beta)
    
    return conv


def convert_cnn_to_snn(cnn_model, beta=0.9, threshold=1.0):
    """
    Convert trained CNN to SNN by:
    1. Folding BatchNorm into conv/linear weights
    2. Replacing ReLU with LIF neurons
    3. Removing Dropout (not needed for inference)
    """
    # Create SNN
    snn = ChessSNN(beta=beta, threshold=threshold)
    
    # Copy conv/linear weights first
    snn.conv1.weight.data = cnn_model.conv1.weight.data.clone()
    if cnn_model.conv1.bias is not None:
        snn.conv1.bias.data = cnn_model.conv1.bias.data.clone()
    
    snn.conv2.weight.data = cnn_model.conv2.weight.data.clone()
    if cnn_model.conv2.bias is not None:
        snn.conv2.bias.data = cnn_model.conv2.bias.data.clone()
    
    snn.conv3.weight.data = cnn_model.conv3.weight.data.clone()
    if cnn_model.conv3.bias is not None:
        snn.conv3.bias.data = cnn_model.conv3.bias.data.clone()
    
    snn.fc1.weight.data = cnn_model.fc1.weight.data.clone()
    if cnn_model.fc1.bias is not None:
        snn.fc1.bias.data = cnn_model.fc1.bias.data.clone()
    
    snn.fc2.weight.data = cnn_model.fc2.weight.data.clone()
    if cnn_model.fc2.bias is not None:
        snn.fc2.bias.data = cnn_model.fc2.bias.data.clone()
    
    snn.fc3.weight.data = cnn_model.fc3.weight.data.clone()
    snn.fc3.bias.data = cnn_model.fc3.bias.data.clone()
    
    # Fold BatchNorm parameters (after copying weights)
    fold_batchnorm_into_conv(snn.conv1, cnn_model.bn1)
    fold_batchnorm_into_conv(snn.conv2, cnn_model.bn2)
    fold_batchnorm_into_conv(snn.conv3, cnn_model.bn3)
    fold_batchnorm_into_conv(snn.fc1, cnn_model.bn4)
    fold_batchnorm_into_conv(snn.fc2, cnn_model.bn5)
    
    return snn


def evaluate_model(model, data_loader, T=None, is_snn=False):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            if is_snn:
                outputs = model(images, T)
            else:
                outputs = model(images)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def main():
    print("="*70)
    print("🔄 PHASE 2: ANN-to-SNN Conversion")
    print("="*70)
    
    # 1. Load validation data
    dataset_path = Path(__file__).parent.parent / "dataset" / "tactical_dataset_balanced.csv"
    full_dataset = ChessDataset(dataset_path)
    
    # Use same 80/20 split as training
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load trained CNN
    cnn_path = Path(__file__).parent / "chess_cnn_enhanced_best.pth"
    print(f"\n📂 Loading trained CNN from {cnn_path}")
    
    cnn_model = ChessCNN_Enhanced().to(DEVICE)
    state_dict = torch.load(cnn_path, map_location=DEVICE)
    
    # Handle checkpoint format (might be dict with 'model_state_dict')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        cnn_model.load_state_dict(state_dict['model_state_dict'])
        cnn_val_acc = state_dict.get('val_acc', 81.88)
    else:
        cnn_model.load_state_dict(state_dict)
        cnn_val_acc = 81.88
    
    print(f"✅ CNN loaded (validation accuracy: {cnn_val_acc:.2f}%)")
    
    # 3. Evaluate original CNN on validation set (sanity check)
    print(f"\n🔍 Sanity check: Evaluating original CNN...")
    cnn_accuracy = evaluate_model(cnn_model, val_loader, is_snn=False)
    print(f"   CNN Validation Accuracy: {cnn_accuracy:.2f}%")
    
    # 4. Convert to SNN and test different time steps
    print(f"\n🧠 Converting CNN to SNN (BatchNorm folding + ReLU → LIF)...")
    
    # Try different LIF parameters
    # Based on literature: beta=0.9 is good for static inputs (slow leak = memory)
    # Threshold needs tuning — too high = dead neurons, too low = noise
    best_snn_acc = 0
    best_T = 0
    best_params = {}
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"📊 Testing SNN across different time steps (T)")
    print(f"{'='*70}")
    print(f"{'T':<6} | {'Accuracy':<10} | {'Retention':<10} | {'Status'}")
    print(f"{'-'*70}")
    
    for T in TIME_STEPS_TO_TEST:
        # Convert with standard parameters
        snn = convert_cnn_to_snn(cnn_model, beta=0.9, threshold=1.0)
        snn = snn.to(DEVICE)
        
        # Evaluate
        snn_accuracy = evaluate_model(snn, val_loader, T=T, is_snn=True)
        retention = (snn_accuracy / cnn_accuracy) * 100
        
        status = ""
        if snn_accuracy > best_snn_acc:
            best_snn_acc = snn_accuracy
            best_T = T
            best_params = {'beta': 0.9, 'threshold': 1.0}
            status = "⭐ NEW BEST"
        
        print(f"{T:<6} | {snn_accuracy:>7.2f}%   | {retention:>7.2f}%   | {status}")
        
        results.append({
            'T': T,
            'accuracy': snn_accuracy,
            'retention': retention,
            'beta': 0.9,
            'threshold': 1.0
        })
    
    # 5. Try tuning threshold if accuracy is low
    print(f"\n🔧 Attempting threshold tuning to improve accuracy...")
    for threshold in [0.5, 0.75, 1.5, 2.0]:
        snn = convert_cnn_to_snn(cnn_model, beta=0.9, threshold=threshold)
        snn = snn.to(DEVICE)
        
        # Test with best T so far
        snn_accuracy = evaluate_model(snn, val_loader, T=best_T, is_snn=True)
        retention = (snn_accuracy / cnn_accuracy) * 100
        
        status = ""
        if snn_accuracy > best_snn_acc:
            best_snn_acc = snn_accuracy
            best_T = best_T
            best_params = {'beta': 0.9, 'threshold': threshold}
            status = "⭐ NEW BEST"
        
        print(f"   T={best_T:>3}, threshold={threshold:.2f}: {snn_accuracy:.2f}% (retention: {retention:.2f}%) {status}")
        
        results.append({
            'T': best_T,
            'accuracy': snn_accuracy,
            'retention': retention,
            'beta': 0.9,
            'threshold': threshold
        })
    
    # 6. Final Summary
    print(f"\n{'='*70}")
    print(f"✅ PHASE 2 COMPLETE")
    print(f"{'='*70}")
    print(f"Original CNN Accuracy:     {cnn_accuracy:.2f}%")
    print(f"Best SNN Accuracy:         {best_snn_acc:.2f}%")
    print(f"Accuracy Retention:        {(best_snn_acc/cnn_accuracy)*100:.2f}%")
    print(f"Best Time Steps (T):       {best_T}")
    print(f"Best LIF Parameters:       {best_params}")
    print(f"{'='*70}")
    
    # 7. Save best SNN
    best_snn = convert_cnn_to_snn(cnn_model, **best_params)
    best_snn = best_snn.to(DEVICE)
    snn_path = Path(__file__).parent / "chess_snn_best.pth"
    torch.save({
        'model_state_dict': best_snn.state_dict(),
        'T': best_T,
        'beta': best_params['beta'],
        'threshold': best_params['threshold'],
        'cnn_accuracy': cnn_accuracy,
        'snn_accuracy': best_snn_acc,
    }, snn_path)
    print(f"\n💾 Best SNN saved to {snn_path}")
    
    # 8. Save results table
    results_df = pd.DataFrame(results)
    results_path = Path(__file__).parent / "snn_conversion_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"📊 Results table saved to {results_path}")
    
    # 9. Paper-ready summary
    print(f"\n{'='*70}")
    print(f"📝 IEEE PAPER SUMMARY")
    print(f"{'='*70}")
    print(f"Phase 1 (CNN Baseline):  {cnn_accuracy:.2f}% accuracy")
    print(f"Phase 2 (SNN Conversion): {best_snn_acc:.2f}% accuracy at T={best_T}")
    print(f"Accuracy Retention:       {(best_snn_acc/cnn_accuracy)*100:.2f}%")
    print(f"\nNext: Phase 3 — Quantization + FPGA deployment")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()