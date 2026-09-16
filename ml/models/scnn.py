import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer

class SpikeGambitSCNN(nn.Module):
    def __init__(self, T=16):
        super().__init__()
        self.T = T  # Number of time steps for simulation
        
        # Conv Layer 1: Local pattern detection (e.g., piece pairs)
        self.conv1 = layer.Conv2d(12, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(32)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        
        # Conv Layer 2: Tactical pattern detection (e.g., forks, pins)
        self.conv2 = layer.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(64)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        
        # Spatial Pooling & Classification
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(64, 4)  # 4 Classes: Quiet, Check, Hanging, Threat
        self.lif_out = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())

    def forward(self, x):
        """
        x shape: (Batch, T, 12, 8, 8) -> Transpose to (T, Batch, 12, 8, 8)
        Returns: (Batch, 4) -> Average firing rate of output neurons over T steps
        """
        # Transpose to put time dimension first
        x = x.transpose(0, 1)  # (Batch, T, C, H, W) -> (T, Batch, C, H, W)
        
        out = 0
        for t in range(self.T):
            x_t = x[t]  # Now correctly gets (Batch, 12, 8, 8)
            
            # Layer 1
            x_t = self.conv1(x_t)
            x_t = self.bn1(x_t)
            x_t = self.lif1(x_t)
            
            # Layer 2
            x_t = self.conv2(x_t)
            x_t = self.bn2(x_t)
            x_t = self.lif2(x_t)
            
            # Pooling and Output
            x_t = self.pool(x_t)
            x_t = torch.flatten(x_t, 1)  # (Batch, 64)
            x_t = self.fc(x_t)
            x_t = self.lif_out(x_t)
            
            out += x_t  # Accumulate spikes
            
        return out / self.T  # Return average firing rate