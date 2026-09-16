#!/usr/bin/env python3
"""
quantize_snn.py
---------------
PHASE 3: Quantize SNN weights from float32 to int8 for FPGA deployment.
Exports weights in a format suitable for Verilog initialization.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from training.convert_to_snn import ChessSNN

# === CONFIGURATION ===
QUANT_BITS = 8  # 8-bit quantization
DEVICE = torch.device("cpu")

def quantize_tensor(tensor, bits=8):
    """Quantize a float tensor to fixed-point integer."""
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    # Handle edge case where all values are the same
    if min_val == max_val:
        return torch.zeros_like(tensor, dtype=torch.int8), 1.0, 0.0
    
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = int(round(qmin - min_val / scale))
    
    # Quantize
    quantized = torch.clamp(
        torch.round(tensor / scale + zero_point),
        qmin, qmax
    ).to(torch.int8)
    
    return quantized, scale, zero_point

def quantize_model(model, bits=8):
    """Quantize all weights in the SNN model."""
    quantized_model = ChessSNN(beta=model.beta, threshold=model.threshold)
    quantization_params = {}
    
    # Quantize conv layers
    for name in ['conv1', 'conv2', 'conv3']:
        layer = getattr(model, name)
        quant_layer = getattr(quantized_model, name)
        
        q_weights, scale, zero_point = quantize_tensor(layer.weight.data, bits)
        quant_layer.weight.data = q_weights.float()
        
        if layer.bias is not None:
            q_bias, bias_scale, bias_zero_point = quantize_tensor(layer.bias.data, bits)
            quant_layer.bias.data = q_bias.float()
            quantization_params[name + '_bias'] = {
                'scale': float(bias_scale),
                'zero_point': int(bias_zero_point)
            }
        
        quantization_params[name] = {
            'scale': float(scale),
            'zero_point': int(zero_point)
        }
    
    # Quantize FC layers
    for name in ['fc1', 'fc2', 'fc3']:
        layer = getattr(model, name)
        quant_layer = getattr(quantized_model, name)
        
        q_weights, scale, zero_point = quantize_tensor(layer.weight.data, bits)
        quant_layer.weight.data = q_weights.float()
        
        if layer.bias is not None:
            q_bias, bias_scale, bias_zero_point = quantize_tensor(layer.bias.data, bits)
            quant_layer.bias.data = q_bias.float()
            quantization_params[name + '_bias'] = {
                'scale': float(bias_scale),
                'zero_point': int(bias_zero_point)
            }
        
        quantization_params[name] = {
            'scale': float(scale),
            'zero_point': int(zero_point)
        }
    
    return quantized_model, quantization_params

def export_weights_for_verilog(model, quantization_params, output_dir):
    """Export quantized weights in Verilog-friendly format (.mem files)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export conv weights
    for name in ['conv1', 'conv2', 'conv3']:
        layer = getattr(model, name)
        weights = layer.weight.data.cpu().numpy()
        
        # Convert int8 to uint8 for Verilog hex representation
        # This maps -128..127 to 128..255 (0x80..0xFF) and 0..127 to 0..127
        weights_uint8 = weights.astype(np.int8).view(np.uint8)
        weights_flat = weights_uint8.flatten()
        
        mem_file = output_dir / f"{name}_weights.mem"
        with open(mem_file, 'w') as f:
            for w in weights_flat:
                f.write(f"{int(w):02X}\n")
        
        print(f"✅ Exported {name} weights: {weights.shape} -> {mem_file.name}")
        
        if layer.bias is not None:
            bias = layer.bias.data.cpu().numpy().astype(np.int8).view(np.uint8)
            bias_file = output_dir / f"{name}_bias.mem"
            with open(bias_file, 'w') as f:
                for b in bias:
                    f.write(f"{int(b):02X}\n")
            print(f"✅ Exported {name} bias: {bias.shape} -> {bias_file.name}")
    
    # Export FC weights
    for name in ['fc1', 'fc2', 'fc3']:
        layer = getattr(model, name)
        weights = layer.weight.data.cpu().numpy()
        
        weights_uint8 = weights.astype(np.int8).view(np.uint8)
        weights_flat = weights_uint8.flatten()
        
        mem_file = output_dir / f"{name}_weights.mem"
        with open(mem_file, 'w') as f:
            for w in weights_flat:
                f.write(f"{int(w):02X}\n")
        
        print(f"✅ Exported {name} weights: {weights.shape} -> {mem_file.name}")
        
        if layer.bias is not None:
            bias = layer.bias.data.cpu().numpy().astype(np.int8).view(np.uint8)
            bias_file = output_dir / f"{name}_bias.mem"
            with open(bias_file, 'w') as f:
                for b in bias:
                    f.write(f"{int(b):02X}\n")
            print(f"✅ Exported {name} bias: {bias.shape} -> {bias_file.name}")
    
    # Export quantization parameters
    params_file = output_dir / "quantization_params.txt"
    with open(params_file, 'w') as f:
        for layer_name, params in quantization_params.items():
            f.write(f"{layer_name}: scale={params['scale']:.6f}, zero_point={params['zero_point']}\n")
    
    print(f"✅ Exported quantization parameters -> {params_file.name}")

def main():
    print("="*70)
    print("🔧 PHASE 3: Weight Quantization for FPGA Deployment")
    print("="*70)
    
    # 1. Load the best SNN
    snn_path = Path(__file__).parent.parent / "training" / "chess_snn_best.pth"
    print(f"\n📂 Loading SNN from {snn_path}")
    
    checkpoint = torch.load(snn_path, map_location=DEVICE)
    
    model = ChessSNN(
        beta=checkpoint['beta'],
        threshold=checkpoint['threshold']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    print(f"✅ SNN loaded (T={checkpoint['T']}, beta={checkpoint['beta']}, threshold={checkpoint['threshold']})")
    print(f"   CNN Accuracy: {checkpoint['cnn_accuracy']:.2f}%")
    print(f"   SNN Accuracy: {checkpoint['snn_accuracy']:.2f}%")
    
    # 2. Quantize the model
    print(f"\n🔢 Quantizing weights to {QUANT_BITS}-bit integers...")
    quantized_model, quantization_params = quantize_model(model, bits=QUANT_BITS)
    
    # 3. Export weights for Verilog
    output_dir = Path(__file__).parent / "verilog_weights"
    print(f"\n💾 Exporting weights for Verilog...")
    export_weights_for_verilog(quantized_model, quantization_params, output_dir)
    
    # 4. Save quantized model
    quantized_path = Path(__file__).parent / "chess_snn_quantized.pth"
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'quantization_params': quantization_params,
        'bits': QUANT_BITS,
        'beta': checkpoint['beta'],
        'threshold': checkpoint['threshold'],
    }, quantized_path)
    print(f"\n💾 Quantized model saved to {quantized_path}")
    
    # 5. Calculate model size reduction
    original_size = sum(p.numel() * 4 for p in model.parameters())
    quantized_size = sum(p.numel() * 1 for p in quantized_model.parameters())
    
    print(f"\n📊 Model Size Reduction:")
    print(f"   Original (float32):  {original_size / 1024:.2f} KB")
    print(f"   Quantized (int8):    {quantized_size / 1024:.2f} KB")
    print(f"   Compression Ratio:   {original_size / quantized_size:.2f}x")
    
    print(f"\n{'='*70}")
    print(f"✅ QUANTIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Next: Implement Verilog LIF neuron and convolution engine")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()