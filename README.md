# 🦑 SpikeGambit: Neuromorphic Chess on FPGA

**SpikeGambit** is a novel hardware-accelerated chess engine that utilizes a **Spiking Convolutional Neural Network (SCNN)** deployed on an FPGA for ultra-low-power tactical pattern recognition. 

Unlike traditional chess engines that rely on power-hungry Minimax search or dense CNNs (like AlphaZero), SpikeGambit leverages the event-driven, sparse computation of neuromorphic hardware to evaluate discrete spatial board states with a fraction of the energy footprint.

## 🎯 Project Goals
1. **Algorithmic Novelty:** Develop a temporal spike-encoding scheme to translate static 8x8 chess boards into dynamic spike trains.
2. **Hardware Acceleration:** Design a custom RTL architecture for sparse spiking convolutions, optimizing for BRAM and DSP efficiency.
3. **Benchmarking:** Compare the energy-per-inference (Joules/position) and throughput against a traditional Python Minimax baseline and an INT8 CNN.

## 🏗️ Architecture Overview
*(Insert a block diagram here later showing: FEN Input -> Spike Encoder -> SCNN Conv Layers -> Tactic Classification)*

## 📂 Repository Structure
- `baseline_engine/`: Traditional Python Minimax engine (Baseline for Elo/Speed comparison).
- `ml/`: PyTorch/SpikingJelly scripts for dataset generation, SCNN training, and fixed-point quantization.
- `hw/`: Verilog/SystemVerilog RTL, testbenches, and FPGA constraint files.
- `docs/`: Design documents, paper drafts, and mathematical proofs for the encoding scheme.

## 🚀 Getting Started
### Software (ML Training)
```bash
cd ml
pip install -r requirements.txt
python dataset/generate_tactical_dataset.py
```


```text
SpikeGambit/
├── README.md                 # The face of the project (Elevator pitch, architecture, setup)
├── LICENSE                   # MIT License (Standard for open-source academic projects)
├── .gitignore                # Crucial for ignoring Vivado/Python junk files
├── docs/                     # Architecture diagrams, paper drafts, dataset info
│   └── architecture.md
├── baseline_engine/          # Your Python Minimax engine (from Sept 13-15 logs)
│   ├── src/
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
├── ml/                       # Python SCNN Training, Quantization, Dataset Gen
│   ├── dataset/              # Scripts to generate FEN + tactical labels
│   ├── models/               # SpikingJelly/snnTorch SCNN definitions
│   ├── training/             # Training loops and evaluation scripts
│   ├── quantization/         # Exporting weights to C headers/BRAM init files
│   └── requirements.txt
├── hw/                       # FPGA Hardware Implementation
│   ├── rtl/                  # Verilog/SystemVerilog source files
│   ├── tb/                   # Testbenches (Cocotb or Verilog)
│   ├── constraints/          # XDC (Xilinx) or SDC (Intel) pin/timing constraints
│   └── scripts/              # Tcl scripts for Vivado/Quartus automation
└── .github/
    └── workflows/            # CI/CD for Python linting/testing
        └── python-ml-tests.yml
```

---


### Hardware (FPGA Synthesis)
*(Instructions for Vivado/Quartus will be added as RTL development progresses)*

## 📜 Target Publication
This project is being developed with the target of submission to **IEEE FPL 2027** (Field-Programmable Logic) or **IEEE NICE 2027** (Neuromorphic Computing and Engineering).

## 📄 License
This project is licensed under the MIT License.
```

#### `ml/requirements.txt`
```text
torch>=2.0.0
spikingjelly>=0.0.0.0.14
python-chess>=1.999
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
pandas>=2.0.0
```

---

## 📦 Dependencies & Submodules

The baseline chess engine is included as a git submodule. To clone the full project:

```bash
git clone --recursive https://github.com/YOUR_USERNAME/

## Baseline
The baseline is a standard python-chess minimax search at depth 3, 
representing a typical CPU-bound software approach. 
See `benchmarks/results/baseline_metrics.csv` for full metrics.