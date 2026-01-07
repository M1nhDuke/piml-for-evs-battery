# PIML Battery Model - Quick Start Guide

## System Requirements

- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or similar
- **CUDA**: CUDA 11.8+
- **Python**: 3.9+
- **RAM**: 16GB minimum
- **Storage**: 2GB for model and datasets

## Installation Steps

### 1. Clone or Navigate to Repository
```bash
cd f:\projects\piml-for-evs-battery
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n piml-battery python=3.10
conda activate piml-battery

# OR using venv
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA Support
```bash
# For RTX 4060 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or use the latest CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify GPU Setup
```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Training the Model

### Basic Training
```bash
python piml_model.py
```

This will:
1. Load the dataset from `the_chosen_one - data.csv`
2. Preprocess and normalize features
3. Create train/validation split (80/20)
4. Train the PINN for up to 5000 epochs
5. Apply early stopping if validation loss plateaus
6. Save the trained model to `piml_battery_model.pth`

### Expected Training Time
- GPU (RTX 4060): ~10-15 minutes
- CPU (fallback): ~60-90 minutes

### Training Output Example
```
================================================================================
--- START TRAINING (CUDA Optimized for RTX 4060) ---
================================================================================
Model parameters: 50,432
Training samples: 800
Validation samples: 200
Initial physics params: theta1=0.000100, theta2=0.001000
================================================================================
Epoch    0 | Train Loss: 0.234521 | Val Loss: 0.231245 | Data: 0.198765 | Phys: 0.654321
           theta1: 0.000100 | theta2: 0.001000 | LR: 0.001000
Epoch  200 | Train Loss: 0.087654 | Val Loss: 0.091234 | Data: 0.045678 | Phys: 0.123456
           theta1: 0.000234 | theta2: 0.001456 | LR: 0.000998
...
================================================================================
--- TRAINING FINISHED ---
================================================================================

Final Metrics (Real Units):
  Charging Time - RMSE: 5.23 minutes
  Max Temperature - RMSE: 2.15 °C

Model saved to 'piml_battery_model.pth'
```

## Running Inference

### Single Sample Prediction
```bash
python inference.py
```

### Python API Usage
```python
from inference import load_trained_model, predict

# Load model
model, scaler, battery_encoder, mode_encoder = load_trained_model()

# Make prediction
result = predict(
    model, scaler, battery_encoder, mode_encoder,
    battery_type='Li-ion',      # Type of battery
    soc=50.0,                   # State of Charge (0-100%)
    temperature=30.0,           # Current temp (°C)
    voltage=3.8,                # Voltage (V)
    current=50.0,               # Current (A)
    charging_mode='Fast'        # Charging mode
)

print(f"Optimal Charging Time: {result['optimal_charging_time_minutes']:.2f} min")
print(f"Max Temperature: {result['max_temperature_celsius']:.2f}°C")
```

## Understanding the Outputs

### Model Predictions
- **Optimal Charging Time**: How long the battery should charge (in minutes)
  - Range: 20-120 minutes
  - Depends on: SoC, current, battery type

- **Maximum Temperature**: Peak temperature during charging (in °C)
  - Range: 20-60°C
  - Depends on: current, ambient temp, battery thermal properties

### Physics Parameters
- **θ₁ (Heating Coefficient)**: Represents I²R heating effect
  - Typical range: 0.00001 - 0.001
  - Higher = more heating per ampere

- **θ₂ (Cooling Coefficient)**: Represents heat dissipation
  - Typical range: 0.0001 - 0.01
  - Higher = better cooling capability

## Model Files

### piml_model.py
Main training script containing:
- BatteryPINN model class
- Physics loss functions (LCM constraints)
- Data preprocessing pipeline
- Training loop with early stopping
- CUDA optimization configurations

### inference.py
Inference script containing:
- Model loading with metadata
- Single and batch prediction functions
- Error metrics calculation
- Example usage demonstrations

### piml_battery_model.pth
Saved model checkpoint containing:
- Model state dictionary
- Preprocessing scalers
- Categorical encoders
- Normalization constants
- Physical parameters

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in piml_model.py
BATCH_SIZE = 16  # Instead of 32
```

### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU (not recommended)
# In code: DEVICE = torch.device("cpu")
```

### Data File Not Found
Ensure `the_chosen_one - data.csv` is in the same directory:
```bash
dir  # Windows
ls   # Linux/Mac
```

### Model Not Converging
Check:
1. Learning rate is too high (reduce to 0.0001)
2. Physics loss weight (try LAMBDA_PHYSICS = 0.01)
3. Data is properly normalized

## Performance Optimization Tips

### For RTX 4060 (8GB VRAM)
- Batch Size: Keep at 32
- Model Size: Current ~50K parameters is optimal
- Mixed Precision: Not supported (RTX 4060 lacks TF32)
- Gradient Accumulation: Not needed at current batch size

### Memory Profiling
```python
import torch
print(torch.cuda.memory_allocated() / 1e9)  # GB
print(torch.cuda.memory_reserved() / 1e9)   # GB
```

## Model Architecture Details

```
Input (6 features)
  ↓
Dense(6 → 128) + BatchNorm + Tanh + Dropout(0.1)
  ↓
Dense(128 → 128) + BatchNorm + Tanh + Dropout(0.1)
  ↓
Dense(128 → 128) + BatchNorm + Tanh + Dropout(0.1)
  ↓
Dense(128 → 64) + BatchNorm + Tanh + Dropout(0.1)
  ↓
├─ Dense(64 → 1) + Sigmoid  →  Charging Time [0, 1]
└─ Dense(64 → 1) + Sigmoid  →  Max Temperature [0, 1]
```

Total Parameters: ~50,432

## References

- PyTorch CUDA Documentation: https://pytorch.org/docs/stable/cuda.html
- RTX 4060 Specifications: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/
- Physics-Informed Neural Networks: https://arxiv.org/abs/1711.10566

## Support & Questions

For issues or questions, refer to:
1. Specific error message in console
2. README.md for architecture details
3. Comments in piml_model.py for configuration options
