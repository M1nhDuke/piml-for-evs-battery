# PIML Battery Model - Training & Inference Visual Guide

## 📋 Model Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Physics-Informed Neural Network (PINN)                   │
│                    For EV Battery Thermal Prediction                        │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT LAYER (6 features)
├─ Battery Type (categorical)         → One-hot / Label encoded
├─ State of Charge (0-100%)           → Normalized [0,1]
├─ Temperature (°C)                   → Normalized [0,1]
├─ Voltage (V)                        → Normalized [0,1]
├─ Current (A)                        → Normalized [0,1]
└─ Charging Mode (categorical)        → Label encoded

ENCODING BLOCK
└─ Dense(6→128) → BatchNorm → Tanh → Dropout(0.1)

HIDDEN BLOCKS (×3)
├─ Block 1: Dense(128→128) → BatchNorm → Tanh → Dropout(0.1)
├─ Block 2: Dense(128→128) → BatchNorm → Tanh → Dropout(0.1)
└─ Block 3: Dense(128→64) → BatchNorm → Tanh → Dropout(0.1)

OUTPUT HEADS (Dual prediction)
├─ Charging Time Head:     Dense(64→1) → Sigmoid → [0,1] → Denormalize (min)
└─ Max Temp Head:          Dense(64→1) → Sigmoid → [0,1] → Denormalize (°C)

PHYSICS CONSTRAINTS (LCM)
└─ dT/dt = θ₁·I² - θ₂·(T - T_amb)   [Learnable: θ₁, θ₂]

TOTAL PARAMETERS: ~50,432
VRAM USAGE: ~2-3 GB (RTX 4060)
```

## 🔄 Training Pipeline

```
START
  ↓
[1] LOAD DATA
    ├─ Read CSV (the_chosen_one - data.csv)
    ├─ Extract 6 input features
    ├─ Extract 2 output targets
    └─ Total: 1000 samples
  ↓
[2] PREPROCESS DATA
    ├─ Encode categorical variables
    │  ├─ Battery Type → [0, 1, 2]
    │  └─ Charging Mode → [0, 1, 2]
    ├─ StandardScaler normalization (inputs)
    └─ Output normalization (divide by MAX_TEMP, MAX_CHARGE_TIME)
  ↓
[3] SPLIT DATA
    ├─ Training: 800 samples (80%)
    └─ Validation: 200 samples (20%)
  ↓
[4] CREATE DATALOADERS
    └─ Batch size: 32
  ↓
[5] INITIALIZE MODEL
    ├─ Model: BatteryPINN(input_size=6)
    ├─ Optimizer: AdamW (lr=0.001, weight_decay=1e-5)
    ├─ Scheduler: CosineAnnealingLR (T_max=5000)
    └─ Early Stopping: patience=200
  ↓
[6] TRAINING LOOP (for each epoch)
    │
    ├─ FORWARD PASS
    │  └─ model(batch_X) → (charging_time_pred, max_temp_pred)
    │
    ├─ LOSS CALCULATION
    │  ├─ Data Loss:
    │  │  └─ L_data = MSE(t_pred, t_target) + MSE(T_pred, T_target)
    │  │
    │  └─ Physics Loss:
    │     ├─ L_LCM = ||θ₁·I² - θ₂·(T-T_amb)||²
    │     ├─ L_charge = ||t·I - capacity||²
    │     └─ L_temp = ||max(0, T_min - T)||²
    │
    ├─ TOTAL LOSS
    │  └─ L_total = L_data + λ_physics × L_physics
    │
    ├─ BACKWARD PASS
    │  ├─ backward()
    │  ├─ Gradient clipping (max_norm=1.0)
    │  └─ step()
    │
    └─ VALIDATION
       ├─ Evaluate on validation set
       ├─ Check early stopping criterion
       └─ Update learning rate schedule
  ↓
[7] CONVERGENCE CHECK
    ├─ Val loss improved? → Continue training
    ├─ No improvement for 200 epochs? → STOP (early stopping)
    └─ Reached max epochs (5000)? → STOP
  ↓
[8] SAVE MODEL
    ├─ Model state dict
    ├─ Preprocessing metadata
    ├─ Scaler parameters
    └─ Encoder information
  ↓
[9] FINAL EVALUATION
    ├─ Compute metrics on entire dataset:
    │  ├─ RMSE (Charging Time): ___ minutes
    │  ├─ RMSE (Max Temp): ___ °C
    │  ├─ Physics Parameters:
    │  │  ├─ θ₁ = ___
    │  │  └─ θ₂ = ___
    │  └─ Training summary
    └─ Print results
  ↓
END (Model saved to piml_battery_model.pth)
```

## 🎯 Inference Pipeline

```
START (inference.py)
  ↓
[1] LOAD TRAINED MODEL
    ├─ Load checkpoint (piml_battery_model.pth)
    ├─ Restore model weights
    ├─ Restore preprocessing objects
    │  ├─ StandardScaler
    │  ├─ BatteryType LabelEncoder
    │  └─ ChargingMode LabelEncoder
    └─ Set model to eval mode
  ↓
[2] PREPARE INPUT
    ├─ User inputs:
    │  ├─ battery_type = "Li-ion"
    │  ├─ soc = 50.0
    │  ├─ temperature = 30.0
    │  ├─ voltage = 3.8
    │  ├─ current = 50.0
    │  └─ charging_mode = "Fast"
    │
    ├─ Encode categoricals:
    │  ├─ battery_type → 0/1/2
    │  └─ charging_mode → 0/1/2
    │
    └─ Normalize with saved scaler
  ↓
[3] FORWARD PASS
    └─ model(input_tensor) → (time_norm [0,1], temp_norm [0,1])
  ↓
[4] DENORMALIZE
    ├─ time_denorm = time_norm × 120.0  (MAX_CHARGE_TIME)
    └─ temp_denorm = temp_norm × 60.0   (MAX_TEMP)
  ↓
[5] RETURN PREDICTIONS
    ├─ optimal_charging_time_minutes = ___
    ├─ max_temperature_celsius = ___
    └─ input_features (echo for verification)
  ↓
END (Results ready for use)
```

## 📊 Loss Function Visualization

```
TOTAL LOSS = L_data + λ_physics × L_physics

Where:

L_data (Mean Squared Error)
├─ Component 1: MSE(charging_time_pred, charging_time_target)
├─ Component 2: MSE(max_temp_pred, max_temp_target)
└─ Average of both components

L_physics (LCM Constraints)
├─ L_LCM: Residual of dT/dt = θ₁·I² - θ₂·(T - T_amb)
├─ L_charge: Charge conservation (t·I ≈ constant capacity)
└─ L_temp: Temperature must exceed ambient

Example Training Progression:
───────────────────────────────────────────────────────

Epoch 0:
  L_total = 0.245
  L_data  = 0.198 (high MSE)
  L_phys  = 0.654 (high physics violation)
  ↓ Learning...

Epoch 500:
  L_total = 0.032
  L_data  = 0.015 (improving)
  L_phys  = 0.321 (improving)
  ↓ Learning...

Epoch 1000:
  L_total = 0.008
  L_data  = 0.006 (good)
  L_phys  = 0.089 (good)
  ↓ Learning...

Epoch 1500:
  L_total = 0.005
  L_data  = 0.004 (convergence)
  L_phys  = 0.050 (convergence)
  ↓ Plateauing...

Epoch 1800:
  → Early Stopping Triggered
  → Model Saved
```

## 💾 Data Format

### Input CSV Structure
```
SOC (%),Voltage (V),Current (A),Battery Temp (°C),Ambient Temp (°C),
Charging Duration (min),Degradation Rate (%),Charging Mode,Efficiency (%),
Battery Type,Charging Cycles,EV Model,Optimal Charging Duration Class
────────────────────────────────────────────────────────────────────

Example Row:
43.7, 3.63, 33.6, 33.5, 26.4, 59.4, 8.8, Fast, 98.2, Li-ion, 112, Model B, 1
```

### Model Input Features (Normalized)
```
Feature                 Range       Normalization
────────────────────────────────────────────────
Battery Type           [0, 2]       LabelEncoded
SoC (%)                [0, 100]     StandardScaler
Temperature (°C)       [20, 60]     StandardScaler
Voltage (V)            [3.5, 4.2]   StandardScaler
Current (A)            [10, 100]    StandardScaler
Charging Mode          [0, 2]       LabelEncoded

All inputs are then StandardScaled to mean=0, std=1
```

### Model Output Features (Denormalized)
```
Output                      Range              Denormalization
──────────────────────────────────────────────────────────────
Charging Time (norm)        [0, 1]     × 120 → [0, 120] minutes
Max Temperature (norm)      [0, 1]     × 60  → [0, 60] °C
```

## 🔧 Configuration Quick Reference

```
HARDWARE (RTX 4060)          → BATCH_SIZE = 32
                             → HIDDEN_SIZE = 128
                             → EPOCHS = 5000

LEARNING (Training)          → LEARNING_RATE = 0.001
                             → OPTIMIZER = AdamW
                             → SCHEDULER = CosineAnnealingLR

PHYSICS (Constraints)        → LAMBDA_PHYSICS = 0.05
                             → T_AMBIENT = 25.0
                             → THETA1_INIT = 1e-4
                             → THETA2_INIT = 1e-3

NORMALIZATION (Data)         → MAX_CURRENT = 100.0 A
                             → MAX_VOLTAGE = 4.5 V
                             → MAX_TEMP = 60.0 °C
                             → MAX_CHARGE_TIME = 120.0 min
```

## 📈 Performance Targets

```
Metric                          Target          Threshold
──────────────────────────────────────────────────────────
Charging Time RMSE             5 minutes       < 10 minutes
Max Temperature RMSE           2.5 °C          < 5 °C
Physics Parameter θ₁           0.0002-0.0005   > 0
Physics Parameter θ₂           0.001-0.003    > 0
Training Time (RTX 4060)       ~15 minutes     < 30 minutes
Model Size                     1.8 MB          < 5 MB
```

## 🚀 Quick Command Reference

```bash
# Setup
pip install -r requirements.txt

# Training
python piml_model.py

# Inference
python inference.py

# Configuration check
python config.py

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 📌 Troubleshooting Quick Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA Out of Memory | Batch size too large | Reduce BATCH_SIZE to 16 |
| Loss not decreasing | Poor hyperparameters | Reduce LEARNING_RATE to 0.0001 |
| Model diverging | High physics loss weight | Reduce LAMBDA_PHYSICS to 0.01 |
| NaN loss values | Unstable gradients | Enable gradient clipping |
| Slow training | CPU fallback | Check CUDA availability |
| Data load error | CSV path wrong | Verify file location |

---

**Legend**:
- 📋 = Overview
- 🔄 = Process
- 📊 = Visualization
- 💾 = Data
- 🔧 = Configuration
- 🚀 = Execution
- 📌 = Reference
