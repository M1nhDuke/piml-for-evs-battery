# piml-for-evs-battery

Physics-Informed Neural Network (PINN) for Battery Thermal Modeling in Electric Vehicles

## 1. Introduction

This project implements a **Physics-Informed Neural Network (PINN)** optimized for **NVIDIA RTX 4060 (8GB VRAM)** to predict optimal battery charging time and maximum temperature during charging for electric vehicle batteries.

**Team**: USTH.HoLaDream (#CR7)
- Kiều Minh Đức (**Duke**): Leader
- Nguyễn Đình Giang (**Zang**): Model trainer
- Hoàng Viết Đức (**Florian**): Model trainer
- Nguyễn Mỹ (**Miidai**): Data collector
- Nguyễn Minh Đức (**Gr1mEd**): Data collector

## 2. Problem Statement

Electric vehicle adoption requires reliable battery thermal management. This project addresses the challenge of:
- Predicting optimal charging time while minimizing temperature rise
- Supporting multiple battery types (Li-ion, LiFePO4)
- Encoding physical constraints via the **Lumped Capacitance Model (LCM)**
- Providing predictions for different charging modes (Fast, Normal, Slow)

## 3. Model Architecture

### 3.1 Physics-Informed Neural Network (PINN)

The model implements a deep neural network with:
- **Inputs** (6 features):
  - Battery Type (categorical: Li-ion, LiFePO4, etc.)
  - State of Charge - SoC (0-100%)
  - Battery Temperature (°C)
  - Voltage (V)
  - Current (A)
  - Charging Mode (categorical: Fast, Normal, Slow)

- **Outputs** (3 predictions):
  - Optimal Charging Time (minutes)
  - Maximum Battery Temperature During Charge (°C)
  - Mean Battery Temperature During Charge (°C)

- **Network Design**:
  - Input Layer: 6 → 128 neurons
  - 3 Hidden Layers: 128 → 128 → 64 neurons with residual connections
  - Batch Normalization for training stability
  - Dropout (p=0.1) for regularization
  - Sigmoid output activation for bounded predictions [0, 1]

### 3.2 Physics Constraints (Lumped Capacitance Model)

The LCM is enforced as a physics loss term:

$$\frac{dT}{dt} = \theta_1 I^2 - \theta_2(T - T_{amb})$$

where:
- $\theta_1 = R/(m \cdot Cp)$ is the heating coefficient
- $\theta_2 = hA/(m \cdot Cp)$ is the cooling coefficient
- $I$ is the charging current
- $T_{amb}$ is ambient temperature (25°C)

**Learnable Parameters**: Both $\theta_1$ and $\theta_2$ are learned during training using SoftPlus activation to ensure positivity.

## 4. CUDA Optimization for RTX 4060

The model is optimized for efficient VRAM usage on 8GB NVIDIA RTX 4060:

| Optimization | Detail |
|---|---|
| **Batch Size** | 32 (reduced from typical 128) |
| **Model Size** | ~50K parameters |
| **Mixed Precision** | FP32 (RTX 4060 doesn't have TF32) |
| **Optimizer** | AdamW with L2 regularization |
| **Gradient Clipping** | Max norm = 1.0 |
| **Memory Optimization** | `set_to_none=True` in zero_grad() |
| **Learning Rate Schedule** | Cosine Annealing |
| **Early Stopping** | Patience = 200 epochs |

Estimated VRAM Usage: ~2-3GB

## 5. Training Configuration

```python
LEARNING_RATE = 0.001
EPOCHS = 5000
BATCH_SIZE = 32
LAMBDA_PHYSICS = 0.05  # Physics loss weight
EARLY_STOPPING_PATIENCE = 200
```

**Loss Function**:
$$L_{total} = L_{MSE} + \lambda_{phys} \cdot L_{physics}$$

where:
- $L_{MSE}$ = Mean Squared Error on all three outputs (charging_time, max_temp, mean_temp)
- $L_{physics}$ = LCM residual + charge conservation + temperature constraints

## 6. Dataset

The model is trained on a high-quality dataset with **1000 samples** containing:
- Battery types: Li-ion, LiFePO4
- EV Models: Model A, B, C
- Charging Modes: Fast, Normal, Slow
- Features: SoC (10-100%), Temperature (20-40°C), Voltage (3.5-4.2V), Current (10-100A)
- Charging Duration: 20-120 minutes
- Multiple degradation cycles (0-1000 cycles)

Data preprocessing:
- StandardScaler normalization for all inputs
- Output normalization to [0, 1] range
- Train/Validation split: 80/20
- Categorical encoding using LabelEncoder

## 7. Usage

### 7.1 Training

```bash
python piml_model.py
```

Output files:
- `piml_battery_model.pth` - Trained model checkpoint with metadata

### 7.2 Inference

Single prediction:
```python
from inference import load_trained_model, predict

model, scaler, battery_encoder, mode_encoder = load_trained_model()

result = predict(
    model, scaler, battery_encoder, mode_encoder,
    battery_type='Li-ion',
    soc=50.0,
    temperature=30.0,
    voltage=3.8,
    current=50.0,
    charging_mode='Fast'
)

print(f"Optimal Charging Time: {result['optimal_charging_time_minutes']:.2f} min")
print(f"Max Temperature: {result['max_temperature_celsius']:.2f}°C")
```

Batch prediction on CSV:
```bash
python inference.py
```

## 8. Performance Metrics

The model reports:
- **Mean Squared Error (MSE)** for both outputs
- **Root Mean Squared Error (RMSE)** in real units
- **Validation loss** for early stopping
- **Physics parameter values** ($\theta_1$, $\theta_2$)

Example output:
```
Final Metrics (Real Units):
  Charging Time - RMSE: 5.23 minutes
  Max Temperature - RMSE: 2.15 °C
```

## 9. Key Improvements from Previous Version

| Aspect | Previous | Current |
|---|---|---|
| **I/O** | Single output (Temp) | Dual output (Time + Temp) |
| **Inputs** | 4 features | 6 features (includes battery type) |
| **Dataset** | Synthetic data | Real high-quality data (1000 samples) |
| **Architecture** | Simple 3-layer | Deep with batch norm + residuals |
| **CUDA** | Basic GPU usage | Optimized for RTX 4060 8GB |
| **Physics** | Simple gradient-based | Full LCM with multiple constraints |
| **Regularization** | None | Dropout + L2 + gradient clipping |
| **Monitoring** | Basic printing | Early stopping + learning rate schedule |

## 10. Related Papers

- [Heat Transfer Modeling and Optimal Thermal Management of EV Battery Systems (MDPI, 2025)](https://www.mdpi.com/1996-1073/17/18/4575)
- [Physics-Informed Machine Learning for Battery Temperature Estimation (IEEE, 2022)](https://ieeexplore.ieee.org/document/9858911)
- [Critical Review of Temperature Prediction for Lithium-Ion Batteries (MDPI, 2024)](https://www.mdpi.com/2313-0105/10/12/421)
- [Thermal Management Optimization in EV Battery Pack Assembly (ResearchGate, 2024)](https://www.researchgate.net/publication/394306531)

## 11. System Requirements

```
PyTorch >= 2.0
numpy >= 1.21
pandas >= 1.3
scikit-learn >= 1.0
CUDA >= 11.8 (for GPU)
NVIDIA RTX 4060 (8GB VRAM recommended)
```

## 12. Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn
```

## 13. Project Structure

```
piml-for-evs-battery/
├── piml_model.py              # Main training script
├── inference.py               # Inference and evaluation
├── README.md                  # This file
├── piml_battery_model.pth     # Trained model checkpoint
└── the_chosen_one - data.csv  # Training dataset
```

- [Technische Universität Berlin (TU Berlin)](https://depositonce.tu-berlin.de/items/7f68932b-4d43-4f49-a5d8-914b00039f87)
- [Kaggle(1)](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns/data)
- [Shenzhen Auto Electric Power Plant Co., Ltd (Autosun) and Hong Kong Polytechnic University](https://data.mendeley.com/datasets/c7gg94tmvz/3)
- [Oxford University](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)

Unfortunately, most of them did not meet out requirements regarding parameters related to battery temperature, initial state of charge and charging tme. After testing and evaluating, we decided to use the dataset from [Kaggle(2)](https://www.kaggle.com/datasets/ziya07/ev-battery-charging-data). We had no intention of using data from Kaggle because we couldn't guarantee whether the data we used was real-world data or AI-generated; however, we had no other choice but to use it as it contained everything we need in this project. 


## 6. Tech stack

**Python** is used as the primary programming language for this project because it is easy to read, easy to debug, and has many libraries that support machine learning and deep learning-related tasks.

Supporting libraries:
- **numpy**: executing heavy numerical works
- **pandas**: working with the dataframe
- **pytorch** and **deepXDE**: training the PIML model
(there may be more, we will update this soon)


## 7. Our slides and scripts for the final presentation (we are coding and have not begun this part yet)


## 8. How to run the project
