import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION (RTX 4060 8GB VRAM Optimized)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Hyperparameters optimized for RTX 4060
LEARNING_RATE = 0.001
EPOCHS = 5000
BATCH_SIZE = 32  # Reduced for 8GB VRAM
LAMBDA_PHYSICS = 0.001  # Weight of Physics Loss (LCM constraint)
EARLY_STOPPING_PATIENCE = 200  # Allow 200 epochs without improvement

# Physical constants and scaling factors
T_AMB_CONST = 25.0  # Ambient temperature in Celsius
MAX_CURRENT = 100.0  # Max current for normalization (A)
MAX_VOLTAGE = 4.5  # Max voltage for normalization (V)
MAX_TEMP = 60.0  # Max temperature for normalization (°C)
MAX_CHARGE_TIME = 120.0  # Max charging duration for normalization (minutes)

# ==========================================
# 2. BUILD MODEL (PINN ARCHITECTURE for LCM)
# ==========================================
class BatteryPINN(nn.Module):
    """
    Physics-Informed Neural Network for Battery Thermal Modeling.
    
    Inputs:
      - Battery Type (categorical, 0-2)
      - State of Charge (SoC) [0-100%] normalized to [0,1]
      - Temperature [°C] normalized to [0,1]
      - Voltage [V] normalized to [0,1]
      - Current [A] normalized to [0,1]
      - Charging Mode (categorical, 0-2)
    
    Outputs:
      - Optimized Charging Time [0-1] (denormalized to minutes)
      - Maximum Battery Temperature [0-1] (denormalized to °C)
      - Mean Battery Temperature [0-1] (denormalized to °C)
    """
    def __init__(self, input_size=6, hidden_size=128):
        super(BatteryPINN, self).__init__()
        
        # Input layer with batch normalization for stability
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Hidden layers with residual connections
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
        self.hidden3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn4 = nn.BatchNorm1d(hidden_size // 2)
        
        # Output layers: 3 outputs (charging_time, max_temp, mean_temp)
        self.output_charging_time = nn.Linear(hidden_size // 2, 1)
        self.output_max_temp = nn.Linear(hidden_size // 2, 1)
        self.output_mean_temp = nn.Linear(hidden_size // 2, 1)
        
        # Activation function (Tanh for better PINN performance)
        self.activation = nn.Tanh()
        self.relu = nn.ReLU()
        
        # Learnable physics parameters for LCM
        # Theta1 = R / (m * Cp)  -> Heating Coefficient
        # Theta2 = hA / (m * Cp) -> Cooling Coefficient
        self.theta1 = nn.Parameter(torch.tensor([0.0001], dtype=torch.float32))
        self.theta2 = nn.Parameter(torch.tensor([0.001], dtype=torch.float32))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Encoder
        x = self.input_layer(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # First hidden block with residual connection
        x_res = x
        x = self.hidden1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second hidden block
        x = self.hidden2(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Third hidden block
        x = self.hidden3(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Output predictions (sigmoid to ensure [0, 1] range)
        charging_time = torch.sigmoid(self.output_charging_time(x))
        max_temp = torch.sigmoid(self.output_max_temp(x))
        mean_temp = torch.sigmoid(self.output_mean_temp(x))
        
        return charging_time, max_temp, mean_temp

# ==========================================
# 3. PHYSICS LOSS FUNCTION (LCM - Lumped Capacitance Model)
# ==========================================
def physics_loss_function(model, inputs, charging_time_pred, max_temp_pred, 
                         current_norm, temp_norm, ambient_temp=T_AMB_CONST):
    """
    Physics-informed loss based on Lumped Capacitance Model (LCM):
    dT/dt = theta1 * I^2 - theta2 * (T - T_amb)
    
    Constraints with proper normalization to prevent loss explosion:
    1. Max temperature should increase with current and decrease with cooling
    2. Charging time should be inversely related to charging current
    3. Temperature rise should follow exponential-like behavior
    """
    
    # Get learnable physics parameters with softplus for positivity
    theta1 = torch.nn.functional.softplus(model.theta1)  # Heating coefficient
    theta2 = torch.nn.functional.softplus(model.theta2)  # Cooling coefficient
    
    # ---- Constraint 1: LCM Temperature Equation (Normalized Space) ----
    # Work in normalized space to prevent magnitude explosion
    I_norm = inputs[:, 3:4]  # Current (already normalized, [0,1])
    T_norm_pred = max_temp_pred  # Already normalized [0,1]
    
    # Normalized heat balance: scale appropriately
    # theta1 operates on I^2 in [0,1] range
    # theta2 operates on T_diff in [0,1] range
    heat_generated_norm = (theta1 / 100.0) * (I_norm ** 2)  # Scale down theta1
    temp_diff_norm = T_norm_pred - (ambient_temp / MAX_TEMP)
    heat_dissipated_norm = (theta2 / 100.0) * temp_diff_norm  # Scale down theta2
    lcm_residual = heat_generated_norm - heat_dissipated_norm
    loss_lcm = torch.mean(lcm_residual ** 2)
    
    # ---- Constraint 2: Charging Time vs Current (Normalized) ----
    t_norm = charging_time_pred  # [0, 1]
    charge_product = t_norm * I_norm  # Both normalized
    expected_product = 0.3  # Reasonable normalized value
    loss_charge_conservation = torch.mean((charge_product - expected_product) ** 2) * 0.1
    
    # ---- Constraint 3: Temperature Physically Reasonable ----
    # Ensure predicted temp is above ambient and reasonable
    T_celsius = T_norm_pred * MAX_TEMP
    loss_temp_lower = torch.mean(torch.relu(ambient_temp - T_celsius + 1.0))  # T >= T_amb - 1
    loss_temp_upper = torch.mean(torch.relu(T_celsius - 55.0))  # T <= 55°C (reasonable)
    loss_temp_constraint = (loss_temp_lower + loss_temp_upper) / 2.0
    
    # ---- Combined Physics Loss ----
    # Keep all components in comparable magnitude
    loss_physics = loss_lcm + loss_charge_conservation + loss_temp_constraint
    
    return loss_physics, loss_lcm, loss_charge_conservation, loss_temp_constraint

# ==========================================
# 4. DATA PREPARATION
# ==========================================
def load_data(csv_path):
    """Load and preprocess the high-quality dataset."""
    print("Loading and preprocessing data...")
    
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify inputs and outputs
    # Inputs: Battery Type, SoC, Temp, Voltage, Current, Charging Mode (6 features)
    # Outputs: Charging Duration, Max Temp
    
    # Encode categorical variables
    le_battery = LabelEncoder()
    le_mode = LabelEncoder()
    
    battery_types = df['Battery Type'].values
    df['Battery_Type_Encoded'] = le_battery.fit_transform(battery_types)
    
    mode_types = df['Charging Mode'].values
    df['Mode_Encoded'] = le_mode.fit_transform(mode_types)
    
    # Extract input features (6 total - includes charging mode)
    input_features = np.column_stack([
        df['Battery_Type_Encoded'].values,
        df['SOC (%)'].values,
        df['Battery Temp (Â°C)'].values,
        df['Voltage (V)'].values,
        df['Current (A)'].values,
        df['Mode_Encoded'].values,
    ])
    
    # Extract target outputs
    # 1. Optimal Charging Duration (normalized)
    charging_duration = df['Charging Duration (min)'].values
    
    # 2. Maximum Battery Temperature (using actual temp as proxy for max)
    max_temp = df['Battery Temp (Â°C)'].values
    
    # 3. Mean Battery Temperature (approximate as initial + 0.6 * temp_rise)
    ambient_temp = df['Ambient Temp (Â°C)'].values
    mean_temp = ambient_temp + 0.6 * (max_temp - ambient_temp)
    
    # Normalize inputs
    scaler_input = StandardScaler()
    input_features_scaled = scaler_input.fit_transform(input_features)
    
    # Normalize outputs
    charging_duration_norm = charging_duration / MAX_CHARGE_TIME
    max_temp_norm = max_temp / MAX_TEMP
    mean_temp_norm = mean_temp / MAX_TEMP
    
    # Create tensors
    X = torch.from_numpy(input_features_scaled).float().to(DEVICE)
    y_time = torch.from_numpy(charging_duration_norm.reshape(-1, 1)).float().to(DEVICE)
    y_max_temp = torch.from_numpy(max_temp_norm.reshape(-1, 1)).float().to(DEVICE)
    y_mean_temp = torch.from_numpy(mean_temp_norm.reshape(-1, 1)).float().to(DEVICE)
    
    # Combine outputs (3 outputs)
    y = torch.cat([y_time, y_max_temp, y_mean_temp], dim=1)
    
    print(f"Inputs shape: {X.shape}")
    print(f"Outputs shape: {y.shape}")
    print(f"Input stats - Mean: {X.mean(dim=0)[:3]}, Std: {X.std(dim=0)[:3]}")
    
    return X, y, scaler_input, le_battery, le_mode, charging_duration, max_temp, mean_temp

def create_dataloaders(X, y, batch_size=BATCH_SIZE, train_ratio=0.8):
    """Split data and create data loaders."""
    dataset = TensorDataset(X, y)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ==========================================
# 5. TRAINING LOOP
# ==========================================
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=200, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta  # Allow smaller improvements
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.min_epochs = 50
        
    def __call__(self, val_loss, current_epoch):
        # Don't check early stopping until min_epochs reached
        if current_epoch < self.min_epochs:
            return
            
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        elif val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train():
    # Load data
    csv_path = "the_chosen_one  - data.csv"
    X, y, scaler_input, le_battery, le_mode, raw_charging_time, raw_max_temp, raw_mean_temp = load_data(csv_path)
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(X, y, batch_size=BATCH_SIZE)
    
    # Initialize model and optimizer
    model = BatteryPINN(input_size=6).to(DEVICE)
    
    # Use mixed precision for better VRAM efficiency
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion_mse = nn.MSELoss()
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    print("=" * 80)
    print("--- START TRAINING (CUDA Optimized for RTX 4060) ---")
    print("=" * 80)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Initial physics params: theta1={model.theta1.item():.6f}, theta2={model.theta2.item():.6f}")
    print("=" * 80)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            # Forward pass
            charging_time_pred, max_temp_pred, mean_temp_pred = model(batch_X)
            
            # Data loss (MSE for all 3 outputs)
            loss_data = criterion_mse(charging_time_pred, batch_y[:, 0:1]) + \
                       criterion_mse(max_temp_pred, batch_y[:, 1:2]) + \
                       criterion_mse(mean_temp_pred, batch_y[:, 2:3])
            
            # Physics loss (LCM constraints using max_temp)
            loss_phys, _, _, _ = physics_loss_function(
                model, batch_X, charging_time_pred, max_temp_pred,
                batch_X[:, 3:4],  # Current (normalized)
                batch_X[:, 2:3]   # Temperature (normalized)
            )
            
            # Combined loss
            loss_total = loss_data + (LAMBDA_PHYSICS * loss_phys)
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss_total.item()
            epoch_data_loss += loss_data.item()
            epoch_physics_loss += loss_phys.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                charging_time_pred, max_temp_pred, mean_temp_pred = model(batch_X)
                loss_data = criterion_mse(charging_time_pred, batch_y[:, 0:1]) + \
                           criterion_mse(max_temp_pred, batch_y[:, 1:2]) + \
                           criterion_mse(mean_temp_pred, batch_y[:, 2:3])
                epoch_val_loss += loss_data.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping check
        early_stopping(avg_val_loss, epoch)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        # Print progress
        if epoch % 200 == 0:
            theta1_val = torch.nn.functional.softplus(model.theta1).item()
            theta2_val = torch.nn.functional.softplus(model.theta2).item()
            print(f"Epoch {epoch:4d} | Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | Data: {epoch_data_loss/len(train_loader):.6f} | "
                  f"Phys: {epoch_physics_loss/len(train_loader):.6f}")
            print(f"           theta1: {theta1_val:.6f} | theta2: {theta2_val:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("=" * 80)
    print("--- TRAINING FINISHED ---")
    print("=" * 80)
    
    # Save model and metadata
    model_state = {
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler_input.mean_,
        'scaler_scale': scaler_input.scale_,
        'battery_type_classes': le_battery.classes_,
        'mode_classes': le_mode.classes_,
        'max_charge_time': MAX_CHARGE_TIME,
        'max_temp': MAX_TEMP,
        'max_current': MAX_CURRENT,
        'max_voltage': MAX_VOLTAGE,
    }
    
    torch.save(model_state, 'piml_battery_model.pth')
    print("Model saved to 'piml_battery_model.pth'")
    
    # Print final metrics
    model.eval()
    all_preds_time = []
    all_preds_max_temp = []
    all_preds_mean_temp = []
    all_targets_time = []
    all_targets_max_temp = []
    all_targets_mean_temp = []
    
    with torch.no_grad():
        for batch_X, batch_y in DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE):
            t_pred, max_temp_pred, mean_temp_pred = model(batch_X)
            all_preds_time.append(t_pred.cpu().numpy())
            all_preds_max_temp.append(max_temp_pred.cpu().numpy())
            all_preds_mean_temp.append(mean_temp_pred.cpu().numpy())
            all_targets_time.append(batch_y[:, 0:1].cpu().numpy())
            all_targets_max_temp.append(batch_y[:, 1:2].cpu().numpy())
            all_targets_mean_temp.append(batch_y[:, 2:3].cpu().numpy())
    
    preds_time = np.vstack(all_preds_time)
    preds_max_temp = np.vstack(all_preds_max_temp)
    preds_mean_temp = np.vstack(all_preds_mean_temp)
    targets_time = np.vstack(all_targets_time)
    targets_max_temp = np.vstack(all_targets_max_temp)
    targets_mean_temp = np.vstack(all_targets_mean_temp)
    
    mse_time = np.mean((preds_time - targets_time) ** 2)
    mse_max_temp = np.mean((preds_max_temp - targets_max_temp) ** 2)
    mse_mean_temp = np.mean((preds_mean_temp - targets_mean_temp) ** 2)
    rmse_time = np.sqrt(mse_time)
    rmse_max_temp = np.sqrt(mse_max_temp)
    rmse_mean_temp = np.sqrt(mse_mean_temp)
    
    print(f"\nFinal Metrics (Normalized):")
    print(f"  Charging Time - MSE: {mse_time:.6f}, RMSE: {rmse_time:.6f}")
    print(f"  Max Temperature - MSE: {mse_max_temp:.6f}, RMSE: {rmse_max_temp:.6f}")
    print(f"  Mean Temperature - MSE: {mse_mean_temp:.6f}, RMSE: {rmse_mean_temp:.6f}")
    
    print(f"\nFinal Metrics (Real Units):")
    print(f"  Charging Time - RMSE: {rmse_time * MAX_CHARGE_TIME:.4f} minutes")
    print(f"  Max Temperature - RMSE: {rmse_max_temp * MAX_TEMP:.4f} °C")
    print(f"  Mean Temperature - RMSE: {rmse_mean_temp * MAX_TEMP:.4f} °C")

if __name__ == "__main__":
    train()