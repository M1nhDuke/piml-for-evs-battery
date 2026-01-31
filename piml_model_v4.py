import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 5000
BATCH_SIZE = 1024
LAMBDA_PHYSICS = 0.001 # Reduce from 0.05 to 0.01 to prevent initial instability
EARLY_STOPPING_PATIENCE = 200

# Physical constants
T_AMB_CONST = 25.0  # Ambient temperature (°C)
MAX_TIME = 120.0  # Max charging time (minutes)
MAX_CURRENT = 100.0  # Max current (A)
MAX_VOLTAGE = 400.0  # Max voltage (V)
MAX_TEMP = 100.0  # Max temperature (°C) - CHANGED from 60 to handle full range
MAX_SOC = 100.0  # Max SOC (%)

# ==========================================
# 2. BUILD MODEL (PINN ARCHITECTURE v4)
# ==========================================
class BatteryPINN_v4(nn.Module):
    """
    Physics-Informed Neural Network for Sequential Battery Thermal Modeling (v4).
    
    Inputs (3):
      - Time (minutes) [0-120] normalized to [0,1]
      - Current (Amps) [0-100] normalized to [0,1]
      - Voltage (Volts) [0-400] normalized to [0,1]
    
    Outputs (2):
      - Absolute Temperature [0-1] (denormalized to °C, now supports 0-100°C range)
      - State of Charge (SOC) [0-1] (denormalized to %)
    
    Physics Constraints (via Autograd):
      - dT/dt = θ₁*I² + θ₂*(T_amb - T)
      - dSOC/dt ≥ 0 (monotonic increase)
      - dSOC/dt ∝ Current (Coulomb counting)
    """
    def __init__(self, input_size=3, hidden_size=64, embedding_size=8):
        super(BatteryPINN_v4, self).__init__()
        
        self.embedding_size = embedding_size
        
        # Pre-layer embeddings
        self.exp_weights = nn.Parameter(torch.randn(input_size, embedding_size // 2))
        self.freq_weights = nn.Parameter(torch.randn(input_size, embedding_size // 2))
        
        pre_layer_output = input_size + int(input_size * 1.5 * embedding_size)
        
        # Connection layer
        self.connection = nn.Linear(pre_layer_output, hidden_size)
        self.bn_connection = nn.BatchNorm1d(hidden_size)
        
        # Hidden layers
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.hidden3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        # Output layers
        self.output_temperature = nn.Linear(hidden_size // 2, 1)  # Absolute Temperature
        self.output_soc = nn.Linear(hidden_size // 2, 1)  # State of Charge
        
        self.activation = nn.Tanh()
        
        # Learnable physics parameters for LCM
        # θ₁: heating coefficient (I²R losses)
        # θ₂: cooling coefficient (convection/conduction)
        self.theta1 = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))
        self.theta2 = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))
        
        self.dropout = nn.Dropout(p=0.1)

    def physical_embeddings(self, x):
        """Vectorized physical law embeddings."""
        x_expanded = x.unsqueeze(2)  # [batch, input_size, 1]
        exp_embed = torch.exp(-torch.abs(self.exp_weights.unsqueeze(0)) * x_expanded)
        exp_embed = exp_embed.reshape(x.shape[0], -1)
        
        freq_x = self.freq_weights.unsqueeze(0) * x_expanded
        sin_embed = torch.sin(freq_x).reshape(x.shape[0], -1)
        cos_embed = torch.cos(freq_x).reshape(x.shape[0], -1)
        
        all_embeddings = torch.cat([exp_embed, sin_embed, cos_embed], dim=1)
        return all_embeddings

    def forward(self, x):
        """Forward pass: returns absolute temperature and SOC."""
        embeddings = self.physical_embeddings(x)
        
        x_concat = torch.cat([x, embeddings], dim=1)
        x = self.connection(x_concat)
        x = self.bn_connection(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.hidden3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Outputs: both in [0, 1] range
        temperature = torch.sigmoid(self.output_temperature(x))
        soc = torch.sigmoid(self.output_soc(x))
        
        return temperature, soc

# ==========================================
# 3. PHYSICS LOSS WITH AUTOGRAD
# ==========================================
def physics_loss_function(model, inputs, temp_pred, soc_pred):
    """
    Physics-informed loss using torch.autograd.grad for derivatives.
    
    Key improvements:
    1. Compute dT/dt and dSOC/dt using autograd
    2. Enforce dSOC/dt >= 0 (monotonicity) with heavy penalty
    3. Apply Coulomb counting constraint
    4. Enforce thermal dynamics from paper equation
    """
    
    # Extract time from input for derivative computation
    # inputs shape: [batch_size, 3] -> [Time, Current, Voltage]
    time_norm = inputs[:, 0:1]  # Normalized time [0, 1]
    I_norm = inputs[:, 1:2]  # Current normalized
    
    # Get learnable physics parameters
    theta1 = torch.nn.functional.softplus(model.theta1) * 0.1  # Heating coeff
    theta2 = torch.nn.functional.softplus(model.theta2) * 0.1  # Cooling coeff
    
    # -------- COMPUTE DERIVATIVES USING AUTOGRAD --------
    # Enable gradients for time to compute dT/dt and dSOC/dt
    time_norm.requires_grad_(True)
    
    # Re-evaluate model at time points with gradient tracking
    # We need to reconstruct the full input with time having requires_grad=True
    inputs_grad = torch.cat([time_norm, inputs[:, 1:3]], dim=1)
    temp_pred_grad, soc_pred_grad = model(inputs_grad)
    
    # Compute dT/dt and dSOC/dt using autograd
    # Create ones for backward pass
    ones = torch.ones_like(temp_pred_grad)
    
    dtemp_dt = torch.autograd.grad(
        outputs=temp_pred_grad,
        inputs=time_norm,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    dsoc_dt = torch.autograd.grad(
        outputs=soc_pred_grad,
        inputs=time_norm,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    # Handle None gradients (shouldn't happen with proper setup)
    if dtemp_dt is None:
        dtemp_dt = torch.zeros_like(temp_pred_grad)
    if dsoc_dt is None:
        dsoc_dt = torch.zeros_like(soc_pred_grad)
    
    # -------- CONSTRAINT 1: THERMAL DYNAMICS --------
    # Physics equation: dT/dt = θ₁*I² + θ₂*(T_amb - T)
    T_norm_pred = temp_pred_grad
    T_celsius = T_norm_pred * MAX_TEMP
    
    # Heat generation (Joule heating: I²R)
    heat_in = theta1 * (I_norm ** 2)
    
    # Heat dissipation: θ₂*(T_amb - T)
    temp_diff = (T_AMB_CONST - T_celsius) / MAX_TEMP  # Normalize to [-1, 1]
    heat_out = theta2 * temp_diff
    
    # Expected dT/dt from physics
    dtemp_physics = heat_in + heat_out
    
    # Denormalize predicted dT/dt to match physics equation
    dtemp_dt_denorm = dtemp_dt * MAX_TEMP / MAX_TIME  # [0-100°C / 120min]
    
    # Physics constraint: minimize difference
    loss_thermal = torch.mean((dtemp_dt_denorm - dtemp_physics) ** 2)
    
    # -------- CONSTRAINT 2: SOC MONOTONICITY --------
    # dSOC/dt must always be >= 0 (battery only charges, never discharges)
    # Penalize negative slopes heavily
    dsoc_dt_denorm = dsoc_dt * MAX_SOC / MAX_TIME  # [0-100% / 120min]
    loss_soc_monotonic = torch.mean(torch.relu(-dsoc_dt_denorm)) * 10.0  # Heavy weight
    
    # -------- CONSTRAINT 3: COULOMB COUNTING --------
    # dSOC/dt should be proportional to Current
    # Higher current -> faster SOC increase
    # Target: dSOC/dt ≈ k * I, where k is a proportionality constant
    expected_dsoc_dt = 0.5 * I_norm  # Assume ~0.5%/min per Amp normalized
    loss_coulomb = torch.mean((dsoc_dt_denorm - expected_dsoc_dt * MAX_SOC) ** 2) * 0.1
    
    # -------- CONSTRAINT 4: REASONABLE BOUNDS --------
    # Temperature bounds
    loss_temp_lower = torch.mean(torch.relu(T_AMB_CONST - T_celsius + 1.0)) * 0.1
    loss_temp_upper = torch.mean(torch.relu(T_celsius - 95.0)) * 0.1  # Cap at 95°C
    
    # SOC bounds
    SOC_percent = soc_pred_grad * MAX_SOC
    loss_soc_lower = torch.mean(torch.relu(-SOC_percent)) * 0.1
    loss_soc_upper = torch.mean(torch.relu(SOC_percent - 101.0)) * 0.1
    
    # -------- COMBINED PHYSICS LOSS --------
    loss_physics = (loss_thermal + loss_soc_monotonic + loss_coulomb + 
                   loss_temp_lower + loss_temp_upper + loss_soc_lower + loss_soc_upper)
    
    return loss_physics, loss_thermal, loss_soc_monotonic, loss_coulomb

# ==========================================
# 4. DATA PREPARATION
# ==========================================
def load_and_prepare_data(csv_path):
    """
    Load sequential data and prepare for training.
    
    Inputs: time (min), current (A), voltage (V)
    Outputs: absolute temperature (°C), SOC (%)
    """
    print("Loading sequential charging data...")
    
    df = pd.read_csv(csv_path)
    print(f"Total data points: {df.shape[0]}")
    print(f"Unique transactions: {df['transaction_id'].nunique()}")
    
    # Process data
    X_list = []
    y_temp_list = []
    y_soc_list = []
    
    for transaction_id in df['transaction_id'].unique():
        transaction_data = df[df['transaction_id'] == transaction_id].sort_values('total_charging_min')
        
        # Skip short transactions
        if len(transaction_data) < 5:
            continue
        
        times = transaction_data['total_charging_min'].values
        currents = transaction_data['charginga'].values
        voltages = transaction_data['chargingv'].values
        temps = (transaction_data['charging_gun_temperature1'].values + 
                transaction_data['charging_gun_temperature2'].values) / 2.0
        socs = transaction_data['current_soc'].values
        
        # Create input-output pairs
        for i in range(len(times)):
            X_list.append([times[i], currents[i], voltages[i]])
            y_temp_list.append(temps[i])
            y_soc_list.append(socs[i])
    
    print(f"Total samples: {len(X_list)}")
    
    # Convert to arrays
    X = np.array(X_list)
    y_temp = np.array(y_temp_list).reshape(-1, 1)
    y_soc = np.array(y_soc_list).reshape(-1, 1)
    
    # Normalize inputs to [0, 1] range (MinMaxScaler ensures positive current values)
    scaler_input = MinMaxScaler()
    X_scaled = scaler_input.fit_transform(X)
    
    # Normalize outputs (now with MAX_TEMP = 100.0)
    y_temp_norm = np.clip(y_temp / MAX_TEMP, 0, 1)
    y_soc_norm = np.clip(y_soc / MAX_SOC, 0, 1)
    
    # To tensors
    X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
    y_temp_tensor = torch.from_numpy(y_temp_norm).float().to(DEVICE)
    y_soc_tensor = torch.from_numpy(y_soc_norm).float().to(DEVICE)
    
    print(f"Input shape: {X_tensor.shape}")
    print(f"Temperature range (normalized): [{y_temp_tensor.min():.4f}, {y_temp_tensor.max():.4f}]")
    print(f"SOC range (normalized): [{y_soc_tensor.min():.4f}, {y_soc_tensor.max():.4f}]")
    
    return X_tensor, y_temp_tensor, y_soc_tensor, scaler_input

def create_dataloaders(X, y_temp, y_soc, batch_size=BATCH_SIZE, train_ratio=0.8):
    """Split and create data loaders."""
    dataset = TensorDataset(X, y_temp, y_soc)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
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
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.min_epochs = 50
        
    def __call__(self, val_loss, current_epoch):
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
    csv_path = "final_dataset.csv"
    X, y_temp, y_soc, scaler_input = load_and_prepare_data(csv_path)
    
    # Create loaders
    train_loader, val_loader = create_dataloaders(X, y_temp, y_soc, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = BatteryPINN_v4(input_size=3, hidden_size=64, embedding_size=8).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion_mse = nn.MSELoss()
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    print("=" * 80)
    print("--- START TRAINING (PINN v4 - Autograd Physics Loss) ---")
    print("=" * 80)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"MAX_TEMP: {MAX_TEMP}°C (supports full range)")
    print("=" * 80)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_train_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        
        for batch_X, batch_y_temp, batch_y_soc in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            # CRITICAL: Enable gradients for inputs for autograd physics loss
            batch_X.requires_grad_(True)
            
            # Forward
            temp_pred, soc_pred = model(batch_X)
            
            # Data loss (MSE)
            loss_data = criterion_mse(temp_pred, batch_y_temp) + criterion_mse(soc_pred, batch_y_soc)
            
            # Physics loss (with autograd)
            loss_phys, loss_thermal, loss_mono, loss_coulomb = physics_loss_function(
                model, batch_X, temp_pred, soc_pred
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
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y_temp, batch_y_soc in val_loader:
                temp_pred, soc_pred = model(batch_X)
                loss_data = criterion_mse(temp_pred, batch_y_temp) + criterion_mse(soc_pred, batch_y_soc)
                epoch_val_loss += loss_data.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        early_stopping(avg_val_loss, epoch)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        if epoch % 10==0:
            print(f"Epoch {epoch:4d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                  f"Data: {epoch_data_loss/len(train_loader):.6f} | Phys: {epoch_physics_loss/len(train_loader):.6f}")
    
    print("=" * 80)
    print("--- TRAINING FINISHED ---")
    print("=" * 80)
    
    # Save model
    model_state = {
        'model_state_dict': model.state_dict(),
        'scaler_min': scaler_input.data_min_,
        'scaler_max': scaler_input.data_max_,
        'scaler_scale': scaler_input.scale_,
        'max_time': MAX_TIME,
        'max_temp': MAX_TEMP,
        'max_current': MAX_CURRENT,
        'max_voltage': MAX_VOLTAGE,
        'max_soc': MAX_SOC,
        'ambient_temp': T_AMB_CONST,
    }
    
    torch.save(model_state, 'piml_battery_model_v4.pth')
    print("Model saved to 'piml_battery_model_v4.pth'")
    
    # Final metrics
    model.eval()
    all_preds_temp = []
    all_preds_soc = []
    all_targets_temp = []
    all_targets_soc = []
    
    with torch.no_grad():
        for batch_X, batch_y_temp, batch_y_soc in DataLoader(
            TensorDataset(X, y_temp, y_soc), batch_size=BATCH_SIZE
        ):
            temp_pred, soc_pred = model(batch_X)
            all_preds_temp.append(temp_pred.cpu().numpy())
            all_preds_soc.append(soc_pred.cpu().numpy())
            all_targets_temp.append(batch_y_temp.cpu().numpy())
            all_targets_soc.append(batch_y_soc.cpu().numpy())
    
    preds_temp = np.vstack(all_preds_temp)
    preds_soc = np.vstack(all_preds_soc)
    targets_temp = np.vstack(all_targets_temp)
    targets_soc = np.vstack(all_targets_soc)
    
    rmse_temp = np.sqrt(np.mean((preds_temp - targets_temp) ** 2))
    rmse_soc = np.sqrt(np.mean((preds_soc - targets_soc) ** 2))
    r2_temp = 1 - (np.sum((targets_temp - preds_temp)**2) / np.sum((targets_temp - targets_temp.mean())**2))
    r2_soc = 1 - (np.sum((targets_soc - preds_soc)**2) / np.sum((targets_soc - targets_soc.mean())**2))
    
    print(f"\nFinal Metrics (Real Units):")
    print(f"  Temperature: RMSE={rmse_temp * MAX_TEMP:.2f}°C, R²={r2_temp:.4f}")
    print(f"  SOC: RMSE={rmse_soc * MAX_SOC:.2f}%, R²={r2_soc:.4f}")

if __name__ == "__main__":
    train()
