import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0005
EPOCHS = 8000
LAMBDA_PHYSICS = 0.01  # Weight of Physics Loss (needs tuning)

# Ambient temperature assumption (use a constant if not available in data)
T_AMB_CONST = 25.0  # 25 degrees Celsius
TIME_SCALE = 3600.0  # 1 unit model = 3600 seconds
CURRENT_SCALE = 50.0 # 1 unit model = 50 Amperes

# ==========================================
# 2. BUILD MODEL (PINN ARCHITECTURE)
# ==========================================
class BatteryPINN(nn.Module):
    def __init__(self):
        super(BatteryPINN, self).__init__()
        
        # Input: [Time, Current, Voltage, SoC] (4 features)
        # Note: Do not include 'Temperature' in Input as it is the target Output
        self.input_layer = nn.Linear(4, 64)
        
        # Hidden Layers
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        
        # Output: [Temperature]
        self.output_layer = nn.Linear(64, 1)
        
        # Tanh activation (Crucial for high-order derivatives in PINN)
        self.activation = nn.Tanh()

        # --- LEARNABLE PHYSICS PARAMETERS (Parameter Lumping) ---
        # Theta1 = R / (m * Cp)  -> Heating Coefficient
        # Theta2 = hA / (m * Cp) -> Cooling Coefficient
        # Initialize with small positive values
        self.theta1 = nn.Parameter(torch.tensor([1e-4], dtype=torch.float32))
        self.theta2 = nn.Parameter(torch.tensor([1e-3], dtype=torch.float32))

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        return self.output_layer(x)

# ==========================================
# 3. PHYSICS LOSS FUNCTION 
# ==========================================
def physics_loss_function(model, inputs, T_pred):
    """
    Calculate Residual with De-normalization (restoring real units)
    so theta1 and theta2 can learn meaningful physical values.
    """
    
    # 1. Calculate gradient w.r.t inputs (currently Normalized)
    grads = torch.autograd.grad(
        outputs=T_pred,
        inputs=inputs,
        grad_outputs=torch.ones_like(T_pred),
        create_graph=True
    )[0]
    
    # 2. Get Time Derivative and DE-NORMALIZE
    # Formula: dT/dt_real = (dT/dt_norm) / TIME_SCALE
    dT_dt_norm = grads[:, 0:1]
    dT_dt_real = dT_dt_norm / TIME_SCALE  
    
    # 3. Get Current and DE-NORMALIZE
    # Formula: I_real = I_norm * CURRENT_SCALE
    I_norm = inputs[:, 1:2]
    I_real = I_norm * CURRENT_SCALE
    
    # 4. Get Physics Parameters
    # Use softplus instead of abs to ensure smooth positive values
    theta1 = torch.nn.functional.softplus(model.theta1)
    theta2 = torch.nn.functional.softplus(model.theta2)
    
    # 5. Heat Balance Equation (Now both sides are in the same units)
    T_amb = T_AMB_CONST
    
    # Residual = dT/dt_real - (Heat_Gen - Heat_Diss)
    res = dT_dt_real - (theta1 * (I_real**2) - theta2 * (T_pred - T_amb))
    
    return torch.mean(res ** 2)

# ==========================================
# 4. DATA PREPARATION (WITH OUTPUT NORMALIZATION)
# ==========================================
def load_data():
    print("Generating Physics-based Dummy Data...")
    N = 1000
    dt = 1.0
    time = np.arange(0, N * dt, dt)
    
    # Generate Inputs
    current = np.random.uniform(20, 30, N)
    voltage = np.random.uniform(3.6, 4.2, N)
    soc = np.linspace(10, 80, N)
    
    # Generate Ground Truth Temperature (Physics-based)
    TRUE_THETA1 = 0.0005
    TRUE_THETA2 = 0.002
    temp_true = np.zeros(N)
    temp_true[0] = 25.0
    
    for i in range(N - 1):
        dT = (TRUE_THETA1 * (current[i]**2) - TRUE_THETA2 * (temp_true[i] - 25.0)) * dt
        temp_true[i+1] = temp_true[i] + dT
        
    temp_noisy = temp_true + np.random.normal(0, 0.1, N)
    
    df = pd.DataFrame({
        'Time': time, 'Current': current, 'Voltage': voltage, 'SoC': soc, 'Temp_Label': temp_noisy
    })
    
    inputs = df[['Time', 'Current', 'Voltage', 'SoC']].values.astype(np.float32)
    targets = df[['Temp_Label']].values.astype(np.float32)
    
    # Normalize Inputs
    inputs[:, 0] = inputs[:, 0] / time.max()
    inputs[:, 1] = inputs[:, 1] / 50.0 
    
    # --- NEW: Normalize Output (Target) ---
    # Neural Networks struggle to predict values like "40.5". 
    # We scale temperature to [0, 1] range. Assuming max temp is 60°C.
    TEMP_SCALE = 60.0 
    targets = targets / TEMP_SCALE
    # --------------------------------------
    
    x_tensor = torch.from_numpy(inputs).to(DEVICE)
    y_tensor = torch.from_numpy(targets).to(DEVICE)
    x_tensor.requires_grad = True
    
    # Return TEMP_SCALE to use during training
    return x_tensor, y_tensor, TEMP_SCALE

# ==========================================
# 5. TRAINING LOOP (UPDATED)
# ==========================================
def train():
    # Load data and the Temperature Scale factor
    inputs, targets_true, TEMP_SCALE = load_data()
    
    model = BatteryPINN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_data = nn.MSELoss()
    
    print("--- START TRAINING ---")
    print(f"Initial params: theta1={model.theta1.item():.5f}, theta2={model.theta2.item():.5f}")

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # 1. Forward Pass (Output is in range 0-1)
        T_pred_norm = model(inputs)
        
        # 2. Data Loss (Compare Normalized Pred vs Normalized Label)
        # Both are small numbers (0-1), so MSE will be small and stable.
        loss_data = criterion_data(T_pred_norm, targets_true)
        
        # 3. Physics Loss (Needs REAL Units)
        # We must scale the prediction back to real Degrees Celsius 
        # because the physics equation works with real units.
        T_pred_real = T_pred_norm * TEMP_SCALE
        loss_physics = physics_loss_function(model, inputs, T_pred_real)
        
        # 4. Total Loss
        loss_total = loss_data + (LAMBDA_PHYSICS * loss_physics)
        
        loss_total.backward()
        optimizer.step()
        
        if epoch % 500 == 0: # Print every 500 epochs to reduce spam
            print(f"Epoch {epoch}: Total={loss_total.item():.6f} | "
                  f"Data={loss_data.item():.6f} | "
                  f"Phys={loss_physics.item():.6f}")
            print(f"   -> Learned Params: theta1={torch.nn.functional.softplus(model.theta1).item():.5f}, "
                  f"theta2={torch.nn.functional.softplus(model.theta2).item():.5f}")

   
    print("--- FINISHED ---")
    
    # Save model
    torch.save(model.state_dict(), 'piml_battery_model.pth')

if __name__ == "__main__":
    train()