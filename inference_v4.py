import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TEMP = 100.0  # NOW SUPPORTS FULL 0-100°C RANGE
MAX_SOC = 100.0
T_AMB_CONST = 25.0

# ==========================================
# LOAD MODEL v4
# ==========================================
class BatteryPINN_v4(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, embedding_size=8):
        super(BatteryPINN_v4, self).__init__()
        
        self.embedding_size = embedding_size
        
        self.exp_weights = nn.Parameter(torch.randn(input_size, embedding_size // 2))
        self.freq_weights = nn.Parameter(torch.randn(input_size, embedding_size // 2))
        
        pre_layer_output = input_size + int(input_size * 1.5 * embedding_size)
        
        self.connection = nn.Linear(pre_layer_output, hidden_size)
        self.bn_connection = nn.BatchNorm1d(hidden_size)
        
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.hidden3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        self.output_temperature = nn.Linear(hidden_size // 2, 1)
        self.output_soc = nn.Linear(hidden_size // 2, 1)
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)
        
        self.theta1 = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))
        self.theta2 = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))

    def physical_embeddings(self, x):
        x_expanded = x.unsqueeze(2)
        exp_embed = torch.exp(-torch.abs(self.exp_weights.unsqueeze(0)) * x_expanded)
        exp_embed = exp_embed.reshape(x.shape[0], -1)
        
        freq_x = self.freq_weights.unsqueeze(0) * x_expanded
        sin_embed = torch.sin(freq_x).reshape(x.shape[0], -1)
        cos_embed = torch.cos(freq_x).reshape(x.shape[0], -1)
        
        all_embeddings = torch.cat([exp_embed, sin_embed, cos_embed], dim=1)
        return all_embeddings

    def forward(self, x):
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
        
        temperature = torch.sigmoid(self.output_temperature(x))
        soc = torch.sigmoid(self.output_soc(x))
        
        return temperature, soc

def load_trained_model(model_path='piml_battery_model_v4.pth'):
    """Load v4 model with metadata."""
    print(f"Loading model from {model_path}...")
    
    model_state = torch.load(model_path, weights_only=False)
    
    model = BatteryPINN_v4(input_size=3, hidden_size=64, embedding_size=8).to(DEVICE)
    model.load_state_dict(model_state['model_state_dict'])
    model.eval()
    
    scaler_input = MinMaxScaler()
    scaler_input.data_min_ = model_state['scaler_min']
    scaler_input.data_max_ = model_state['scaler_max']
    scaler_input.scale_ = model_state['scaler_scale']
    scaler_input.min_ = np.zeros_like(model_state['scaler_min'])  # MinMaxScaler default: feature_range=(0,1) means min_=0
    
    return model, scaler_input, model_state

# ==========================================
# SINGLE PREDICTION
# ==========================================
def predict(model, scaler, time_min, current_a, voltage_v):
    """
    Single prediction.
    
    Args:
        time_min: Charging time in minutes [0-120]
        current_a: Current in Amps [0-100]
        voltage_v: Voltage in Volts [0-400]
    
    Returns:
        temperature (°C), soc (%), time_to_99pct (min)
    """
    x_raw = np.array([[time_min, current_a, voltage_v]])
    x_scaled = scaler.transform(x_raw)
    x_tensor = torch.from_numpy(x_scaled).float().to(DEVICE)
    
    with torch.no_grad():
        temp_norm, soc_norm = model(x_tensor)
    
    temp_celsius = temp_norm.item() * MAX_TEMP
    soc_percent = soc_norm.item() * MAX_SOC
    
    # Coulomb counting: time to 99% SOC
    time_to_99 = None
    if current_a > 0:
        charge_per_min = (current_a / 100.0) * 1.0  # Normalized
        remaining_soc = 99 - soc_percent
        if remaining_soc > 0:
            time_to_99 = remaining_soc / (charge_per_min * 100)
        else:
            time_to_99 = 0
    
    return temp_celsius, soc_percent, time_to_99

# ==========================================
# BATCH PREDICTION (CSV INPUT)
# ==========================================
def batch_predict(model, scaler, csv_path='final_dataset.csv', output_path='predictions_v4.csv'):
    """
    Batch prediction: process CSV file and output predictions.
    
    Adds 4 new columns:
      - predicted_temperature_celsius
      - predicted_soc_percent
      - predicted_time_to_99pct_minutes
      - ambient_temperature_celsius
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total samples to predict: {len(df)}")
    
    X = df[['total_charging_min', 'charginga', 'chargingv']].values
    X_scaled = scaler.transform(X)
    X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
    
    print("Running predictions...")
    batch_size = 512
    all_temps = []
    all_socs = []
    all_times_to_99 = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            temp_batch, soc_batch = model(batch)
            
            temps = temp_batch.cpu().numpy().flatten() * MAX_TEMP
            socs = soc_batch.cpu().numpy().flatten() * MAX_SOC
            
            all_temps.extend(temps)
            all_socs.extend(socs)
            
            # Coulomb counting for each sample
            currents = X[i:i+batch_size, 1]  # Original current values (in Amps)
            for j, (temp, soc, current) in enumerate(zip(temps, socs, currents)):
                if current > 0:
                    # Estimate: charge rate ≈ 0.8%/min per 100A normalized
                    # For current in Amps, SOC change rate ≈ (current / 100) * 0.8 %/min
                    charge_rate_pct_per_min = (current / 100.0) * 0.8
                    remaining = 99.0 - soc
                    time_to_99 = max(0, remaining / charge_rate_pct_per_min) if remaining > 0 else 0
                else:
                    time_to_99 = 0
                all_times_to_99.append(time_to_99)
            
            if (i + batch_size) % 2000 == 0:
                print(f"  Processed {min(i + batch_size, len(X_tensor))} samples...")
    
    # Create output dataframe
    df['predicted_temperature_celsius'] = all_temps
    df['predicted_soc_percent'] = all_socs
    df['predicted_time_to_99pct_minutes'] = all_times_to_99
    df['ambient_temperature_celsius'] = T_AMB_CONST
    
    print(f"Saving predictions to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Temperature predictions:")
    print(f"  Range: [{all_temps[0]:.2f}°C, {max(all_temps):.2f}°C]")
    print(f"  Mean: {np.mean(all_temps):.2f}°C, Std: {np.std(all_temps):.2f}°C")
    print(f"\nSOC predictions:")
    print(f"  Range: [{all_socs[0]:.2f}%, {max(all_socs):.2f}%]")
    print(f"  Mean: {np.mean(all_socs):.2f}%, Std: {np.std(all_socs):.2f}%")
    print(f"\nTime to 99% predictions:")
    print(f"  Range: [{min(all_times_to_99):.2f} min, {max(all_times_to_99):.2f} min]")
    print(f"  Mean: {np.mean(all_times_to_99):.2f} min, Std: {np.std(all_times_to_99):.2f} min")
    print("=" * 80)
    
    return df

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("=" * 80)
    print("PINN v4 INFERENCE ENGINE")
    print("=" * 80)
    
    # Load model
    model, scaler, model_state = load_trained_model('piml_battery_model_v4.pth')
    
    # Batch prediction
    batch_predict(model, scaler, csv_path='final_dataset.csv', output_path='predictions_v4.csv')
    
    print("\nInference complete!")
    print("Output saved to: predictions_v4.csv")
