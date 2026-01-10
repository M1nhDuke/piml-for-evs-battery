import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from piml_model import BatteryPINN, MAX_CHARGE_TIME, MAX_TEMP, MAX_CURRENT, MAX_VOLTAGE

# ==========================================
# INFERENCE SCRIPT
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(checkpoint_path='piml_battery_model.pth'):
    """Load trained model and metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    model = BatteryPINN(input_size=6).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Restore preprocessing objects
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    
    battery_encoder = LabelEncoder()
    battery_encoder.classes_ = checkpoint['battery_type_classes']
    
    mode_encoder = LabelEncoder()
    mode_encoder.classes_ = checkpoint['mode_classes']
    
    return model, scaler, battery_encoder, mode_encoder
    mode_encoder.classes_ = checkpoint['mode_classes']
    
    return model, scaler, battery_encoder, mode_encoder

def predict(model, scaler, battery_encoder, mode_encoder, 
            battery_type, soc, temperature, voltage, current, charging_mode):
    """
    Make predictions using the trained PINN model.
    
    Args:
        battery_type: str - Type of battery (e.g., 'Li-ion', 'LiFePO4')
        soc: float - State of charge (0-100%)
        temperature: float - Current battery temperature (°C)
        voltage: float - Battery voltage (V)
        current: float - Charging current (A)
        charging_mode: str - Charging mode (e.g., 'Fast', 'Slow', 'Normal')
    
    Returns:
        dict with 'optimal_charging_time' (minutes), 'max_temperature' (°C), and 'mean_temperature' (°C)
    """
    
    # Encode categorical inputs
    battery_encoded = battery_encoder.transform([battery_type])[0]
    mode_encoded = mode_encoder.transform([charging_mode])[0]
    
    # Create input array
    input_array = np.array([[
        battery_encoded,
        soc,
        temperature,
        voltage,
        current,
        mode_encoded
    ]], dtype=np.float32)
    
    # Normalize using training scaler
    input_normalized = scaler.transform(input_array)
    
    # Convert to tensor
    input_tensor = torch.from_numpy(input_normalized).float().to(DEVICE)
    
    # Inference
    with torch.no_grad():
        charging_time_norm, max_temp_norm, mean_temp_norm = model(input_tensor)
    
    # Denormalize outputs
    charging_time_pred = charging_time_norm.cpu().numpy()[0, 0] * MAX_CHARGE_TIME
    max_temp_pred = max_temp_norm.cpu().numpy()[0, 0] * MAX_TEMP
    mean_temp_pred = mean_temp_norm.cpu().numpy()[0, 0] * MAX_TEMP
    
    return {
        'optimal_charging_time_minutes': float(charging_time_pred),
        'max_temperature_celsius': float(max_temp_pred),
        'mean_temperature_celsius': float(mean_temp_pred),
        'input_features': {
            'battery_type': battery_type,
            'soc_percent': soc,
            'temperature_celsius': temperature,
            'voltage_v': voltage,
            'current_a': current,
            'charging_mode': charging_mode
        }
    }

def batch_predict(model, scaler, battery_encoder, mode_encoder, csv_path):
    """
    Make predictions on a batch of data from CSV.
    """
    df = pd.read_csv(csv_path)
    
    results = []
    
    for idx, row in df.iterrows():
        pred = predict(
            model, scaler, battery_encoder, mode_encoder,
            battery_type=row['Battery Type'],
            soc=row['SOC (%)'],
            temperature=row['Battery Temp (Â°C)'],
            voltage=row['Voltage (V)'],
            current=row['Current (A)'],
            charging_mode=row['Charging Mode']
        )
        
        # Add actual values for comparison
        pred['actual_charging_time_minutes'] = row['Charging Duration (min)']
        pred['actual_max_temp_celsius'] = row['Battery Temp (Â°C)']
        
        results.append(pred)
    
    return results

if __name__ == "__main__":
    print("Loading trained model...")
    model, scaler, battery_encoder, mode_encoder = load_trained_model()
    
    # Example 1: Single prediction
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Prediction")
    print("="*80)
    result = predict(
        model, scaler, battery_encoder, mode_encoder,
        battery_type='Li-ion',
        soc=50.0,
        temperature=30.0,
        voltage=3.8,
        current=50.0,
        charging_mode='Fast'
    )
    
    print(f"Input:")
    print(f"  Battery Type: {result['input_features']['battery_type']}")
    print(f"  SoC: {result['input_features']['soc_percent']}%")
    print(f"  Current Temperature: {result['input_features']['temperature_celsius']}°C")
    print(f"  Voltage: {result['input_features']['voltage_v']}V")
    print(f"  Current: {result['input_features']['current_a']}A")
    print(f"  Charging Mode: {result['input_features']['charging_mode']}")
    print(f"\nPredictions:")
    print(f"  Optimal Charging Time: {result['optimal_charging_time_minutes']:.2f} minutes")
    print(f"  Maximum Temperature: {result['max_temperature_celsius']:.2f}°C")
    print(f"  Mean Temperature: {result['mean_temperature_celsius']:.2f}°C")
    
    # Example 2: Batch prediction on dataset
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Predictions (First 5 samples)")
    print("="*80)
    batch_results = batch_predict(model, scaler, battery_encoder, mode_encoder, 
                                   'the_chosen_one  - data.csv')
    
    for i, result in enumerate(batch_results[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Predicted Charging Time: {result['optimal_charging_time_minutes']:.2f} min | "
              f"Actual: {result['actual_charging_time_minutes']:.2f} min | "
              f"Error: {abs(result['optimal_charging_time_minutes'] - result['actual_charging_time_minutes']):.2f} min")
        print(f"  Predicted Max Temp: {result['max_temperature_celsius']:.2f}°C | "
              f"Actual: {result['actual_max_temp_celsius']:.2f}°C | "
              f"Error: {abs(result['max_temperature_celsius'] - result['actual_max_temp_celsius']):.2f}°C")
        print(f"  Predicted Mean Temp: {result['mean_temperature_celsius']:.2f}°C")
    
    # Calculate average errors
    print("\n" + "="*80)
    print("BATCH STATISTICS (All Samples)")
    print("="*80)
    
    time_errors = [abs(r['optimal_charging_time_minutes'] - r['actual_charging_time_minutes']) 
                   for r in batch_results]
    temp_errors = [abs(r['max_temperature_celsius'] - r['actual_max_temp_celsius']) 
                   for r in batch_results]
    
    print(f"Charging Time Prediction:")
    print(f"  Mean Error: {np.mean(time_errors):.4f} minutes")
    print(f"  Std Error: {np.std(time_errors):.4f} minutes")
    print(f"  Max Error: {np.max(time_errors):.4f} minutes")
    
    print(f"\nMax Temperature Prediction:")
    print(f"  Mean Error: {np.mean(temp_errors):.4f}°C")
    print(f"  Std Error: {np.std(temp_errors):.4f}°C")
    print(f"  Max Error: {np.max(temp_errors):.4f}°C")
