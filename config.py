"""
Configuration file for PIML Battery Model
Easy parameter adjustment for different hardware and datasets
"""

# ==========================================
# HARDWARE & DEVICE CONFIGURATION
# ==========================================

# GPU Settings
USE_CUDA = True                    # Set to False to force CPU
GPU_MEMORY_FRACTION = 0.95         # Use up to 95% of GPU VRAM
DEVICE_ID = 0                      # GPU device index (0 for single GPU)

# RTX 4060 Specific (8GB VRAM)
BATCH_SIZE = 32                    # Optimized for 8GB VRAM
ACCUMULATION_STEPS = 1             # Gradient accumulation (1 = disabled)
NUM_WORKERS = 0                    # DataLoader workers (0 on Windows)

# ==========================================
# MODEL ARCHITECTURE
# ==========================================

INPUT_SIZE = 6                     # Number of input features
HIDDEN_SIZE = 128                  # Hidden layer neurons
NUM_HIDDEN_LAYERS = 3              # Number of hidden layers
DROPOUT_RATE = 0.1                 # Dropout probability
USE_BATCH_NORM = True              # Use batch normalization
USE_RESIDUAL = True                # Use residual connections
OUTPUT_SIZE = 2                    # Two outputs: charging_time, max_temp

# ==========================================
# TRAINING CONFIGURATION
# ==========================================

# Learning Rate Schedule
LEARNING_RATE = 0.001              # Initial learning rate
LR_SCHEDULER = "cosine"            # Options: "cosine", "linear", "exponential"
SCHEDULER_T_MAX = 5000             # Max epochs for cosine annealing
WEIGHT_DECAY = 1e-5                # L2 regularization

# Optimization
OPTIMIZER = "AdamW"                # Options: "Adam", "AdamW", "SGD"
GRADIENT_CLIP_NORM = 1.0           # Max gradient norm for clipping
GRADIENT_CLIP_VALUE = None         # Set to value for value-based clipping

# Training Loop
EPOCHS = 5000                      # Maximum number of epochs
EARLY_STOPPING_PATIENCE = 200      # Patience for early stopping
EARLY_STOPPING_MIN_DELTA = 0.0001  # Minimum improvement for early stopping
VALIDATION_RATIO = 0.2             # Train/validation split ratio

# ==========================================
# PHYSICS LOSS CONFIGURATION (LCM)
# ==========================================

# Loss Weights
LAMBDA_PHYSICS = 0.05              # Physics loss weight
LAMBDA_DATA = 1.0                  # Data loss weight (MSE)
LAMBDA_REGULARIZATION = 0.1        # Regularization loss weight

# Physics Parameters
T_AMBIENT = 25.0                   # Ambient temperature (°C)
THETA1_INIT = 1e-4                 # Initial heating coefficient
THETA2_INIT = 1e-3                 # Initial cooling coefficient

# ==========================================
# DATA NORMALIZATION
# ==========================================

# Feature Ranges (for denormalization)
MAX_CURRENT = 100.0                # Maximum current (A)
MAX_VOLTAGE = 4.5                  # Maximum voltage (V)
MAX_TEMP = 60.0                    # Maximum temperature (°C)
MAX_CHARGE_TIME = 120.0            # Maximum charging time (minutes)
MAX_SOC = 100.0                    # Maximum SoC (%)

# Normalization Strategy
USE_STANDARD_SCALER = True         # Use StandardScaler (True) or MinMaxScaler (False)
NORMALIZE_TARGETS = True           # Normalize output targets
NORMALIZE_INPUTS = True            # Normalize input features

# ==========================================
# DATA CONFIGURATION
# ==========================================

# Dataset Paths
DATA_CSV_PATH = "the_chosen_one  - data.csv"
MODEL_SAVE_PATH = "piml_battery_model.pth"
LOG_DIR = "logs"                   # Directory for training logs

# Data Augmentation
USE_DATA_AUGMENTATION = False      # Enable data augmentation
AUGMENTATION_NOISE = 0.01          # Gaussian noise std for augmentation
AUGMENTATION_FACTOR = 1.0          # Factor to increase dataset size

# ==========================================
# LOGGING & MONITORING
# ==========================================

# Printing/Logging
PRINT_INTERVAL = 200               # Print stats every N epochs
SAVE_INTERVAL = 500                # Save checkpoint every N epochs
USE_TENSORBOARD = False            # Log to TensorBoard
TENSORBOARD_DIR = "runs"           # TensorBoard log directory

# Metrics to Track
TRACK_METRICS = [
    "train_loss_data",
    "train_loss_physics",
    "val_loss_data",
    "theta1",
    "theta2"
]

# ==========================================
# INFERENCE CONFIGURATION
# ==========================================

# Batch Inference
INFERENCE_BATCH_SIZE = 64          # Batch size for inference (larger is faster)
INFERENCE_DEVICE = "cuda"          # Device for inference

# Output Formatting
SAVE_PREDICTIONS_CSV = True        # Save batch predictions to CSV
PREDICTION_OUTPUT_PATH = "predictions.csv"
DECIMAL_PLACES = 4                 # Decimal places in output

# ==========================================
# HARDWARE PRESETS
# ==========================================

HARDWARE_PRESETS = {
    "rtx4060_8gb": {
        "batch_size": 32,
        "hidden_size": 128,
        "max_epochs": 5000,
        "learning_rate": 0.001,
    },
    "rtx3080_10gb": {
        "batch_size": 64,
        "hidden_size": 256,
        "max_epochs": 5000,
        "learning_rate": 0.001,
    },
    "rtx3090_24gb": {
        "batch_size": 128,
        "hidden_size": 512,
        "max_epochs": 5000,
        "learning_rate": 0.001,
    },
    "cpu": {
        "batch_size": 16,
        "hidden_size": 64,
        "max_epochs": 3000,
        "learning_rate": 0.0005,
    }
}

# ==========================================
# EXPERIMENTAL/DEBUG OPTIONS
# ==========================================

# Debug Mode
DEBUG_MODE = False                 # Enable debug prints
PROFILING_ENABLED = False          # Enable memory/speed profiling
SAVE_GRADIENTS = False             # Save gradient statistics

# Reproducibility
RANDOM_SEED = 42                   # Random seed for reproducibility
DETERMINISTIC = True               # Force deterministic behavior

# Data Validation
VALIDATE_DATA = True               # Check data for NaN/Inf values
CHECK_PHYSICS_LOSS = True          # Verify physics loss computation

# ==========================================
# ADVANCED PARAMETERS (Expert Only)
# ==========================================

# Batch Normalization Momentum
BN_MOMENTUM = 0.1                  # BatchNorm momentum
BN_EPSILON = 1e-5                  # BatchNorm epsilon

# Activation Functions
ACTIVATION_FUNCTION = "tanh"       # Options: "tanh", "relu", "leaky_relu", "elu"
OUTPUT_ACTIVATION = "sigmoid"      # Options: "sigmoid", "softmax", "linear"

# Loss Function Variants
LOSS_FUNCTION = "mse"              # Options: "mse", "mae", "smooth_l1"
PHYSICS_LOSS_VARIANT = "lcm"       # Options: "lcm", "lcm_extended", "custom"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_hardware_config(hardware_type="rtx4060_8gb"):
    """Get configuration for specific hardware."""
    return HARDWARE_PRESETS.get(hardware_type, HARDWARE_PRESETS["rtx4060_8gb"])

def update_config(config_dict):
    """Update configuration from dictionary."""
    import sys
    current_module = sys.modules[__name__]
    for key, value in config_dict.items():
        if hasattr(current_module, key):
            setattr(current_module, key, value)
        else:
            print(f"Warning: Unknown configuration parameter '{key}'")

# ==========================================
# CONFIGURATION VALIDATION
# ==========================================

def validate_config():
    """Validate configuration parameters."""
    issues = []
    
    # Check ranges
    if not (0 < LEARNING_RATE < 0.1):
        issues.append(f"LEARNING_RATE {LEARNING_RATE} outside typical range [0.0001, 0.1]")
    
    if not (0 < LAMBDA_PHYSICS < 1.0):
        issues.append(f"LAMBDA_PHYSICS {LAMBDA_PHYSICS} should be in [0, 1]")
    
    if BATCH_SIZE < 1 or BATCH_SIZE > 256:
        issues.append(f"BATCH_SIZE {BATCH_SIZE} seems unreasonable")
    
    if DROPOUT_RATE < 0 or DROPOUT_RATE > 0.5:
        issues.append(f"DROPOUT_RATE {DROPOUT_RATE} should be in [0, 0.5]")
    
    if EARLY_STOPPING_PATIENCE > EPOCHS:
        issues.append(f"EARLY_STOPPING_PATIENCE exceeds EPOCHS")
    
    if issues:
        print("Configuration Warnings:")
        for issue in issues:
            print(f"  - {issue}")
    
    return len(issues) == 0

if __name__ == "__main__":
    validate_config()
    print("✓ Configuration valid for RTX 4060 8GB VRAM")
