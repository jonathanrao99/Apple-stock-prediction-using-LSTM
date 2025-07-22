"""
Configuration file for Apple Stock Prediction
Contains all the parameters and settings for the LSTM model.
"""

# Data Configuration
DATA_CONFIG = {
    'data_path': 'AAPL.csv',
    'date_column': 'Date',
    'target_column': 'Close',
    'feature_columns': ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
}

# Model Configuration
MODEL_CONFIG = {
    'window_size': 50,  # Number of time steps to look back
    'test_size_ratio': 0.1,  # Ratio of data for testing
    'lstm_units_1': 500,  # Units in first LSTM layer
    'lstm_units_2': 100,  # Units in second LSTM layer
    'dropout_rate': 0.2,  # Dropout rate for regularization
    'learning_rate': 0.001,  # Learning rate for optimizer
    'batch_size': 32,  # Batch size for training
    'epochs': 25,  # Number of training epochs
    'validation_split': 0.1,  # Validation split ratio
}

# Training Configuration
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'loss_function': 'mean_squared_error',
    'metrics': ['mae'],
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'plot_style': 'plotly_white',
    'figure_width': 1200,
    'figure_height': 600,
    'colors': {
        'actual_train': '#1f77b4',
        'actual_test': '#2ca02c',
        'predicted_train': '#ff7f0e',
        'predicted_test': '#d62728',
        'training_loss': '#1f77b4',
        'validation_loss': '#ff7f0e'
    }
}

# File Paths
PATHS = {
    'data_dir': './data',
    'models_dir': './models',
    'results_dir': './results',
    'logs_dir': './logs'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'stock_prediction.log'
}
