"""
Configuration file for Apple Stock Prediction
Contains all the parameters and settings for the LSTM model.
"""

import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# Suppress all warnings aggressively
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*oneDNN.*')
warnings.filterwarnings('ignore', message='.*protobuf.*')
warnings.filterwarnings('ignore', message='.*deprecated.*')
warnings.filterwarnings('ignore', message='.*will be removed.*')
warnings.filterwarnings('ignore', message='.*tensorflow.*')
warnings.filterwarnings('ignore', message='.*google.*')

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
    'epochs': 25,  # Number of training epochs
}

# Training Configuration
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'loss_function': 'mean_squared_error',
    'batch_size': 32,  # Batch size for training
}
