"""
Utility functions for Apple Stock Prediction
Helper functions for data analysis, visualization, and model evaluation.
"""

# Suppress warnings before any other imports
import os
import warnings

# Set environment variables for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# Suppress all warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*oneDNN.*')
warnings.filterwarnings('ignore', message='.*protobuf.*')
warnings.filterwarnings('ignore', message='.*deprecated.*')
warnings.filterwarnings('ignore', message='.*tensorflow.*')
warnings.filterwarnings('ignore', message='.*keras.*')
warnings.filterwarnings('ignore', message='.*reset_default_graph.*')

# Import config after warning suppression
import config

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def create_technical_indicators(data):
    """
    Create technical indicators for the stock data.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
    
    Returns:
        pd.DataFrame: Data with technical indicators
    """
    df = data.copy()
    
    # Moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def calculate_model_metrics(y_true, y_pred):
    """
    Calculate comprehensive model evaluation metrics.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
    
    Returns:
        dict: Evaluation metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Percentage metrics
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    metrics['directional_accuracy'] = np.mean(direction_true == direction_pred) * 100
    
    return metrics
