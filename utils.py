"""
Utility functions for Apple Stock Prediction
Helper functions for data analysis, visualization, and model evaluation.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def analyze_stock_data(data):
    """
    Perform comprehensive analysis of stock data.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
    
    Returns:
        dict: Analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['basic_stats'] = data.describe()
    
    # Price statistics
    analysis['price_stats'] = {
        'total_return': ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100,
        'volatility': data['Close'].pct_change().std() * np.sqrt(252) * 100,
        'max_price': data['Close'].max(),
        'min_price': data['Close'].min(),
        'current_price': data['Close'].iloc[-1]
    }
    
    # Volume analysis
    analysis['volume_stats'] = {
        'avg_volume': data['Volume'].mean(),
        'max_volume': data['Volume'].max(),
        'volume_trend': 'Increasing' if data['Volume'].iloc[-30:].mean() > data['Volume'].iloc[-60:-30].mean() else 'Decreasing'
    }
    
    return analysis

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

def plot_comprehensive_analysis(data):
    """
    Create comprehensive visualization of stock data.
    
    Args:
        data (pd.DataFrame): Stock data with technical indicators
    """
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Stock Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Stock price and moving averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA_50'], name='MA 50', line=dict(color='red')),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD_Signal'], name='MACD Signal', line=dict(color='red')),
        row=4, col=1
    )
    
    fig.update_layout(
        title='Comprehensive Stock Analysis',
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.show()

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

def plot_model_evaluation(y_true, y_pred, dates):
    """
    Create comprehensive model evaluation plots.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        dates (array): Date array
    """
    # Calculate metrics
    metrics = calculate_model_metrics(y_true, y_pred)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Actual vs Predicted',
            'Residuals Plot',
            'Scatter Plot',
            'Prediction Errors'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=dates, y=y_true, name='Actual', mode='lines', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=y_pred, name='Predicted', mode='lines', line=dict(color='red')),
        row=1, col=1
    )
    
    # Residuals
    residuals = y_true - y_pred
    fig.add_trace(
        go.Scatter(x=dates, y=residuals, name='Residuals', mode='lines', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=y_true, y=y_pred, mode='markers', name='Scatter', marker=dict(color='purple')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], 
                  mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    
    # Prediction errors
    errors = np.abs(y_true - y_pred)
    fig.add_trace(
        go.Scatter(x=dates, y=errors, name='Absolute Errors', mode='lines', line=dict(color='orange')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'Model Evaluation (RMSE: {metrics["rmse"]:.4f}, R²: {metrics["r2"]:.4f})',
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.show()
    
    # Print metrics
    print("📊 Model Evaluation Metrics:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

def save_model_results(model, history, metrics, filepath):
    """
    Save model results and metrics to file.
    
    Args:
        model: Trained model
        history: Training history
        metrics (dict): Evaluation metrics
        filepath (str): Path to save results
    """
    import json
    from datetime import datetime
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'training_history': {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        },
        'model_summary': str(model.summary())
    }
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Results saved to {filepath}")

def load_model_results(filepath):
    """
    Load model results from file.
    
    Args:
        filepath (str): Path to results file
    
    Returns:
        dict: Loaded results
    """
    import json
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results
