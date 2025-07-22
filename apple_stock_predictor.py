#!/usr/bin/env python3
"""
Apple Stock Prediction using LSTM Neural Networks
A comprehensive stock price prediction system using deep learning.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AppleStockPredictor:
    """
    A class to handle Apple stock prediction using LSTM neural networks.
    """
    
    def __init__(self, data_path='AAPL.csv', window_size=50, test_size_ratio=0.1):
        """
        Initialize the stock predictor.
        
        Args:
            data_path (str): Path to the CSV file containing stock data
            window_size (int): Number of time steps to look back
            test_size_ratio (float): Ratio of data to use for testing
        """
        self.data_path = data_path
        self.window_size = window_size
        self.test_size_ratio = test_size_ratio
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the stock data."""
        print("📊 Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path, parse_dates=['Date'])
        
        # Prepare features
        self.data['date'] = self.data['Date']
        self.data['close'] = self.data['Close']
        self.data.drop(columns=['Date', 'Close'], inplace=True)
        
        # Store original close prices for later use
        self.original_close = self.data['close'].copy()
        
        # Separate date and features
        self.df_date = self.data[['date']]
        self.df_close = self.data[['close']]
        self.df_features = self.data.drop(columns=['date'])
        
        # Scale features
        self._scale_features()
        
        print(f"✅ Data loaded successfully! Shape: {self.data.shape}")
        return self.data
    
    def _scale_features(self):
        """Scale all features using StandardScaler."""
        for feature in self.df_features.columns:
            scaler = StandardScaler()
            scaler.fit(self.df_features[[feature]])
            self.df_features[feature] = scaler.transform(self.df_features[[feature]])
    
    def create_sequences(self):
        """Create time-series sequences for LSTM training."""
        print("🔄 Creating time-series sequences...")
        
        X, y, dates = [], [], []
        
        for i in range(self.df_features.shape[0] - self.window_size):
            X.append(np.asarray(self.df_features.values[i:i+self.window_size]).astype(np.float64))
            y.append(self.df_features['close'].values[i+self.window_size])
            dates.append(self.df_date['date'].iloc[i+self.window_size])
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.dates = dates
        
        print(f"✅ Sequences created! X shape: {self.X.shape}, y shape: {self.y.shape}")
        return self.X, self.y, self.dates
    
    def split_data(self):
        """Split data into training and testing sets."""
        print("✂️ Splitting data into train/test sets...")
        
        train_size = int((1 - self.test_size_ratio) * len(self.y))
        
        # Split sequences
        self.X_train = self.X[:train_size]
        self.X_test = self.X[train_size:]
        self.y_train = self.y[:train_size]
        self.y_test = self.y[train_size:]
        
        # Split dates
        self.dates_train = self.dates[:train_size]
        self.dates_test = self.dates[train_size:]
        
        # Store original close prices for plotting
        self.y_train_original = self.original_close[self.window_size:train_size+self.window_size]
        self.y_test_original = self.original_close[train_size+self.window_size:]
        
        print(f"✅ Data split complete! Train: {len(self.y_train)}, Test: {len(self.y_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self):
        """Build and compile the LSTM model."""
        print("🧠 Building LSTM model...")
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(500, return_sequences=True, input_shape=(self.window_size, self.df_features.shape[1])),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dense(1)
        ])
        
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        print("✅ Model built successfully!")
        self.model.summary()
        return self.model
    
    def train_model(self, epochs=25, verbose=1):
        """Train the LSTM model."""
        print(f"🚀 Training model for {epochs} epochs...")
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            verbose=verbose
        )
        
        print("✅ Training completed!")
        return self.history
    
    def predict(self):
        """Generate predictions for training and test sets."""
        print("�� Generating predictions...")
        
        # Predict on test set
        y_pred = self.model.predict(self.X_test)
        y_pred = self.scaler.inverse_transform(y_pred)
        
        # Predict on training set
        y_pred_train = self.model.predict(self.X_train)
        y_pred_train = self.scaler.inverse_transform(y_pred_train)
        
        # Reshape predictions
        self.y_pred = self._reshape_predictions(y_pred)
        self.y_pred_train = self._reshape_predictions(y_pred_train)
        
        print("✅ Predictions generated!")
        return self.y_pred, self.y_pred_train
    
    def _reshape_predictions(self, array):
        """Reshape prediction array to 1D."""
        return [pred[0] for pred in array]
    
    def calculate_metrics(self):
        """Calculate performance metrics."""
        print("📊 Calculating performance metrics...")
        
        rmse = np.sqrt(mean_squared_error(self.y_pred, self.y_test_original))
        print(f"✅ Root Mean Square Error (RMSE): {rmse:.4f}")
        
        return rmse
    
    def predict_next_day(self):
        """Predict the next day's stock price."""
        print("🔮 Predicting next day's stock price...")
        
        # Get last window_size values
        last_values = self.df_features.iloc[-self.window_size:].values
        
        # Predict
        pred_close = self.model.predict(last_values.reshape(1, self.window_size, self.df_features.shape[1]))[0][0]
        pred_close = self.scaler.inverse_transform(pred_close.reshape(-1, 1))[0][0]
        
        # Calculate next date
        next_date = self.dates_test[-1] + timedelta(days=1)
        
        self.future_pred = {
            'date': next_date,
            'predicted_close': pred_close
        }
        
        print(f"✅ Next day prediction: {next_date.strftime('%Y-%m-%d')} - ${pred_close:.2f}")
        return self.future_pred
    
    def plot_training_history(self):
        """Plot training and validation loss."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=self.history.history['loss'],
            name='Training Loss',
            mode='lines',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            y=self.history.history['val_loss'],
            name='Validation Loss',
            mode='lines',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Model Training History',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_predictions(self):
        """Plot actual vs predicted values."""
        fig = go.Figure()
        
        # Training data
        fig.add_trace(go.Scatter(
            x=self.dates_train,
            y=self.y_train_original,
            mode='lines',
            name='Actual (Train)',
            line=dict(color='blue')
        ))
        
        # Test data
        fig.add_trace(go.Scatter(
            x=self.dates_test,
            y=self.y_test_original,
            mode='lines',
            name='Actual (Test)',
            line=dict(color='green')
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=self.dates_train,
            y=self.y_pred_train,
            mode='lines',
            name='Predicted (Train)',
            line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.dates_test,
            y=self.y_pred,
            mode='lines',
            name='Predicted (Test)',
            line=dict(color='orange', dash='dash')
        ))
        
        fig.update_layout(
            title='Apple Stock Price: Actual vs Predicted',
            xaxis_title='Date',
            yaxis_title='Stock Price ($)',
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_historical_data(self):
        """Plot historical stock data."""
        fig = px.line(
            x=self.data['date'],
            y=self.data['close'],
            title='Apple Stock Price History'
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Stock Price ($)',
            template='plotly_white'
        )
        
        fig.show()
    
    def run_complete_analysis(self):
        """Run the complete stock prediction analysis."""
        print("🎯 Starting complete Apple stock prediction analysis...")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Create sequences
        self.create_sequences()
        
        # Split data
        self.split_data()
        
        # Build model
        self.build_model()
        
        # Train model
        self.train_model()
        
        # Generate predictions
        self.predict()
        
        # Calculate metrics
        rmse = self.calculate_metrics()
        
        # Predict next day
        future_pred = self.predict_next_day()
        
        print("=" * 60)
        print("🎉 Analysis completed successfully!")
        
        return {
            'rmse': rmse,
            'future_prediction': future_pred,
            'model': self.model,
            'history': self.history
        }

def main():
    """Main function to run the stock prediction."""
    # Initialize predictor
    predictor = AppleStockPredictor()
    
    # Run complete analysis
    results = predictor.run_complete_analysis()
    
    # Plot results
    print("\n📈 Generating visualizations...")
    predictor.plot_historical_data()
    predictor.plot_training_history()
    predictor.plot_predictions()
    
    print("\n🎯 Analysis Summary:")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Next Day Prediction: ${results['future_prediction']['predicted_close']:.2f}")

if __name__ == "__main__":
    main()
