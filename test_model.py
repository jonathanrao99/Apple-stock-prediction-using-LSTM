#!/usr/bin/env python3
"""
Test script for Apple Stock Prediction Model
Comprehensive testing of the LSTM model and utilities.
"""

import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from apple_stock_predictor import AppleStockPredictor
from utils import analyze_stock_data, create_technical_indicators, calculate_model_metrics
from config import MODEL_CONFIG, DATA_CONFIG

class TestAppleStockPredictor(unittest.TestCase):
    """Test cases for AppleStockPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = AppleStockPredictor()
        
    def test_data_loading(self):
        """Test data loading functionality."""
        try:
            data = self.predictor.load_and_preprocess_data()
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            print("✅ Data loading test passed")
        except Exception as e:
            self.fail(f"Data loading failed: {str(e)}")
    
    def test_sequence_creation(self):
        """Test sequence creation functionality."""
        try:
            self.predictor.load_and_preprocess_data()
            X, y, dates = self.predictor.create_sequences()
            
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertGreater(len(X), 0)
            self.assertEqual(len(X), len(y))
            print("✅ Sequence creation test passed")
        except Exception as e:
            self.fail(f"Sequence creation failed: {str(e)}")
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        try:
            self.predictor.load_and_preprocess_data()
            self.predictor.create_sequences()
            X_train, X_test, y_train, y_test = self.predictor.split_data()
            
            self.assertGreater(len(X_train), 0)
            self.assertGreater(len(X_test), 0)
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            print("✅ Data splitting test passed")
        except Exception as e:
            self.fail(f"Data splitting failed: {str(e)}")
    
    def test_model_building(self):
        """Test model building functionality."""
        try:
            self.predictor.load_and_preprocess_data()
            self.predictor.create_sequences()
            self.predictor.split_data()
            model = self.predictor.build_model()
            
            self.assertIsInstance(model, tf.keras.Model)
            print("✅ Model building test passed")
        except Exception as e:
            self.fail(f"Model building failed: {str(e)}")
    
    def test_model_training(self):
        """Test model training functionality."""
        try:
            self.predictor.load_and_preprocess_data()
            self.predictor.create_sequences()
            self.predictor.split_data()
            self.predictor.build_model()
            history = self.predictor.train_model(epochs=5, verbose=0)
            
            self.assertIsInstance(history, tf.keras.callbacks.History)
            self.assertIn('loss', history.history)
            print("✅ Model training test passed")
        except Exception as e:
            self.fail(f"Model training failed: {str(e)}")
    
    def test_prediction(self):
        """Test prediction functionality."""
        try:
            self.predictor.load_and_preprocess_data()
            self.predictor.create_sequences()
            self.predictor.split_data()
            self.predictor.build_model()
            self.predictor.train_model(epochs=5, verbose=0)
            y_pred, y_pred_train = self.predictor.predict()
            
            self.assertIsInstance(y_pred, list)
            self.assertIsInstance(y_pred_train, list)
            self.assertGreater(len(y_pred), 0)
            print("✅ Prediction test passed")
        except Exception as e:
            self.fail(f"Prediction failed: {str(e)}")

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(50, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Adj Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 5000000, 100)
        })
    
    def test_analyze_stock_data(self):
        """Test stock data analysis."""
        try:
            analysis = analyze_stock_data(self.sample_data)
            
            self.assertIsInstance(analysis, dict)
            self.assertIn('basic_stats', analysis)
            self.assertIn('price_stats', analysis)
            self.assertIn('volume_stats', analysis)
            print("✅ Stock data analysis test passed")
        except Exception as e:
            self.fail(f"Stock data analysis failed: {str(e)}")
    
    def test_create_technical_indicators(self):
        """Test technical indicators creation."""
        try:
            data_with_indicators = create_technical_indicators(self.sample_data)
            
            self.assertIsInstance(data_with_indicators, pd.DataFrame)
            self.assertIn('MA_20', data_with_indicators.columns)
            self.assertIn('RSI', data_with_indicators.columns)
            self.assertIn('MACD', data_with_indicators.columns)
            print("✅ Technical indicators test passed")
        except Exception as e:
            self.fail(f"Technical indicators creation failed: {str(e)}")
    
    def test_calculate_model_metrics(self):
        """Test model metrics calculation."""
        try:
            y_true = np.random.uniform(100, 200, 50)
            y_pred = y_true + np.random.normal(0, 5, 50)
            
            metrics = calculate_model_metrics(y_true, y_pred)
            
            self.assertIsInstance(metrics, dict)
            self.assertIn('rmse', metrics)
            self.assertIn('mae', metrics)
            self.assertIn('r2', metrics)
            print("✅ Model metrics calculation test passed")
        except Exception as e:
            self.fail(f"Model metrics calculation failed: {str(e)}")

def run_performance_test():
    """Run performance tests."""
    print("\n🚀 Running Performance Tests...")
    
    try:
        # Initialize predictor
        predictor = AppleStockPredictor(window_size=30, test_size_ratio=0.1)
        
        # Load data
        print("📊 Loading data...")
        predictor.load_and_preprocess_data()
        
        # Create sequences
        print("🔄 Creating sequences...")
        predictor.create_sequences()
        
        # Split data
        print("✂️ Splitting data...")
        predictor.split_data()
        
        # Build model
        print("🧠 Building model...")
        predictor.build_model()
        
        # Train model
        print("🚀 Training model...")
        history = predictor.train_model(epochs=10, verbose=0)
        
        # Generate predictions
        print("🔮 Generating predictions...")
        predictor.predict()
        
        # Calculate metrics
        rmse = predictor.calculate_metrics()
        
        print(f"✅ Performance test completed! RMSE: {rmse:.4f}")
        
        # Performance benchmarks
        if rmse < 10:
            print("🎉 Excellent performance! RMSE < 10")
        elif rmse < 20:
            print("👍 Good performance! RMSE < 20")
        elif rmse < 50:
            print("📈 Acceptable performance! RMSE < 50")
        else:
            print("⚠️ Performance needs improvement! RMSE >= 50")
            
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")

def main():
    """Main test function."""
    print("�� Starting Apple Stock Predictor Tests...")
    print("=" * 50)
    
    # Run unit tests
    print("\n📋 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance test
    run_performance_test()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
