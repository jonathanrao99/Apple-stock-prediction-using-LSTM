#!/usr/bin/env python3
"""
Performance Monitoring Script for Apple Stock Predictor
Monitors model performance and provides insights.
"""

import time
import psutil
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_system_info():
    """Get system information."""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent
    }

def get_model_performance():
    """Get model performance metrics."""
    try:
        from apple_stock_predictor import AppleStockPredictor
        
        start_time = time.time()
        
        # Initialize predictor with smaller parameters for monitoring
        predictor = AppleStockPredictor(window_size=30, test_size_ratio=0.1)
        
        # Load data
        data_load_time = time.time()
        predictor.load_and_preprocess_data()
        data_load_duration = time.time() - data_load_time
        
        # Create sequences
        seq_time = time.time()
        predictor.create_sequences()
        seq_duration = time.time() - seq_time
        
        # Split data
        split_time = time.time()
        predictor.split_data()
        split_duration = time.time() - split_time
        
        # Build model
        build_time = time.time()
        predictor.build_model()
        build_duration = time.time() - build_time
        
        # Train model (fewer epochs for monitoring)
        train_time = time.time()
        history = predictor.train_model(epochs=5, verbose=0)
        train_duration = time.time() - train_time
        
        # Generate predictions
        pred_time = time.time()
        predictor.predict()
        pred_duration = time.time() - pred_time
        
        # Calculate metrics
        rmse = predictor.calculate_metrics()
        
        total_duration = time.time() - start_time
        
        return {
            'data_load_duration': data_load_duration,
            'sequence_duration': seq_duration,
            'split_duration': split_duration,
            'build_duration': build_duration,
            'train_duration': train_duration,
            'prediction_duration': pred_duration,
            'total_duration': total_duration,
            'rmse': rmse,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        
    except Exception as e:
        return {'error': str(e)}

def monitor_performance():
    """Monitor overall performance."""
    print("🔍 Performance Monitoring Report")
    print("=" * 50)
    
    # System information
    print("\n💻 System Information:")
    sys_info = get_system_info()
    print(f"  CPU Cores: {sys_info['cpu_count']}")
    print(f"  Memory Total: {sys_info['memory_total'] / (1024**3):.2f} GB")
    print(f"  Memory Available: {sys_info['memory_available'] / (1024**3):.2f} GB")
    print(f"  Disk Usage: {sys_info['disk_usage']:.1f}%")
    
    # Model performance
    print("\n🧠 Model Performance:")
    model_perf = get_model_performance()
    
    if 'error' in model_perf:
        print(f"  ❌ Error: {model_perf['error']}")
    else:
        print(f"  Data Loading: {model_perf['data_load_duration']:.2f}s")
        print(f"  Sequence Creation: {model_perf['sequence_duration']:.2f}s")
        print(f"  Data Splitting: {model_perf['split_duration']:.2f}s")
        print(f"  Model Building: {model_perf['build_duration']:.2f}s")
        print(f"  Training: {model_perf['train_duration']:.2f}s")
        print(f"  Prediction: {model_perf['prediction_duration']:.2f}s")
        print(f"  Total Time: {model_perf['total_duration']:.2f}s")
        print(f"  RMSE: {model_perf['rmse']:.4f}")
        print(f"  Final Loss: {model_perf['final_loss']:.4f}")
        print(f"  Final Val Loss: {model_perf['final_val_loss']:.4f}")
        
        # Performance assessment
        print("\n📊 Performance Assessment:")
        if model_perf['total_duration'] < 30:
            print("  ⚡ Excellent: Fast execution time")
        elif model_perf['total_duration'] < 60:
            print("  👍 Good: Reasonable execution time")
        else:
            print("  ⏳ Slow: Consider optimization")
        
        if model_perf['rmse'] < 10:
            print("  🎯 Excellent: Low prediction error")
        elif model_perf['rmse'] < 20:
            print("  📈 Good: Acceptable prediction error")
        else:
            print("  ⚠️  High: Consider model improvements")
    
    # Save monitoring results
    monitoring_data = {
        'timestamp': datetime.now().isoformat(),
        'system_info': sys_info,
        'model_performance': model_perf
    }
    
    os.makedirs('logs', exist_ok=True)
    with open(f'logs/monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(monitoring_data, f, indent=2)
    
    print(f"\n💾 Monitoring results saved to logs/")

def continuous_monitoring(interval=300):  # 5 minutes
    """Continuous performance monitoring."""
    print(f"🔄 Starting continuous monitoring (interval: {interval}s)")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        while True:
            monitor_performance()
            print(f"\n⏰ Next check in {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance monitoring for Apple Stock Predictor')
    parser.add_argument('--continuous', '-c', action='store_true', 
                       help='Run continuous monitoring')
    parser.add_argument('--interval', '-i', type=int, default=300,
                       help='Monitoring interval in seconds (default: 300)')
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_monitoring(args.interval)
    else:
        monitor_performance()
