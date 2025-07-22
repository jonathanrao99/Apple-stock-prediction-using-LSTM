#!/usr/bin/env python3
"""
Command Line Interface for Apple Stock Predictor
Provides easy command-line access to the stock prediction functionality.
"""

import argparse
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from apple_stock_predictor import AppleStockPredictor
from utils import analyze_stock_data, create_technical_indicators, calculate_model_metrics
from config import MODEL_CONFIG

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='🍎 Apple Stock Predictor - LSTM-based stock price prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --quick                    # Quick prediction with default settings
  python cli.py --train --epochs 50        # Train model with 50 epochs
  python cli.py --predict-next             # Predict next day's price
  python cli.py --analyze                  # Analyze stock data
  python cli.py --technical                # Generate technical indicators
  python cli.py --full                     # Run complete analysis
        """
    )
    
    # Main options
    parser.add_argument('--quick', action='store_true', 
                       help='Quick prediction with default settings')
    parser.add_argument('--train', action='store_true',
                       help='Train the LSTM model')
    parser.add_argument('--predict-next', action='store_true',
                       help='Predict next day\'s stock price')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze stock data')
    parser.add_argument('--technical', action='store_true',
                       help='Generate technical indicators')
    parser.add_argument('--full', action='store_true',
                       help='Run complete analysis')
    
    # Model parameters
    parser.add_argument('--window-size', type=int, default=MODEL_CONFIG['window_size'],
                       help=f'Window size for LSTM (default: {MODEL_CONFIG["window_size"]})')
    parser.add_argument('--epochs', type=int, default=MODEL_CONFIG['epochs'],
                       help=f'Number of training epochs (default: {MODEL_CONFIG["epochs"]})')
    parser.add_argument('--test-size', type=float, default=MODEL_CONFIG['test_size_ratio'],
                       help=f'Test size ratio (default: {MODEL_CONFIG["test_size_ratio"]})')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--output-file', type=str,
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    try:
        # Initialize predictor
        predictor = AppleStockPredictor(
            window_size=args.window_size,
            test_size_ratio=args.test_size
        )
        
        if args.quick:
            print("🚀 Running quick prediction...")
            results = predictor.run_complete_analysis()
            print(f"✅ RMSE: {results['rmse']:.4f}")
            print(f"�� Next day prediction: ${results['future_prediction']['predicted_close']:.2f}")
        
        elif args.train:
            print("🧠 Training LSTM model...")
            predictor.load_and_preprocess_data()
            predictor.create_sequences()
            predictor.split_data()
            predictor.build_model()
            history = predictor.train_model(epochs=args.epochs, verbose=1 if args.verbose else 0)
            print(f"✅ Training completed! Final loss: {history.history['loss'][-1]:.4f}")
            
            if args.save_model:
                model_path = f"models/apple_stock_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
                predictor.model.save(model_path)
                print(f"💾 Model saved to: {model_path}")
        
        elif args.predict_next:
            print("🔮 Predicting next day's stock price...")
            predictor.load_and_preprocess_data()
            predictor.create_sequences()
            predictor.split_data()
            predictor.build_model()
            predictor.train_model(epochs=args.epochs, verbose=0)
            future_pred = predictor.predict_next_day()
            print(f"📅 Date: {future_pred['date'].strftime('%Y-%m-%d')}")
            print(f"💰 Predicted Price: ${future_pred['predicted_close']:.2f}")
        
        elif args.analyze:
            print("📊 Analyzing stock data...")
            data = pd.read_csv('AAPL.csv', parse_dates=['Date'])
            analysis = analyze_stock_data(data)
            
            print("\n📈 Price Statistics:")
            for key, value in analysis['price_stats'].items():
                if isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            
            print("\n📊 Volume Statistics:")
            for key, value in analysis['volume_stats'].items():
                if isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:,.0f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        
        elif args.technical:
            print("📈 Generating technical indicators...")
            data = pd.read_csv('AAPL.csv', parse_dates=['Date'])
            data_with_indicators = create_technical_indicators(data)
            
            print("✅ Technical indicators generated:")
            print(f"  - Moving Averages (20, 50, 200 day)")
            print(f"  - RSI (Relative Strength Index)")
            print(f"  - MACD (Moving Average Convergence Divergence)")
            print(f"  - Bollinger Bands")
            
            # Show latest values
            latest = data_with_indicators.iloc[-1]
            print(f"\n📊 Latest Values:")
            print(f"  RSI: {latest['RSI']:.2f}")
            print(f"  MACD: {latest['MACD']:.4f}")
            print(f"  MA20: ${latest['MA_20']:.2f}")
            print(f"  MA50: ${latest['MA_50']:.2f}")
        
        elif args.full:
            print("🎯 Running complete analysis...")
            results = predictor.run_complete_analysis()
            
            print("\n📊 Results Summary:")
            print(f"  RMSE: {results['rmse']:.4f}")
            print(f"  Next Day Prediction: ${results['future_prediction']['predicted_close']:.2f}")
            print(f"  Prediction Date: {results['future_prediction']['date'].strftime('%Y-%m-%d')}")
            
            if args.output_file:
                import json
                output_data = {
                    'timestamp': datetime.now().isoformat(),
                    'rmse': results['rmse'],
                    'future_prediction': {
                        'date': results['future_prediction']['date'].isoformat(),
                        'predicted_close': results['future_prediction']['predicted_close']
                    }
                }
                with open(args.output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"💾 Results saved to: {args.output_file}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
