#!/usr/bin/env python3
"""
Quick Start Script for Apple Stock Predictor
A simple script to get started with the stock prediction model.
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

import sys
import os
import subprocess
import time

# Type hints for Pylance
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    import tensorflow as tf

# Import TensorFlow for Pylance resolution (will be caught by try-except if not available)
try:
    import tensorflow as tf  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore

def print_banner():
    """Print the application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🍎 Apple Stock Predictor                  ║
    ║                                                              ║
    ║              AI-Powered Stock Price Prediction               ║
    ║              Using LSTM Neural Networks                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('plotly', 'plotly'),
        ('streamlit', 'streamlit')
    ]
    
    missing_packages = []
    
    # Check non-TensorFlow packages first
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} (missing)")
            missing_packages.append(package_name)
    
    # Check scikit-learn separately
    try:
        import sklearn
        print(f"  ✅ scikit-learn")
    except ImportError:
        print(f"  ❌ scikit-learn (missing)")
        missing_packages.append('scikit-learn')
    
    # Check TensorFlow using pre-imported status
    if TENSORFLOW_AVAILABLE:
        print(f"  ✅ tensorflow")
    else:
        print(f"  ❌ tensorflow (missing)")
        missing_packages.append('tensorflow')
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run:")
            print("   pip install -r requirements.txt")
            return False
    
    return True

def check_data():
    """Check if required data files exist."""
    print("\n📊 Checking data files...")
    
    if os.path.exists('AAPL.csv'):
        print("  ✅ AAPL.csv found")
        return True
    else:
        print("  ❌ AAPL.csv not found")
        print("📥 Please ensure AAPL.csv is in the current directory")
        return False



def show_menu():
    """Show the main menu."""
    menu = """
    🎯 Choose an option:
    
    1. 🌐 Launch Web Interface
    2. 📊 Run Full Analysis
    3. 🚪 Exit
    
    Enter your choice (1-3): """
    
    while True:
        choice = input(menu).strip()
        
        if choice == '1':
            print("\n🌐 Launching web interface...")
            print("📱 Open your browser and go to: http://localhost:8503")
            print("⏹️  Press Ctrl+C to stop the server")
            try:
                # Use the same Python executable to ensure same environment
                subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', '8503'])
            except KeyboardInterrupt:
                print("\n👋 Web interface stopped")
            except Exception as e:
                print(f"❌ Failed to launch web interface: {str(e)}")
                print("💡 Try running manually: py -m streamlit run app.py")
        elif choice == '2':
            print("\n📊 Running full analysis...")
            try:
                from apple_stock_predictor import AppleStockPredictor
                predictor = AppleStockPredictor()
                results = predictor.run_complete_analysis()
                print(f"\n📊 Results Summary:")
                print(f"  RMSE: {results['rmse']:.4f}")
                print(f"  Next Day Prediction: ${results['future_prediction']['predicted_close']:.2f}")
                print(f"  Prediction Date: {results['future_prediction']['date'].strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"❌ Analysis failed: {str(e)}")
        elif choice == '3':
            print("\n👋 Thanks for using Apple Stock Predictor!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-3.")
        
        input("\nPress Enter to continue...")

def main():
    """Main function."""
    print_banner()
    
    print("🎉 Welcome to Apple Stock Predictor!")
    print("This tool will help you get started with stock price prediction.\n")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Setup failed. Please install dependencies manually.")
        return
    
    # Check data
    if not check_data():
        print("❌ Setup failed. Please ensure data files are available.")
        return
    
    print("\n✅ Setup completed successfully!")
    
    # Show menu
    show_menu()

if __name__ == "__main__":
    main()
