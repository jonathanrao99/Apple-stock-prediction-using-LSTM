#!/usr/bin/env python3
"""
Quick Start Script for Apple Stock Predictor
A simple script to get started with the stock prediction model.
"""

import sys
import os
import subprocess
import time

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
        'pandas', 'numpy', 'tensorflow', 'scikit-learn', 
        'plotly', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
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

def run_demo():
    """Run a quick demo of the stock predictor."""
    print("\n🚀 Running quick demo...")
    
    try:
        from apple_stock_predictor import AppleStockPredictor
        
        # Initialize with smaller parameters for quick demo
        predictor = AppleStockPredictor(window_size=30, test_size_ratio=0.1)
        
        print("  📊 Loading data...")
        predictor.load_and_preprocess_data()
        
        print("  🔄 Creating sequences...")
        predictor.create_sequences()
        
        print("  ✂️ Splitting data...")
        predictor.split_data()
        
        print("  🧠 Building model...")
        predictor.build_model()
        
        print("  🚀 Training model (5 epochs for demo)...")
        history = predictor.train_model(epochs=5, verbose=0)
        
        print("  🔮 Generating predictions...")
        predictor.predict()
        
        rmse = predictor.calculate_metrics()
        future_pred = predictor.predict_next_day()
        
        print("\n📊 Demo Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Next Day Prediction: ${future_pred['predicted_close']:.2f}")
        print(f"  Prediction Date: {future_pred['date'].strftime('%Y-%m-%d')}")
        
        print("\n✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False

def show_menu():
    """Show the main menu."""
    menu = """
    🎯 Choose an option:
    
    1. 🚀 Run Quick Demo
    2. 🌐 Launch Web Interface
    3. 📊 Run Full Analysis
    4. 🔮 Predict Next Day
    5. 📈 Technical Analysis
    6. 🧪 Run Tests
    7. 📚 View Documentation
    8. 🚪 Exit
    
    Enter your choice (1-8): """
    
    while True:
        choice = input(menu).strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            print("\n🌐 Launching web interface...")
            print("📱 Open your browser and go to: http://localhost:8501")
            print("⏹️  Press Ctrl+C to stop the server")
            try:
                subprocess.run(['streamlit', 'run', 'app.py'])
            except KeyboardInterrupt:
                print("\n👋 Web interface stopped")
        elif choice == '3':
            print("\n📊 Running full analysis...")
            subprocess.run([sys.executable, 'cli.py', '--full'])
        elif choice == '4':
            print("\n🔮 Predicting next day...")
            subprocess.run([sys.executable, 'cli.py', '--predict-next'])
        elif choice == '5':
            print("\n📈 Running technical analysis...")
            subprocess.run([sys.executable, 'cli.py', '--technical'])
        elif choice == '6':
            print("\n🧪 Running tests...")
            subprocess.run([sys.executable, 'test_model.py'])
        elif choice == '7':
            print("\n📚 Documentation:")
            print("  - README.md: Project overview and setup")
            print("  - DOCUMENTATION.md: Complete technical documentation")
            print("  - Run 'python cli.py --help' for command-line options")
        elif choice == '8':
            print("\n👋 Thanks for using Apple Stock Predictor!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-8.")
        
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
