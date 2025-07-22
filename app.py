"""
Streamlit Web Application for Apple Stock Prediction
A user-friendly web interface for the LSTM stock prediction model.
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

# Import config after warning suppression
import config

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("❌ TensorFlow is not available. Please install it using: `pip install tensorflow`")

# Try to import sklearn
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("❌ Scikit-learn is not available. Please install it using: `pip install scikit-learn`")

# Import our custom modules with error handling
try:
    from apple_stock_predictor import AppleStockPredictor
    from utils import create_technical_indicators
    from config import MODEL_CONFIG, DATA_CONFIG
    CUSTOM_MODULES_AVAILABLE = True
except ImportError as e:
    CUSTOM_MODULES_AVAILABLE = False
    st.error(f"❌ Error importing custom modules: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="🍎 Apple Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Check if all dependencies are available
    if not all([TENSORFLOW_AVAILABLE, SKLEARN_AVAILABLE, CUSTOM_MODULES_AVAILABLE]):
        st.error("🚫 Some required dependencies are missing. Please check the error messages above and install the missing packages.")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🍎 Apple Stock Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Stock Price Prediction using LSTM Neural Networks")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    window_size = st.sidebar.slider("Window Size", 10, 100, MODEL_CONFIG['window_size'])
    epochs = st.sidebar.slider("Training Epochs", 10, 50, MODEL_CONFIG['epochs'])
    test_size = st.sidebar.slider("Test Size Ratio", 0.05, 0.3, MODEL_CONFIG['test_size_ratio'])
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_technical_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)
    show_model_evaluation = st.sidebar.checkbox("Show Model Evaluation", value=True)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["�� Data Analysis", "🧠 Model Training", "🔮 Predictions", "📈 Technical Analysis"])
    
    with tab1:
        st.header("📊 Data Analysis")
        
        # Load data
        try:
            data = pd.read_csv(DATA_CONFIG['data_path'], parse_dates=[DATA_CONFIG['date_column']])
            st.success("✅ Data loaded successfully!")
            
            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Date Range", f"{data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
            with col4:
                total_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            
            # Stock price chart
            st.subheader("📈 Stock Price History")
            fig = px.line(data, x='Date', y='Close', title='Apple Stock Price Over Time')
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Data statistics
            st.subheader("📋 Data Statistics")
            st.dataframe(data.describe())
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    with tab2:
        st.header("🧠 Model Training")
        
        if st.button("🚀 Train LSTM Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Initialize predictor
                    predictor = AppleStockPredictor(
                        window_size=window_size,
                        test_size_ratio=test_size
                    )
                    
                    # Load and preprocess data
                    predictor.load_and_preprocess_data()
                    predictor.create_sequences()
                    predictor.split_data()
                    predictor.build_model()
                    
                    # Train model
                    history = predictor.train_model(epochs=epochs, verbose=0)
                    
                    # Generate predictions
                    predictor.predict()
                    
                    # Calculate metrics
                    rmse = predictor.calculate_metrics()
                    
                    # Store in session state
                    st.session_state['predictor'] = predictor
                    st.session_state['rmse'] = rmse
                    st.session_state['history'] = history
                    
                    st.success("✅ Model training completed!")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with col2:
                        st.metric("Final Loss", f"{history.history['loss'][-1]:.4f}")
                    with col3:
                        st.metric("Final Val Loss", f"{history.history['val_loss'][-1]:.4f}")
                    
                    # Training history plot
                    st.subheader("📊 Training History")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                    fig.update_layout(title='Model Training History', template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
        else:
            st.info("👆 Click the button above to start training the model.")
    
    with tab3:
        st.header("🔮 Predictions")
        
        if 'predictor' in st.session_state:
            predictor = st.session_state['predictor']
            
            # Plot predictions
            st.subheader("📈 Actual vs Predicted Prices")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictor.dates_train,
                y=predictor.y_train_original,
                mode='lines',
                name='Actual (Train)',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=predictor.dates_test,
                y=predictor.y_test_original,
                mode='lines',
                name='Actual (Test)',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=predictor.dates_train,
                y=predictor.y_pred_train,
                mode='lines',
                name='Predicted (Train)',
                line=dict(color='red', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=predictor.dates_test,
                y=predictor.y_pred,
                mode='lines',
                name='Predicted (Test)',
                line=dict(color='orange', dash='dash')
            ))
            
            fig.update_layout(
                title='Apple Stock Price: Actual vs Predicted',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Next day prediction
            if st.button("🔮 Predict Next Day"):
                future_pred = predictor.predict_next_day()
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🎯 Next Day Prediction</h3>
                    <p><strong>Date:</strong> {future_pred['date'].strftime('%Y-%m-%d')}</p>
                    <p><strong>Predicted Price:</strong> ${future_pred['predicted_close']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Model performance
            st.subheader("📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{st.session_state['rmse']:.4f}")
            with col2:
                current_price = predictor.data['close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
            with col3:
                if 'future_pred' in locals():
                    price_change = future_pred['predicted_close'] - current_price
                    st.metric("Predicted Change", f"${price_change:.2f}")
        
        else:
            st.info("👆 Please train the model first in the 'Model Training' tab.")
    
    with tab4:
        st.header("📈 Technical Analysis")
        
        if st.button("📊 Generate Technical Analysis"):
            try:
                data = pd.read_csv(DATA_CONFIG['data_path'], parse_dates=[DATA_CONFIG['date_column']])
                data_with_indicators = create_technical_indicators(data)
                
                # Technical indicators
                st.subheader("📊 Technical Indicators")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI
                    fig_rsi = px.line(data_with_indicators, x='Date', y='RSI', title='RSI (Relative Strength Index)')
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(template='plotly_white')
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    # MACD
                    fig_macd = px.line(data_with_indicators, x='Date', y=['MACD', 'MACD_Signal'], 
                                      title='MACD (Moving Average Convergence Divergence)')
                    fig_macd.update_layout(template='plotly_white')
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Moving averages
                st.subheader("📈 Moving Averages")
                fig_ma = px.line(data_with_indicators, x='Date', y=['Close', 'MA_20', 'MA_50'], 
                                title='Stock Price with Moving Averages')
                fig_ma.update_layout(template='plotly_white')
                st.plotly_chart(fig_ma, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error generating technical analysis: {str(e)}")

if __name__ == "__main__":
    main()
