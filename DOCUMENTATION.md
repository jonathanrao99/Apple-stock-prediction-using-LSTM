# 📚 Apple Stock Predictor - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation Guide](#installation-guide)
4. [Usage Guide](#usage-guide)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)

## Project Overview

The Apple Stock Predictor is a sophisticated machine learning application that uses Long Short-Term Memory (LSTM) neural networks to predict Apple Inc. stock prices. The project combines advanced deep learning techniques with comprehensive data analysis and visualization capabilities.

### Key Features
- **Multi-variate LSTM Model**: Uses multiple stock features for prediction
- **Interactive Web Interface**: Streamlit-based dashboard
- **Technical Analysis**: Comprehensive technical indicators
- **Real-time Predictions**: Next-day price forecasting
- **Performance Metrics**: Detailed model evaluation
- **Docker Support**: Easy deployment and scaling

## Architecture

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Model Layer    │    │  Interface      │
│                 │    │                 │    │                 │
│ • AAPL.csv      │───▶│ • LSTM Model    │───▶│ • Streamlit App │
│ • Data Loading  │    │ • Preprocessing │    │ • Jupyter NB    │
│ • Preprocessing │    │ • Training      │    │ • Python Script │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **Data Loading**: CSV file containing historical stock data
2. **Preprocessing**: Feature scaling and sequence creation
3. **Model Training**: LSTM neural network training
4. **Prediction**: Generate forecasts and evaluate performance
5. **Visualization**: Interactive charts and analysis

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- Git
- Docker (optional, for containerized deployment)

### Local Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd apple-stock-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

### Docker Installation

1. **Build and run with Docker Compose**
```bash
./deploy.sh
```

2. **Or manually**
```bash
docker-compose up --build -d
```

## Usage Guide

### Quick Start

1. **Run the web application**
```bash
streamlit run app.py
```

2. **Or run the Python script**
```bash
python apple_stock_predictor.py
```

3. **Or use Jupyter Notebook**
```bash
jupyter notebook Applestockprediction.ipynb
```

### Web Interface Usage

1. **Data Analysis Tab**
   - View historical stock data
   - Analyze basic statistics
   - Explore price trends

2. **Model Training Tab**
   - Configure model parameters
   - Train the LSTM model
   - Monitor training progress

3. **Predictions Tab**
   - View actual vs predicted prices
   - Generate next-day predictions
   - Analyze model performance

4. **Technical Analysis Tab**
   - View technical indicators
   - Analyze RSI, MACD, moving averages
   - Identify trading signals

### Command Line Usage

```python
from apple_stock_predictor import AppleStockPredictor

# Initialize predictor
predictor = AppleStockPredictor()

# Run complete analysis
results = predictor.run_complete_analysis()

# Access results
print(f"RMSE: {results['rmse']}")
print(f"Next day prediction: ${results['future_prediction']['predicted_close']:.2f}")
```

## API Reference

### AppleStockPredictor Class

#### Constructor
```python
AppleStockPredictor(data_path='AAPL.csv', window_size=50, test_size_ratio=0.1)
```

**Parameters:**
- `data_path` (str): Path to the CSV data file
- `window_size` (int): Number of time steps for LSTM
- `test_size_ratio` (float): Ratio of data for testing

#### Methods

##### `load_and_preprocess_data()`
Load and preprocess the stock data.

**Returns:** pandas.DataFrame

##### `create_sequences()`
Create time-series sequences for LSTM training.

**Returns:** tuple (X, y, dates)

##### `split_data()`
Split data into training and testing sets.

**Returns:** tuple (X_train, X_test, y_train, y_test)

##### `build_model()`
Build and compile the LSTM model.

**Returns:** tensorflow.keras.Model

##### `train_model(epochs=25, verbose=1)`
Train the LSTM model.

**Parameters:**
- `epochs` (int): Number of training epochs
- `verbose` (int): Verbosity level

**Returns:** tensorflow.keras.callbacks.History

##### `predict()`
Generate predictions for training and test sets.

**Returns:** tuple (y_pred, y_pred_train)

##### `predict_next_day()`
Predict the next day's stock price.

**Returns:** dict with 'date' and 'predicted_close'

##### `calculate_metrics()`
Calculate performance metrics.

**Returns:** float (RMSE)

##### `plot_predictions()`
Plot actual vs predicted values.

##### `run_complete_analysis()`
Run the complete stock prediction analysis.

**Returns:** dict with results

### Utility Functions

#### `analyze_stock_data(data)`
Perform comprehensive analysis of stock data.

**Parameters:**
- `data` (pd.DataFrame): Stock data

**Returns:** dict with analysis results

#### `create_technical_indicators(data)`
Create technical indicators for stock data.

**Parameters:**
- `data` (pd.DataFrame): Stock data

**Returns:** pd.DataFrame with technical indicators

#### `calculate_model_metrics(y_true, y_pred)`
Calculate comprehensive model evaluation metrics.

**Parameters:**
- `y_true` (array): True values
- `y_pred` (array): Predicted values

**Returns:** dict with metrics

## Configuration

### Model Configuration (config.py)

```python
MODEL_CONFIG = {
    'window_size': 50,           # Time steps for LSTM
    'test_size_ratio': 0.1,      # Test data ratio
    'lstm_units_1': 500,         # First LSTM layer units
    'lstm_units_2': 100,         # Second LSTM layer units
    'dropout_rate': 0.2,         # Dropout rate
    'learning_rate': 0.001,      # Learning rate
    'batch_size': 32,            # Batch size
    'epochs': 25,                # Training epochs
    'validation_split': 0.1,     # Validation split
}
```

### Training Configuration

```python
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'loss_function': 'mean_squared_error',
    'metrics': ['mae'],
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7
}
```

## Deployment

### Local Deployment

1. **Development Mode**
```bash
streamlit run app.py
```

2. **Production Mode**
```bash
python apple_stock_predictor.py
```

### Docker Deployment

1. **Using Docker Compose**
```bash
docker-compose up --build -d
```

2. **Using Docker directly**
```bash
docker build -t apple-stock-predictor .
docker run -p 8501:8501 apple-stock-predictor
```

### Cloud Deployment

#### Heroku
1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:
```bash
heroku create
git push heroku main
```

#### AWS/GCP/Azure
Use the provided Dockerfile and docker-compose.yml for container deployment.

## Troubleshooting

### Common Issues

#### 1. Memory Issues
**Problem:** Out of memory during training
**Solution:** Reduce batch size or window size in config.py

#### 2. CUDA Issues
**Problem:** TensorFlow GPU errors
**Solution:** Install CPU-only version:
```bash
pip install tensorflow-cpu
```

#### 3. Data Loading Issues
**Problem:** File not found errors
**Solution:** Ensure AAPL.csv is in the project directory

#### 4. Streamlit Issues
**Problem:** App not loading
**Solution:** Check port availability and firewall settings

### Performance Optimization

1. **Reduce Model Complexity**
   - Decrease LSTM units
   - Reduce window size
   - Use fewer epochs

2. **Data Optimization**
   - Use smaller dataset
   - Reduce feature count
   - Implement data sampling

3. **Hardware Optimization**
   - Use GPU acceleration
   - Increase RAM
   - Use SSD storage

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
```bash
git checkout -b feature/new-feature
```

3. **Make changes and test**
```bash
python test_model.py
```

4. **Commit changes**
```bash
git commit -m "Add new feature"
```

5. **Push and create PR**

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings
- Write unit tests

### Testing

Run tests:
```bash
python test_model.py
```

### Documentation

- Update README.md for user-facing changes
- Update DOCUMENTATION.md for technical changes
- Add inline comments for complex logic

---

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

*This documentation is maintained by the development team.*
