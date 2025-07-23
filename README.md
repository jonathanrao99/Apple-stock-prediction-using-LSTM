# 🍎 Apple Stock Predictor

> **Predict Apple's stock price like a pro!** 📈

A lightning-fast LSTM neural network that predicts Apple stock prices with real-time training progress and a sleek web interface.

## 🚀 Quick Start

```bash
# Clone & setup
git clone <your-repo>
cd Apple-stock-prediction-using-LSTM
pip install -r requirements.txt

# Launch the magic! ✨
python quick_start.py
```

## ☁️ Streamlit Cloud Deployment

This app is ready for Streamlit Cloud! Just connect your GitHub repo and deploy:

1. **Fork/Clone** this repository
2. **Connect** to Streamlit Cloud
3. **Deploy** - it will automatically use Python 3.10 and TensorFlow 2.13.0

## 🎯 What's Cool About This?

- **🤖 Smart AI**: LSTM neural network with 40+ years of Apple data (1980-2022)
- **⚡ Optimized Training**: Early stopping, learning rate scheduling, dropout regularization
- **🌐 Web Interface**: Beautiful Streamlit dashboard with real-time progress
- **📊 10,468 Data Points**: Massive dataset for better predictions
- **🎨 Interactive Charts**: Plotly-powered visualizations
- **⚙️ Zero Warnings**: Clean, optimized codebase

## 🎮 How to Use

### Option 1: Interactive Menu
```bash
python quick_start.py
# Choose: Web Interface (1) or Full Analysis (2)
```

### Option 2: Web Interface Only
```bash
streamlit run app.py
# Open: http://localhost:8501
```

### Option 3: Console Analysis
```bash
python apple_stock_predictor.py
```

## 🏗️ Architecture

```
📊 Data (1980-2022) → 🧠 LSTM Model → 🔮 Predictions
     ↓                      ↓              ↓
  10,468 points        500→100 units    RMSE: ~$103
```

## 🎨 Features

- **Real-time Training Progress**: Watch your model learn live! 🎯
- **Technical Indicators**: RSI, MACD, Moving Averages 📈
- **Performance Metrics**: RMSE, Loss tracking 📊
- **Next Day Predictions**: See tomorrow's price today 🔮
- **Responsive Design**: Works on desktop & mobile 📱

## 🛠️ Tech Stack

- **Python 3.13** + **TensorFlow 2.21** (nightly)
- **Streamlit** for web interface
- **Plotly** for interactive charts
- **Pandas** + **NumPy** for data processing
- **Scikit-learn** for scaling & metrics

## 📈 Sample Output

```
🎯 Analysis Summary:
RMSE: 102.9965
Next Day Prediction: $151.21
Training Time: ~15 minutes
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| TensorFlow Import Error | `pip install tf-nightly` |
| Streamlit Cloud Error | Use Python 3.10 (runtime.txt included) |
| Port Already in Use | Change port in `app.py` |
| Missing Dependencies | `pip install -r requirements.txt` |

## 📊 Performance

- **Data Points**: 10,468 (vs 3,126 before)
- **Training Time**: ~15 minutes (optimized)
- **Model Size**: 1.25M parameters
- **Accuracy**: RMSE ~$103

## 🤝 Contributing

Found a bug? Want to add features? PRs welcome! 🎉

---

**Made with ❤️ and ☕ by AI enthusiasts**
