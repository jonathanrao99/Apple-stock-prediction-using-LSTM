# 🍎 Apple Stock Predictor

> **Predict Apple's stock price like a pro!** 📈

A lightning-fast LSTM neural network that predicts Apple stock prices with real-time training progress and a sleek web interface.

## 🌐 Live Demo

**[🚀 Try it now!](https://applestocklstm.streamlit.app/)**

## 🚀 Quick Start

```bash
# Clone & setup
git clone <your-repo>
cd Apple-stock-prediction-using-LSTM
pip install -r requirements.txt

# Launch the magic! ✨
python quick_start.py
```

## 🎯 What's Cool About This?

- **🤖 Smart AI**: LSTM neural network with 40+ years of Apple data
- **⚡ Optimized Training**: Early stopping, learning rate scheduling, dropout
- **🌐 Live Web App**: Beautiful Streamlit dashboard with real-time progress
- **📊 10,468 Data Points**: Massive dataset for better predictions
- **🎨 Interactive Charts**: Plotly-powered visualizations
- **⚙️ Zero Warnings**: Clean, optimized codebase

## 🎮 How to Use

### 🌐 Live Web App
**[🚀 Try it now!](https://applestocklstm.streamlit.app/)**

### 💻 Local Setup
```bash
python quick_start.py  # Interactive menu
# OR
streamlit run app.py   # Web interface only
# OR  
python apple_stock_predictor.py  # Console analysis
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

- **Python 3.13** + **TensorFlow nightly**
- **Streamlit** + **Plotly** for web interface
- **Pandas** + **NumPy** + **Scikit-learn**

## 📈 Sample Output

```
🎯 Analysis Summary:
RMSE: 102.9965
Next Day Prediction: $151.21
Training Time: ~15 minutes
```

## 🔧 Quick Fixes

| Issue | Fix |
|-------|-----|
| TensorFlow Error | `pip install tf-nightly` |
| Port in Use | Change port in `app.py` |
| Missing Deps | `pip install -r requirements.txt` |

## 📊 Performance

- **Data Points**: 10,468
- **Training Time**: ~15 minutes
- **Model Size**: 1.25M parameters
- **Accuracy**: RMSE ~$103

## 🤝 Contributing

Found a bug? Want to add features? PRs welcome! 🎉

---

**Made with ❤️ and ☕ by AI enthusiasts**
