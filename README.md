# 🍎 Apple Stock Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Nightly-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Predict Apple's stock price like a pro! 📈**

*LSTM neural network with 40+ years of data and real-time training progress*

**🌐 [Live Web App](https://applestocklstm.streamlit.app/)**

</div>

---

## 🎯 What's This?

A **lightning-fast** LSTM neural network that predicts Apple stock prices with incredible accuracy. Think of it as your personal AI financial advisor that never sleeps! 🧠

### ✨ What You Get
- 🤖 **Smart AI**: LSTM neural network with 40+ years of Apple data (1980-2022)
- ⚡ **Optimized Training**: Early stopping, learning rate scheduling, dropout regularization
- 🌐 **Web Interface**: Beautiful Streamlit dashboard with real-time progress
- 📊 **10,468 Data Points**: Massive dataset for better predictions
- 🎨 **Interactive Charts**: Plotly-powered visualizations
- ⚙️ **Zero Warnings**: Clean, optimized codebase

---

## 🚀 Quick Start

```bash
# 1. Clone it
git clone <your-repo-url>
cd Apple-stock-prediction-using-LSTM

# 2. Install stuff
pip install -r requirements.txt

# 3. Run it!
python quick_start.py
```

**That's it!** 🎉

---

## 🎮 How to Use

### Option 1: Interactive Menu (Recommended)
```bash
python quick_start.py
# Choose: Web Interface (1) or Full Analysis (2)
```
*Perfect for beginners and pros alike*

### Option 2: Web Interface Only
```bash
streamlit run app.py
# Open: http://localhost:8501
```
*For the visual learners*

### Option 3: Console Analysis
```bash
python apple_stock_predictor.py
```
*For the terminal warriors*

---

## 📊 Sample Output

```
🎯 Analysis Summary:
RMSE: 102.9965
Next Day Prediction: $151.21
Training Time: ~15 minutes
Data Points: 10,468 (1980-2022)
Model Size: 1.25M parameters
```

---

## 🏗️ Architecture

```
📊 Data (1980-2022) → 🧠 LSTM Model → 🔮 Predictions
     ↓                      ↓              ↓
  10,468 points        500→100 units    RMSE: ~$103
```

---

## 🛠️ What's Inside

```
Apple-stock-prediction-using-LSTM/
├── 📄 app.py                      # Streamlit web interface
├── 🧠 apple_stock_predictor.py    # Core LSTM model
├── ⚙️ config.py                   # Configuration settings
├── 🔧 utils.py                    # Utility functions
├── 🚀 quick_start.py              # Main launcher
├── 📊 AAPL.csv                    # Historical stock data
└── 📋 requirements.txt            # Dependencies
```

---

## 🎨 Features

- **Real-time Training Progress**: Watch your model learn live! 🎯
- **Technical Indicators**: RSI, MACD, Moving Averages 📈
- **Performance Metrics**: RMSE, Loss tracking 📊
- **Next Day Predictions**: See tomorrow's price today 🔮
- **Responsive Design**: Works on desktop & mobile 📱

---

## 🛠️ Tech Stack

- **Python 3.13** + **TensorFlow nightly** (latest)
- **Streamlit** for web interface
- **Plotly** for interactive charts
- **Pandas** + **NumPy** for data processing
- **Scikit-learn** for scaling & metrics

---

## ☁️ Streamlit Cloud Deployment

This app is ready for Streamlit Cloud! Just connect your GitHub repo and deploy:

1. **Fork/Clone** this repository
2. **Connect** to Streamlit Cloud
3. **Deploy** - it will automatically use tf-nightly (compatible with Python 3.13)

**🌐 [Live Demo](https://applestocklstm.streamlit.app/)**

---

## 🐛 Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`
**Solution**: `pip install tf-nightly`

**Problem**: Streamlit Cloud Error
**Solution**: Use tf-nightly (compatible with Python 3.13)

**Problem**: Port Already in Use
**Solution**: Change port in `app.py`

**Problem**: Missing Dependencies
**Solution**: `pip install -r requirements.txt`

---

## 📊 Performance

- **Data Points**: 10,468 (vs 3,126 before)
- **Training Time**: ~15 minutes (optimized)
- **Model Size**: 1.25M parameters
- **Accuracy**: RMSE ~$103

---

## 🤝 Contributing

1. **Fork it** 🍴
2. **Create a branch** 🌿
3. **Make changes** ✏️
4. **Submit PR** 🚀

*Ideas welcome!* 💡

---

## ⚠️ Disclaimer

**For educational purposes only!** Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial advisors! 📈

---

<div align="center">

### 🌟 Star the Repository
If you find this project helpful, please give it a ⭐ on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/jonathanrao99/Apple-stock-prediction-using-LSTM?style=social)](https://github.com/jonathanrao99/Apple-stock-prediction-using-LSTM)

### 📞 Connect & Support
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jonathanrao99)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jonathanrao99)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support%20me-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/jonathanthota)

---

**Made with ❤️ and ☕ by Jonathan Thota**

*Predicting the future, one stock at a time! 📈*

</div>
