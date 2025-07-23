# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository
✅ This repository is already configured for Streamlit Cloud deployment with:
- `requirements.txt` - Uses tf-nightly (compatible with Python 3.13)
- `.streamlit/config.toml` - Optimized configuration

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Select your repository**: `Apple-stock-prediction-using-LSTM`
5. **Set the main file path**: `app.py`
6. **Click "Deploy"**

### Step 3: Wait for Deployment
- First deployment takes 2-3 minutes
- Subsequent updates are faster
- Check the logs if there are any issues

## 🔧 Configuration Files



### requirements.txt
```
pandas>=1.5.0
numpy>=1.21.0
tf-nightly
scikit-learn>=1.1.0
plotly>=5.10.0
streamlit>=1.28.0
```
- Uses tf-nightly (compatible with Python 3.13)
- All other dependencies are compatible

### .streamlit/config.toml
```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🐛 Common Issues & Solutions

### Issue: "TensorFlow not found"
**Solution**: The `runtime.txt` and `requirements.txt` files should fix this automatically.

### Issue: "Python version incompatible"
**Solution**: Using tf-nightly which is compatible with Python 3.13.

### Issue: "App takes too long to load"
**Solution**: This is normal for the first load. The model needs to download and train.

## 📊 Performance Notes

- **First Load**: 2-3 minutes (model training)
- **Subsequent Loads**: 30-60 seconds
- **Memory Usage**: ~1GB RAM
- **CPU**: Multi-core processing for training

## 🔄 Updating Your App

1. **Make changes** to your local repository
2. **Commit and push** to GitHub
3. **Streamlit Cloud** automatically redeploys
4. **Check the logs** for any deployment issues

---

**Your app should now deploy successfully on Streamlit Cloud! 🎉** 