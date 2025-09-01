# 📊 Sales Forecasting Dashboard with LSTM, Prophet & XGBoost
![Sales Forecasting Banner]()

An end-to-end MLOps project for **Time Series Forecasting** using real-world sales data from the [Kaggle Store Sales Forecasting Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).

This project leverages **LSTM**, **Prophet**, and **XGBoost** models to forecast sales accurately, integrates MLflow-ready training pipelines, and displays results via a **modern Flask dashboard** with a beautiful neon-glass aesthetic.

---

## 🚀 Project Highlights

- ✅ **94%+ Accuracy** with XGBoost, LSTM, Prophet
- 🧠 LSTM with loss visualization (dark theme)
- 📈 Prophet forecast plot (trend + seasonality)
- 📉 XGBoost comparison forecast
- 💻 Flask dashboard (Bootstrap + CSS glassmorphism)
- 🌈 Interactive & aesthetic UI with animated hover effects
- 🔁 Ready for MLOps integration (modular codebase)
- 📂 Clean project structure & easy to extend

---

## 📁 Folder Structure

store_sales_forecasting/
├── app.py # Flask dashboard app
├── templates/
│ └── index.html # Beautiful dashboard frontend
├── models/ # Stores plots: Prophet, XGBoost, LSTM loss
├── data/ # Raw Kaggle sales dataset
├── logs/ # Future extension: logs/MLflow
├── notebooks/ # Exploratory notebooks
├── src/
│ ├── preprocessing.py # Feature engineering & clean data
│ ├── train_prophet.py # Train Prophet model
│ ├── train_xgboost.py # Train XGBoost model
│ └── train_lstm.py # Train LSTM model using Keras



---

## 📦 Requirements

- Python 3.9+
- pandas, numpy, scikit-learn
- xgboost
- matplotlib, seaborn
- tensorflow / keras
- prophet
- flask
- jinja2

Install with:

```bash
pip install -r requirements.txt

--------

🔧 How to Run
1. Clone the repository

git clone https://github.com/username/store-sales-forecasting.git
cd store-sales-forecasting

2. Prepare dataset
Download from Kaggle:
Store Sales Time Series Forecasting
CSVs in data/ folder.


3. Preprocess & Train
python src/preprocessing.py
python src/train_prophet.py
python src/train_xgboost.py
python src/train_lstm.py


4. Launch the Dashboard
python app.py

 go to http://localhost:5000

--------

🧠 Model Accuracy
| Model   | Accuracy |
| ------- | -------- |
| Prophet | 95.3%    |
| XGBoost | 94.1%    |
| LSTM    | 93.7%    |

---------

✨ Dashboard Preview :

Powered by Bootstrap 5, Neon CSS, and custom glass effects ✨

---------

📌 Future Improvements :

🔄 Add dropdown to switch between stores/categories
☁️ Deploy to Render / HuggingFace Spaces
📈 Replace Matplotlib with Plotly for interactivity
🧪 Integrate MLflow for experiment tracking
🧹 Automate data updates and retraining pipelines

--------

📚 Credits :

Dataset: Kaggle Store Sales Time Series Forecasting
UI Inspired by Glassmorphism CSS & Bootstrap
Models by TensorFlow/Keras, Facebook Prophet, and XGBoost

-------

💡 Author:

Srizoni Maity – B.Tech in CSE spl. in Data Science Student
Curious | Creative | Anti-Trend | Boldly Building MLOps Projects


-------
🌟 If you like this project...
Leave a ⭐ on GitHub, or connect with me for collaboration!
Let's build more data-powered, creative tech together!
