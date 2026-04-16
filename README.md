# 🚀 AI Customer Risk Analyzer

An interactive **Machine Learning web app** that predicts whether a customer is likely to churn using an Artificial Neural Network (ANN).

🔗 Built with Streamlit | TensorFlow | Scikit-learn

---

## 📊 Features

* 🔮 Predict customer churn probability
* 📈 Displays risk percentage (Churn vs Retention)
* 🎯 Clean and interactive UI dashboard
* ⚡ Real-time predictions using trained ANN model

---

## 🧠 How It Works

The model is trained on customer data and uses:

* Gender encoding
* Geography one-hot encoding
* Feature scaling (StandardScaler)
* Artificial Neural Network (ANN)

---

## 📁 Project Structure

```
.
├── app.py
├── model.ipynb
├── requirements.txt
├── artifacts/
│   ├── churn_ann_model.keras
│   ├── scaler.pkl
│   └── feature_columns.json
├── Artificial_Neural_Network_Case_Study_data.csv
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Abhi-Sh4rma/Abhi_Ann_Analyze.git
cd Abhi_Ann_Analyze
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🧪 Train the Model (Optional)

If you want to retrain the model:

1. Open `model.ipynb`
2. Run all cells
3. Ensure artifacts are generated inside `/artifacts`

---

## ▶️ Run the App Locally

```
streamlit run app.py
```

---

## 🌍 Deployment

This app is deployed using **Streamlit Community Cloud**

👉 Live App: https://abhi-sh4rma-ann-analyze.streamlit.app/

---

## 📌 Notes

* Model predicts churn probability based on user input
* Threshold:

  * ≥ 50% → High Risk
  * < 50% → Low Risk
* Make sure artifacts exist before running the app

---

## 👨‍💻 Author

**Abhishek Sharma**
Aspiring Cybersecurity & AI Enthusiast 🔐🤖

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
