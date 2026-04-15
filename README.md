# 🚗 Car Insurance Claim Prediction

An AI-powered fraud detection system built using ensemble machine learning techniques. This project comes with a premium, fully responsive, dark-themed **Streamlit** dashboard that provides real-time fraud probability scoring, interactive data visualizations, and robust model metric comparisons.

## ✨ Features

- **Dual Ensemble Models**: Compare predictions side-by-side between **Random Forest** (200 trees) and **XGBoost** (200 rounds).
- **Live Interactive Dashboard**: Built elegantly with Streamlit, completely unbranded (no watermarks) and responsive.
- **Deep Performance Insights**: Automatic rendering of Confusion Matrices, ROC-AUC curves, and Feature Importance metric charts.
- **Real-Time Fraud Risk Scoring**: Form interface where users can enter real-world scenario metrics (in INR) to generate predictions on the fly.
- **Data Explainer**: Hoverable visual breakdowns explaining exactly *why* a particular claim was classified as legit or fraudulent based on your specific input.
- **Session History Logging**: Tracks the latest predictions you've made directly in the sidebar for audit trailing.

## 🛠️ Technology Stack

- **Frontend Application:** Streamlit
- **Machine Learning Tooling:** Scikit-Learn, XGBoost
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Plotly (Express & Graph Objects)
- **Model Serialization:** Joblib

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You also need to install the project dependencies.

```bash
pip install -r requirements.txt
```

### 2. Train the Models
Before firing up the application, you need to execute the training script. This script processes your initial dataset, trains the Random Forest and XGBoost models, and serializes (saves) them along with the required encoders into `.pkl` format.

```bash
python train_models.py
```

### 3. Run the Dashboard
Once the `pkl` files are successfully generated in your directory, you can launch the Streamlit frontend.

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser.

## 📁 Project Structure

```
d:/DA_project/
├── app.py                   # Main Streamlit dashboard application file
├── train_models.py          # Script for building, training, and saving the ML models
├── requirements.txt         # Project dependencies list
├── generate_notebook.py     # Utility script for project notebooks
└── *.pkl                    # Pre-trained models and label encoders (generated post-training)
```

## 🎨 UI/UX Features Built-in

- Completely tailored **Dark Mode** via customized global CSS formatting.
- Integrated **Plotly Dark Template** for native-looking visual cohesiveness.
- Robust overrides to ensure Streamlit's Material Icons aren't stripped or text-merged.
- Removed arbitrary Streamlit popover footer branding for a cleaner, more professional look.

## 🛡️ License

This project is licensed under the MIT License.
