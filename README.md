# Stop the Churn 

## 1) What this is
- Streamlit app to predict customer churn
- Trains on your CSV and visualizes risk, performance, and feature importance

## 2) Prerequisites
- Python 3.11+
- pip

## 3) Install
```bash
pip install -r requirements.txt
```

## 4) Run
- Windows (recommended):
```powershell
py -m streamlit run app.py
```
- Generic:
```bash
streamlit run app.py
```

## 5) Data requirements
- File type: CSV (`.csv` or `.CSV`)
- Required column: `churn` (values 0/1 or Yes/No)
- Helpful columns (examples): `CustomerID, Total Spend, Tenure, Support Calls, Usage Frequency, Payment Delay, Subscription Type, Contract Length, Last Interaction`
- Max file size: ~100 MB

## 6) How to use
- Open the app URL shown in the terminal (e.g., `http://localhost:8501`)
- In the sidebar:
  - Upload your CSV
  - Pick a page from Navigation
- The app will:
  - Split train/test (no leakage)
  - Engineer features and scale/encode
  - Train an ensemble (RF + LightGBM + XGBoost)
  - Show metrics (ROC AUC, F1, Precision, Recall)
  - Generate predictions and risk categories

## 7) Pages
- Overview: distributions and quick metrics
- Risk Analysis: counts, filters, top high‑risk, table
- Feature Importance: bar chart, SHAP (test set)
- Real‑time Prediction: enter one customer, get probability + SHAP impact

## 8) Export
- “Export Predictions” section → Download full predictions CSV

## 9) Troubleshooting (quick)
- “streamlit not recognized”: use `py -m streamlit run app.py`
- File won’t upload: ensure `.csv` or `.CSV`, has `churn`, size < 100 MB
- Nested expander error: refresh the page (fixed in code)
- Plotly axis error: refresh (fixed in code; using `update_xaxes/yaxes`)
- SHAP error on single input: handled; ensure enough features

## 10) Tech notes
- No data leakage: split before preprocessing; fit only on train
- Balancing: SMOTE on train set
- SHAP: TreeExplainer on RandomForest

## 11) License / contributions
- Open to issues and PRs. Use this README’s steps for local runs. 
