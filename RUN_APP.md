# How to Run the Application

## Quick Start

```powershell
# Navigate to project directory
cd "D:\Ai powered customer churn prediction"

# Run the application
py -m streamlit run app.py
```

## The app will be available at:
- **Local URL:** http://localhost:8501
- **Network URL:** http://192.168.25.96:8501 (may vary)

## Common Issues & Solutions

### Issue 1: "streamlit not recognized"
**Solution:** Use `py -m streamlit run app.py` instead of `streamlit run app.py`

### Issue 2: Port already in use
**Solution:** 
```powershell
# Kill existing processes
Get-Process -Name python | Stop-Process -Force
# Then restart
py -m streamlit run app.py
```

### Issue 3: Module not found errors
**Solution:** Install dependencies
```powershell
pip install -r requirements.txt
```

### Issue 4: CSS file not found
**Solution:** Ensure `style.css` is in the same directory as `app.py`

## Testing the App

1. **Upload a CSV file** with customer data
2. **Ensure your CSV has:**
   - A 'churn' column with 0/1 or Yes/No values
   - Features like: Total Spend, Tenure, Support Calls, etc.
3. **View results** in the dashboard tabs

## App Features

- ✅ Data upload with validation
- ✅ Model training and prediction
- ✅ Risk analysis dashboard
- ✅ Feature importance visualization
- ✅ Real-time single customer prediction
- ✅ Export predictions to CSV

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Mobile browsers

