# Sales Demand Forecasting System

## 📌 Project Overview
A retail sales prediction system that forecasts product demand using machine learning. The system helps optimize inventory management by predicting:
- Daily sales for specific products
- Monthly demand across all products in a store
- Item popularity across multiple locations


## 🤖 Model Architecture
### Core Algorithm: XGBoost Regression
- **Why XGBoost?**
  - Handles non-linear relationships well
  - Built-in feature importance
  - Robust to outliers
  - Excellent performance on tabular data

### Key Features Engineered:
1. **Temporal Features**:
   - Day/month/year extraction
   - Weekend/weekday flags
   - Holiday indicators (India-specific)
   
2. **Cyclical Encoding**:
   - Sine/cosine transforms for monthly patterns

3. **Store-Item Interactions**:
   - Cross-features between location and product

## 💻 Streamlit Web Interface
### Three Prediction Modes:
1. **Item-Level Forecast**
   - Daily sales prediction for a specific product in a store
   - Inputs: Date, Store ID, Item ID

2. **Store Analysis**
   - Monthly demand for all products in a store
   - Inputs: Year, Month, Store ID

3. **Product Analysis**  
   - Monthly demand for a product across all stores
   - Inputs: Year, Month, Item ID


## 🛠️ Setup 
Access the interface at 
```
will be added soon
```

   

## 📈 Sample Output
Prediction Interface:
<img width="959" alt="image" src="https://github.com/user-attachments/assets/09e59245-3948-4c54-a6b2-ac347eb8edb3" />




