import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
import pickle
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Sales Predictor Class
# ---------------------------
class SalesPredictor:
    def __init__(self, scaler_path="C:\\Users\\mahan\\Downloads\\Play\\Projects\\scaler.pkl", 
                 model_path="ICPS_model.pkl"):
        # Load scaler and model during initialization
        self.scaler = self.load_pickle(scaler_path)
        self.model = self.load_pickle(model_path)

    def load_pickle(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def preprocess_data(self, df):
        # Split 'date' into day, month, year components
        parts = df["date"].str.split("-", n=3, expand=True)
        df["year"] = parts[2].astype('int')
        df["month"] = parts[1].astype('int')
        df["day"] = parts[0].astype('int')
        
        # Additional features: weekend indicator, holiday, cyclical transformations, and weekday
        df['weekend'] = df.apply(lambda x: self.weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
        df['holidays'] = df['date'].apply(self.is_holiday)
        df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
        df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))
        df['weekday'] = df.apply(lambda x: self.which_day(x['year'], x['month'], x['day']), axis=1)
        
        # Drop columns not used in prediction
        df.drop(['date', 'year'], axis=1, inplace=True)
        return df.reset_index(drop=True)

    def weekend_or_weekday(self, year, month, day):
        d = datetime(year, month, day)
        return 1 if d.weekday() > 4 else 0

    def is_holiday(self, date_str):
        india_holidays = holidays.country_holidays('IN')
        return 1 if india_holidays.get(date_str) else 0

    def which_day(self, year, month, day):
        d = datetime(year, month, day)
        return d.weekday()

    def predict(self, df):
        features = self.preprocess_data(df)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return np.round(prediction)

# ---------------------------
# Helper Functions for Monthly Sales
# ---------------------------
stores_range = 10
items_range = 50
month_range = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

def get_monthly_sales_store(predictor, year, month, store_id):
    """Calculate total monthly sales for each item in a given store."""
    days_in_month = month_range[month]
    if month == 2 and ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
        days_in_month = 29

    sales_data = []
    for item_id in range(1, items_range + 1):
        dates = [f"{day:02d}-{month:02d}-{year}" for day in range(1, days_in_month + 1)]
        df = pd.DataFrame({
            'date': dates,
            'store': [store_id] * days_in_month,
            'item': [item_id] * days_in_month
        })
        daily_sales = predictor.predict(df)
        total_sales = daily_sales.sum()
        sales_data.append((item_id, total_sales))
    
    df_sales = pd.DataFrame(sales_data, columns=['Item ID', 'Monthly Sales'])
    return df_sales.sort_values(by='Item ID', ascending=True)  # Sorted by Item ID

def get_monthly_sales_item(predictor, year, month, item_id):
    """Calculate total monthly sales for a given item across all stores."""
    days_in_month = month_range[month]
    if month == 2 and ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
        days_in_month = 29

    sales_data = []
    for store_id in range(1, stores_range + 1):
        dates = [f"{day:02d}-{month:02d}-{year}" for day in range(1, days_in_month + 1)]
        df = pd.DataFrame({
            'date': dates,
            'store': [store_id] * days_in_month,
            'item': [item_id] * days_in_month
        })
        daily_sales = predictor.predict(df)
        total_sales = daily_sales.sum()
        sales_data.append((store_id, total_sales))
    
    df_sales = pd.DataFrame(sales_data, columns=['Store ID', 'Monthly Sales'])
    return df_sales.sort_values(by='Store ID', ascending=True)  # Sorted by Store ID

# ---------------------------
# Streamlit App Configuration & Styling
# ---------------------------
st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

st.markdown(
    """
    <style>
        body { background-color: #ffffff; }
        .main { background-color: #ffffff; }
        .title {
            text-align: center; 
            font-size: 42px; 
            font-weight: 700; 
            color: #003366;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center; 
            font-size: 24px; 
            color: #555555;
            margin-top: 0;
        }
        .stButton>button {
            background-color: #003366; 
            color: #ffffff; 
            font-size: 20px; 
            padding: 10px 24px; 
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #002244;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
    </style>
    """, unsafe_allow_html=True
)

# Main Header
st.markdown('<p class="title">Sales Demand Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Anticipate your future sales with our advanced predictive models</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------
# Sidebar Mode Selection & Inputs
# ---------------------------
st.sidebar.header("Select Prediction Mode")
mode = st.sidebar.radio("Choose a mode:", 
    ("Predict sales for an item", 
     "Monthly sales in a store (all items)", 
     "Monthly sales of an item (all stores)"))

# Initialize predictor
predictor = SalesPredictor(
    scaler_path="C:\\Users\\mahan\\Downloads\\Play\\Projects\\scaler.pkl", 
    model_path="C:\\Users\\mahan\\Downloads\\Play\\Projects\\ICPS_model.pkl"
)

# ---------------------------
# Mode 1: Predict Sales for an Item (Store + Date)
# ---------------------------
if mode == "Predict sales for an item":
    st.sidebar.subheader("Item Sales Prediction")
    selected_date = st.sidebar.date_input("Select a Date (2018 or later)", min_value=datetime(2018, 1, 1))
    store_id = st.sidebar.selectbox("Select Store ID", options=list(range(1, stores_range + 1)))
    item_id = st.sidebar.selectbox("Select Item ID", options=list(range(1, items_range + 1)))
    
    if st.sidebar.button("Predict Item Sales"):
        input_data = pd.DataFrame({
            "date": [selected_date.strftime("%d-%m-%Y")],
            "store": [store_id],
            "item": [item_id]
        })
        try:
            prediction = predictor.predict(input_data)
            st.success(f"**Predicted Sales for Store {store_id}, Item {item_id} on {selected_date.strftime('%d-%m-%Y')}:** {int(prediction[0])}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ---------------------------
# Mode 2: Monthly Sales in a Store (All Items)
# ---------------------------
elif mode == "Monthly sales in a store (all items)":
    st.sidebar.subheader("Monthly Sales for All Items in a Store")
    year_val = st.sidebar.number_input("Enter Year", min_value=2018, max_value=2050, value=datetime.now().year, step=1)
    month_val = st.sidebar.selectbox("Select Month", options=list(range(1, 13)))
    store_id = st.sidebar.selectbox("Select Store ID", options=list(range(1, stores_range + 1)))
    
    if st.sidebar.button("Predict Monthly Sales (Store)"):
        try:
            df_sales = get_monthly_sales_store(predictor, int(year_val), int(month_val), store_id)
            st.success(f"**Monthly Sales for all items in Store {store_id} for {month_val}/{year_val}:**")
            # Display sorted by Item ID (ascending)
            st.dataframe(df_sales.sort_values(by='Item ID', ascending=True).reset_index(drop=True))
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------
# Mode 3: Monthly Sales of an Item (All Stores)
# ---------------------------
elif mode == "Monthly sales of an item (all stores)":
    st.sidebar.subheader("Monthly Sales for an Item Across All Stores")
    year_val = st.sidebar.number_input("Enter Year", min_value=2018, max_value=2050, value=datetime.now().year, step=1)
    month_val = st.sidebar.selectbox("Select Month", options=list(range(1, 13)))
    item_id = st.sidebar.selectbox("Select Item ID", options=list(range(1, items_range + 1)))
    
    if st.sidebar.button("Predict Monthly Sales (Item)"):
        try:
            df_sales = get_monthly_sales_item(predictor, int(year_val), int(month_val), item_id)
            st.success(f"**Monthly Sales for Item {item_id} across all stores for {month_val}/{year_val}:**")
            # Display sorted by Store ID (ascending)
            st.dataframe(df_sales.sort_values(by='Store ID', ascending=True).reset_index(drop=True))
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------
# Instructions Section
# ---------------------------
st.markdown(
    """
    **Instructions:**
    - Use the sidebar to select a prediction mode.
    - For **Predict sales for an item**, select the date, store, and item.
    - For **Monthly sales in a store (all items)**, provide the year, month, and store to see sales for each item.
    - For **Monthly sales of an item (all stores)**, provide the year, month, and item to see sales across stores.
    - Results are sorted by ID (Store/Item) in ascending order.
    """, unsafe_allow_html=True
)

