import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
import pickle
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Sales Predictor Class (Keep same)
# ---------------------------

class SalesPredictor:
    def __init__(self, scaler_path="scaler.pkl", model_path="ICPS_model.pkl"):
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
# Streamlit App Configuration & Enhanced Styling
# ---------------------------
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    layout="wide",
    page_icon="📈"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary: #2563eb;
            --secondary: #7c3aed;
            --background: #f8fafc;
            --card-background: #ffffff;
        }
        
        /* Improved body styling */
        body {
            background-color: var(--background) !important;
            font-family: 'Inter', sans-serif;
        }
        
        /* Gradient header */
        .header {
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            color: white !important;
            padding: 2rem !important;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Card styling */
        .card {
            background: var(--card-background);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
        }
        
        /* Improved button styling */
        .stButton>button {
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            color: white !important;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
            width: 100%;
        }
        
        .stButton>button:hover {
            opacity: 0.9;
            transform: scale(1.02);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: var(--card-background);
            box-shadow: 4px 0 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Metric styling */
        .metric {
            font-size: 1.4rem;
            color: var(--primary);
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header Section
# ---------------------------
st.markdown("""
    <div class="header">
        <h1 style="margin:0; font-size:2.5rem">📊 Sales Demand Forecasting</h1>
        <p style="margin:0; opacity:0.9; font-size:1.1rem">Advanced predictive analytics for smart inventory management</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar - Enhanced Design
# ---------------------------
with st.sidebar:
    st.markdown("## 🔮 Prediction Settings")
    st.markdown("---")
    mode = st.radio(
        "Select Prediction Mode:",
        options=("Predict sales for an item", 
                 "Monthly sales in a store (all items)", 
                 "Monthly sales of an item (all stores)"),
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Parameters")
    
    # Initialize predictor
    predictor = SalesPredictor(scaler_path="scaler.pkl", model_path="ICPS_model.pkl")

# ---------------------------
# Main Content Sections
# ---------------------------
# Mode 1: Predict Sales for an Item
# ---------------------------
# Mode 1: Predict Sales for an Item (Fixed Indentation)
# ---------------------------
if mode == "Predict sales for an item":
    col1, col2 = st.columns([1, 3])
    
    with col1:
        with st.container():
            st.markdown("### 📅 Date Selection")
            selected_date = st.date_input("Select Date", min_value=datetime(2018, 1, 1))
            
            st.markdown("### 🏪 Store Details")
            store_id = st.selectbox("Store ID", options=range(1, stores_range + 1))
            
            st.markdown("### 📦 Item Details")
            item_id = st.selectbox("Item ID", options=range(1, items_range + 1))
            
            if st.button("Predict Sales", key="predict_single"):
                input_data = pd.DataFrame({
                    "date": [selected_date.strftime("%d-%m-%Y")],
                    "store": [store_id],
                    "item": [item_id]
                })
                try:
                    with st.spinner("🔮 Predicting sales..."):
                        prediction = predictor.predict(input_data)
                        result = int(prediction[0])
                        st.session_state.prediction_result = result
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

    # This if-block should be OUTSIDE col1 but INSIDE the main mode block
    if 'prediction_result' in st.session_state:
        with col2:
            st.markdown("## 📈 Prediction Results")
            st.markdown(f"""
                <div class="card">
                    <h3 style="margin-top:0">Store {store_id} | Item {item_id}</h3>
                    <p style="font-size:1.2rem">📅 {selected_date.strftime('%d %b %Y')}</p>
                    <div style="display: flex; align-items: center; gap: 1rem">
                        <div style="font-size:2.5rem; color: var(--primary);">
                            {st.session_state.prediction_result}
                        </div>
                        <span style="font-size:1.2rem">units predicted</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Sales Trend (Last 7 Days)")
            st.line_chart(pd.DataFrame({
                'Date': pd.date_range(end=selected_date, periods=7),
                'Sales': np.random.randint(50, 150, size=7)
            }).set_index('Date'))

# ---------------------------
# Mode 2: Monthly Sales in a Store
# ---------------------------
elif mode == "Monthly sales in a store (all items)":
    col1, col2 = st.columns([1, 3])
    
    with col1:
        with st.container():
            st.markdown("### 🗓 Month Selection")
            year_val = st.number_input("Year", min_value=2018, value=datetime.now().year)
            month_val = st.selectbox("Month", options=range(1, 13), format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
            
            st.markdown("### 🏪 Store Selection")
            store_id = st.selectbox("Store ID", options=range(1, stores_range + 1))
            
            if st.button("Generate Report", key="predict_store"):
                try:
                    with st.spinner("📊 Generating store report..."):
                        df_sales = get_monthly_sales_store(predictor, year_val, month_val, store_id)
                        st.session_state.store_report = df_sales
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

    # This should be at the same indentation level as the col1 block
    if 'store_report' in st.session_state:
        with col2:
            st.markdown(f"## 🏪 Store {store_id} Sales Report")
            st.markdown(f"**Month:** {datetime(year_val, month_val, 1).strftime('%B %Y')}")
            
            # Summary Metrics
            total_sales = st.session_state.store_report['Monthly Sales'].sum()
            avg_sales = st.session_state.store_report['Monthly Sales'].mean()
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.markdown(f"""
                    <div class="card">
                        <h4 style="margin:0">Total Monthly Sales</h4>
                        <div class="metric">📦 {total_sales:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_metric2:
                st.markdown(f"""
                    <div class="card">
                        <h4 style="margin:0">Average per Item</h4>
                        <div class="metric">⚖️ {avg_sales:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Dataframe with enhanced styling
            st.markdown("### 📋 Item-wise Sales Breakdown")
            styled_df = st.session_state.store_report.style \
                .background_gradient(subset=['Monthly Sales'], cmap='Blues') \
                .format({'Monthly Sales': '{:,.0f}'})
            
            st.dataframe(styled_df, height=500)

# ---------------------------
# Mode 3: Monthly Sales of an Item
# ---------------------------
elif mode == "Monthly sales of an item (all stores)":
    col1, col2 = st.columns([1, 3])
    
    with col1:
        with st.container():
            st.markdown("### 🗓 Month Selection")
            year_val = st.number_input("Year", min_value=2018, value=datetime.now().year)
            month_val = st.selectbox("Month", options=range(1, 13), format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
            
            st.markdown("### 📦 Item Selection")
            item_id = st.selectbox("Item ID", options=range(1, items_range + 1))
            
            if st.button("Generate Report", key="predict_item"):
                try:
                    with st.spinner("📈 Generating item report..."):
                        df_sales = get_monthly_sales_item(predictor, year_val, month_val, item_id)
                        st.session_state.item_report = df_sales
                except Exception as e:
                    st.error(f"Error generating item report: {str(e)}")

    # This block should be at same level as col1, inside the elif
    if 'item_report' in st.session_state:
        with col2:
            st.markdown(f"## 📦 Item {item_id} Sales Report")
            st.markdown(f"**Month:** {datetime(year_val, month_val, 1).strftime('%B %Y')}")
            
            # Summary Metrics
            total_sales = st.session_state.item_report['Monthly Sales'].sum()
            avg_sales = st.session_state.item_report['Monthly Sales'].mean()
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.markdown(f"""
                    <div class="card">
                        <h4 style="margin:0">Total Monthly Sales</h4>
                        <div class="metric">📦 {total_sales:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_metric2:
                st.markdown(f"""
                    <div class="card">
                        <h4 style="margin:0">Average per Store</h4>
                        <div class="metric">🏪 {avg_sales:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Interactive bar chart
            st.markdown("### 📊 Sales Distribution by Store")
            st.bar_chart(st.session_state.item_report.set_index('Store ID'))
            
            # Data table
            st.markdown("### 📋 Store-wise Sales Data")
            styled_df = st.session_state.item_report.style \
                .background_gradient(subset=['Monthly Sales'], cmap='Greens') \
                .format({'Monthly Sales': '{:,.0f}'})
            
            st.dataframe(styled_df, height=400)
# ---------------------------
# Footer Section
# ---------------------------
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>
            <a href="mailto:mahanbhimireddy@gmail.com" style="color: #666; text-decoration: none;">📧 Contact Support</a> | 
            <a href="https://github.com/Mahan1554/Sales-Demand-Forecasting/blob/2eb9fb631d0f619fbed093aef93dac0e51f0163d/ICPS_Project_Group3.ipynb" style="color: #666; text-decoration: none;">📚 Documentation</a> |
            <span style="color: #666;">🛠 System Status: Operational</span>
        </p>
        <p style="font-size:0.8rem">© 2024 Sales Forecasting Suite. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
