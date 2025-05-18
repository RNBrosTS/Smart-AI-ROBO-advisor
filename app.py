import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load saved assets
model = joblib.load("Recommending Personal Portfolio/extra_trees_model.pkl")
scaler = joblib.load("Recommending Personal Portfolio/scaler.pkl")
label_encoder = joblib.load("Recommending Personal Portfolio/label_encoder.pkl")
column_order = joblib.load("Recommending Personal Portfolio/column_order.pkl")

# Set up the page
st.set_page_config(page_title="Smart Investment Advisor", layout="wide")

# In-memory user store (username: password)
if 'users' not in st.session_state:
    st.session_state['users'] = {"admin": "admin123"}
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
if st.session_state['logged_in']:
    if st.session_state['username'] == "admin":
        page = st.sidebar.radio("Go to", ["🏠 Home", "📁 View Records", "ℹ️ About", "🚪 Logout"])
    else:
        page = st.sidebar.radio("Go to", ["🏠 Home", "🤖 Risk Predictor", "ℹ️ About", "🚪 Logout"])
else:
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔑 Login", "🆕 Sign Up"])

# Home Page (Always Visible)
if page == "🏠 Home":
    st.title("💼 Smart Investment Advisor")
    st.markdown("""
    Welcome to the **Smart Investment Advisor** 🧠📈  
    This platform helps you:
    - Predict your **financial risk profile**
    - Get **personalized stock recommendations** based on your risk
    - Learn how to **balance your portfolio** better

    👉 Please login to access prediction features.
    """)

# Login
if page == "🔑 Login":
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state['users'] and st.session_state['users'][username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# Signup
if page == "🆕 Sign Up":
    st.title("🆕 Create Account")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    if st.button("Sign Up"):
        if new_username in st.session_state['users']:
            st.error("Username already exists")
        else:
            st.session_state['users'][new_username] = new_password
            st.success("Account created! Please login.")

# Logout
if page == "🚪 Logout":
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""
    st.success("You have been logged out.")
    st.rerun()

# About Page
if page == "ℹ️ About" and st.session_state['logged_in']:
    st.title("ℹ️ About This App")
    st.markdown("""
    This web application uses **Machine Learning** to predict an investor's financial risk rating
    based on personal and financial inputs.

    Once your risk is assessed as **Low**, **Medium**, or **High**, the app:
    - Recommends stocks from a curated list
    - Suggests a counter-strategy to balance risk

    **Tools used:** Streamlit, scikit-learn, SMOTE, Extra Trees Classifier
    """)

# Admin View
if page == "📁 View Records" and st.session_state['username'] == "admin":
    st.title("📁 All Investor Predictions")
    if os.path.exists("investor_data.csv"):
        df = pd.read_csv("investor_data.csv")
        st.dataframe(df)
    else:
        st.warning("No prediction data available.")

# Risk Predictor
if page == "🤖 Risk Predictor" and st.session_state['logged_in'] and st.session_state['username'] != "admin":
    st.title("🤖 Financial Risk Rating Predictor")

    def user_input_form():
        with st.form("form"):
            age = st.number_input("Age", min_value=18, max_value=100, value=49)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            education = st.selectbox("Education Level", ["PhD", "Master's", "Bachelor's", "Other"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            income = st.number_input("Income", value=72799)
            credit_score = st.number_input("Credit Score", 300, 850, value=688)
            loan_amount = st.number_input("Loan Amount", value=45713)
            loan_purpose = st.selectbox("Loan Purpose", ["Business", "Personal", "Auto", "Home"])
            employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed"])
            years_current_job = st.number_input("Years at Current Job", value=6)
            payment_history = st.selectbox("Payment History", ["Poor", "Fair", "Excellent"])
            dti = st.number_input("Debt-to-Income Ratio", value=0.15)
            assets_value = st.number_input("Assets Value", value=120228)
            dependents = st.number_input("Number of Dependents", value=0)
            previous_defaults = st.number_input("Previous Defaults", value=2)
            marital_status_change = st.number_input("Marital Status Change", value=2)

            submitted = st.form_submit_button("Predict Risk Rating")

        input_data = {
            "Username": st.session_state['username'],
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Marital Status": marital,
            "Income": income,
            "Credit Score": credit_score,
            "Loan Amount": loan_amount,
            "Loan Purpose": loan_purpose,
            "Employment Status": employment_status,
            "Years at Current Job": years_current_job,
            "Payment History": payment_history,
            "Debt-to-Income Ratio": dti,
            "Assets Value": assets_value,
            "Number of Dependents": dependents,
            "Previous Defaults": previous_defaults,
            "Marital Status Change": marital_status_change
        }

        return pd.DataFrame([input_data]), submitted

    user_df, submitted = user_input_form()

    if submitted:
        features_only = user_df.drop(columns=["Username"])
        for col in column_order:
            if col not in features_only.columns:
                features_only[col] = 0
        features_only = features_only[column_order]
        for col in features_only.select_dtypes(include='object').columns:
            features_only[col] = LabelEncoder().fit_transform(features_only[col])

        user_scaled = scaler.transform(features_only)
        prediction = model.predict(user_scaled)
        numeric_label = int(prediction[0])
 
        emoji_mapping = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
        display_result = emoji_mapping.get(numeric_label)
        st.success(f"🎯 Predicted Risk Rating: **{display_result}**")

        low_stocks = pd.DataFrame({"Stock_Symbol": ["12.Mutual_Funds"], "Trend_Pct": [-9.08], "Volatility": [0.47]})
        medium_stocks = pd.DataFrame({
            "Stock_Symbol": ["00DSEX", "3RDICB", "00DSES", "00DS30", "06.Food_&_Allied", "14.Pharmaceuticals_&_Chemicals",
                             "07.Fuel_&_Power", "16.Tannery_Industries", "20.Bond", "08.Insurance", "02.Cement", "05.Financial_Institutions"],
            "Trend_Pct": [17.77, 17.45, 14.39, 13.25, 8.90, 3.38, 0.57, -1.85, -2.14, -2.68, -7.86, -13.37],
            "Volatility": [0.69, 0.85, 0.63, 0.69, 0.80, 0.86, 0.94, 0.67, 0.63, 0.84, 0.97, 0.86]
        })
        high_stocks = pd.DataFrame({
            "Stock_Symbol": ["ACIFORMULA", "03.Ceramics_Sector", "ALARABANK", "ACI", "AIBL1STIMF", "1STICB",
                             "13.Paper_&_Printing", "04.Engineering", "7THICB", "5THICB", "10.Jute",
                             "ACIZCBOND", "8THICB", "18.Textile", "4THICB"],
            "Trend_Pct": [62.60, 45.63, 31.11, 28.18, 20.85, 20.63, 20.17, 17.32, 14.87, 8.46, 7.93, 5.51, 4.82, 3.95, 3.50],
            "Volatility": [2.19, 2.05, 1.62, 1.76, 2.34, 1.08, 2.63, 1.28, 2.14, 1.47, 2.73, 1.17, 1.52, 2.49, 1.45]
        })

        if numeric_label == 2:
            st.info("🛡️ You're a high-risk investor. Consider safer, stable investments to reduce volatility.")
            recommended = low_stocks
        elif numeric_label == 1:
            st.info("⚖️ You're a medium-risk investor. These balanced stocks align with your risk level.")
            recommended = medium_stocks
        else:
            st.info("🚀 You're a low-risk investor. You may explore high-growth, high-volatility options.")
            recommended = high_stocks

        st.dataframe(recommended)

        # Save to CSV
        user_df["Predicted Risk"] = display_result
        user_df["Recommended Stocks"] = ", ".join(recommended["Stock_Symbol"].values[:5])

        if os.path.exists("investor_data.csv"):
            existing = pd.read_csv("investor_data.csv")
            new_df = pd.concat([existing, user_df], ignore_index=True)
        else:
            new_df = user_df

        new_df.to_csv("investor_data.csv", index=False)
