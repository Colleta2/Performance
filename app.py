import streamlit as st
import pandas as pd    
import joblib
from sklearn.ensemble import RandomForestClassifier 

# Load the trained model
model = RandomForestClassifier()
model = joblib.load("random_forest_model.pkl") 

# Define the correct feature order (based on your dataset's X values)
feature_order = [
    "Age", "DistanceFromHome", "EmpEnvironmentSatisfaction",
    "EmpHourlyRate", "EmpLastSalaryHikePercent", "EmpWorkLifeBalance",
    "ExperienceYearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager", "EmpDepartment_Development"
]

# Streamlit App
st.title("Employee Performance Prediction")

# User inputs for the model
age = st.number_input("Age", min_value=18, max_value=100)
distance_from_home = st.number_input("Distance From Home", min_value=0)
emp_env_satisfaction = st.selectbox("Employee Environment Satisfaction", [1, 2, 3, 4])
emp_hourly_rate = st.number_input("Employee Hourly Rate", min_value=0,max_value=100)
emp_last_salary_hike = st.number_input("Last Salary Hike Percent", min_value=0)
emp_work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
experience_years_current_role = st.number_input("Experience Years in Current Role", min_value=0)
years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0)
years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0)
emp_department_dev = st.selectbox("Employee Department (Development=1, Other=0)", [0, 1])

# Create input data in the correct feature order
input_data = pd.DataFrame([[
    age, distance_from_home, emp_env_satisfaction,
    emp_hourly_rate, emp_last_salary_hike, emp_work_life_balance,
    experience_years_current_role, years_since_last_promotion,
    years_with_curr_manager, emp_department_dev
]], columns=feature_order)

# Prediction Button
if st.button("Predict Performance"):
    prediction = model.predict(input_data)  # Ensure input order matches X values
    st.success(f"Predicted Performance Rating: {prediction[0]}")
