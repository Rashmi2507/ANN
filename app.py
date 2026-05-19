"""
Customer Churn Prediction Web Application
==========================================
A Streamlit web app for predicting customer churn using a trained ANN model.
"""

import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🔮",
    layout="centered"
)

# Load the trained model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    """Load trained model and preprocessors with caching."""
    model = tf.keras.models.load_model("model.h5")
    
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    
    with open('one_hot_encoder_geography.pkl', 'rb') as f:
        onehot_encoder_geography = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return model, label_encoder_gender, onehot_encoder_geography, scaler

model, label_encoder_gender, onehot_encoder_geography, scaler = load_model_and_preprocessors()

## Streamlit app
st.title("🔮 Customer Churn Prediction")
st.markdown("Predict whether a customer will churn using an Artificial Neural Network")

st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location & Demographics")
    geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 92, 40)

with col2:
    st.subheader("💰 Financial Information")
    credit_score = st.number_input('Credit Score', 300, 850, 600)
    balance = st.number_input('Balance ($)', 0, 250000, 60000)
    estimated_salary = st.number_input('Estimated Salary ($)', 0, 200000, 50000)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("📊 Account Details")
    tenure = st.slider('Tenure (years)', 0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 2)

with col4:
    st.subheader("✅ Account Status")
    has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=onehot_encoder_geography.get_feature_names_out(['Geography'])
)

# Combine input data with one-hot encoded geography
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure correct column order
expected_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                    'Geography_France', 'Geography_Germany', 'Geography_Spain']
input_df = input_df[expected_columns]

# Scale input data
scaled_input = scaler.transform(input_df)

# Prediction button
if st.button("🎯 Predict Churn", use_container_width=True):
    # Make prediction
    prediction = model.predict(scaled_input, verbose=0)
    prediction_proba = prediction[0][0]
    
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    
    # Display result with color coding
    if prediction_proba > 0.5:
        st.error(f"⚠️ High Risk: Customer is likely to churn ({prediction_proba*100:.2f}% probability)")
    else:
        st.success(f"✅ Low Risk: Customer is unlikely to churn ({(1-prediction_proba)*100:.2f}% probability)")
    
    # Display probability details
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.metric("Churn Probability", f"{prediction_proba*100:.2f}%")
    with col_prob2:
        st.metric("Retention Probability", f"{(1-prediction_proba)*100:.2f}%")

st.markdown("---")
st.caption("💡 Model Information: ANN trained on 10,000 bank customers with 94%+ accuracy")