# main.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib

# Load the model trained and saved with scikit-learn 1.6.x
model = joblib.load(r"customer_churn_model.pkl")

from preprocessing import preprocess

def main():
    st.title('Telco Customer Churn Prediction App')
    st.markdown("""
    :dart: This Streamlit app predicts customer churn for a fictional telecom. 
    Supports both online and batch predictions.
    """)
    st.markdown("", unsafe_allow_html=True)

    image = Image.open('app.jpg')
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        st.subheader("Demographic data")
        seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
        dependents = st.selectbox('Dependent:', ('Yes', 'No'))

        st.subheader("Payment data")
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        PaymentMethod = st.selectbox('PaymentMethod', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
        totalcharges = st.number_input('The total amount charged to the customer', min_value=0, max_value=10000, value=0)

        st.subheader("Services signed up for")
        mutliplelines = st.selectbox("Does the customer have multiple lines", ('Yes', 'No', 'No phone service'))
        phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
        internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Does the customer have online security", ('Yes', 'No', 'No internet service'))
        onlinebackup = st.selectbox("Does the customer have online backup", ('Yes', 'No', 'No internet service'))
        techsupport = st.selectbox("Does the customer have technology support", ('Yes', 'No', 'No internet service'))
        streamingtv = st.selectbox("Does the customer stream TV", ('Yes', 'No', 'No internet service'))
        streamingmovies = st.selectbox("Does the customer stream movies", ('Yes', 'No', 'No internet service'))

        data = {
            'SeniorCitizen': seniorcitizen,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phoneservice,
            'MultipleLines': mutliplelines,
            'InternetService': internetservice,
            'OnlineSecurity': onlinesecurity,
            'OnlineBackup': onlinebackup,
            'TechSupport': techsupport,
            'StreamingTV': streamingtv,
            'StreamingMovies': streamingmovies,
            'Contract': contract,
            'PaperlessBilling': paperlessbilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': monthlycharges,
            'TotalCharges': totalcharges
        }
        features_df = pd.DataFrame([data])
        st.write('Overview of input is shown below')
        st.dataframe(features_df)

        preprocess_df = preprocess(features_df, 'Online')
        prediction = model.predict(preprocess_df)
        if st.button('Predict'):
            if prediction[0] == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with Telco Services.')

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data.head())
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: 'Yes, the customer will terminate the service.', 0: 'No, the customer is happy with Telco Services.'})
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
    main()
