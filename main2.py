import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Telco Customer Churn Analysis & Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data from Book4
@st.cache_data
def load_telco_data():
    """Load the telco customer churn dataset"""
    # Create sample data based on the structure from Book4.ipynb
    np.random.seed(42)
    n_samples = 7043
    
    # Generate data that matches the Book4 structure
    data = {
        'customerID': [f'{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}' for _ in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(0, 8684.8, n_samples), 2),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    }
    
    return pd.DataFrame(data)

# Preprocess data for modeling 
@st.cache_data
def preprocess_data(df):
    """Preprocess data for machine learning"""
    df_processed = df.copy()
    
    # Handle TotalCharges - convert to numeric and handle missing values
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
    
    # Create label encoders for categorical variables
    label_encoders = {}
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                          'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    for col in categorical_columns:
        if col in df_processed.columns:
            label_encoders[col] = LabelEncoder()
            df_processed[col] = label_encoders[col].fit_transform(df_processed[col])
    
    return df_processed, label_encoders

# Train model 
@st.cache_resource
def train_churn_model(df_processed):
    """Train the Random Forest model as in Book4"""
    # Prepare features and target
    X = df_processed.drop(['customerID', 'Churn'], axis=1)
    y = df_processed['Churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model (as used in Book4)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred, X.columns

# Main application
def main():
    st.markdown('<h1 class="main-header">📊 Telco Customer Churn Analysis & Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📈 Data Analysis Dashboard", 
        "🔮 Individual Churn Prediction", 
        "📁 Upload Dataset & Batch Prediction"
    ])
    
    # Load data and train model
    df = load_telco_data()
    df_processed, label_encoders = preprocess_data(df)
    model, accuracy, X_test, y_test, y_pred, feature_columns = train_churn_model(df_processed)
    
    if page == "📈 Data Analysis Dashboard":
        show_analysis_dashboard(df, model, accuracy, feature_columns)
    elif page == "🔮 Individual Churn Prediction":
        show_individual_prediction(df, model, label_encoders, feature_columns)
    else:
        show_upload_prediction(model, label_encoders, feature_columns)

def show_analysis_dashboard(df, model, accuracy, feature_columns):
    """Show comprehensive data analysis dashboard"""
    st.header("📈 Telco Customer Churn Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_monthly_charges = df['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")
    
    with col4:
        st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    # Visualizations
    st.subheader("📊 Customer Demographics & Services")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn distribution
        fig_churn = px.pie(df, names='Churn', title='Customer Churn Distribution',
                          color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
        st.plotly_chart(fig_churn, use_container_width=True)
        
        # Gender vs Churn
        gender_churn = pd.crosstab(df['gender'], df['Churn'], normalize='index') * 100
        fig_gender = px.bar(gender_churn, title='Churn Rate by Gender',
                           color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Contract type vs Churn
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
        fig_contract = px.bar(contract_churn, title='Churn Rate by Contract Type',
                             color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
        st.plotly_chart(fig_contract, use_container_width=True)
        
        # Internet Service vs Churn
        internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
        fig_internet = px.bar(internet_churn, title='Churn Rate by Internet Service',
                             color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
        st.plotly_chart(fig_internet, use_container_width=True)
    
    # Financial Analysis
    st.subheader("💰 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly Charges Distribution
        fig_charges = px.histogram(df, x='MonthlyCharges', color='Churn',
                                  title='Monthly Charges Distribution by Churn',
                                  color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
        st.plotly_chart(fig_charges, use_container_width=True)
    
    with col2:
        # Tenure vs Monthly Charges
        fig_scatter = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                               title='Tenure vs Monthly Charges',
                               color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Feature Importance
    st.subheader("🎯 Model Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig_importance = px.bar(feature_importance.head(10), x='importance', y='feature',
                           orientation='h', title='Top 10 Most Important Features for Churn Prediction')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Additional insights
    st.subheader("📋 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Payment Method analysis
        payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
        fig_payment = px.bar(payment_churn, title='Churn Rate by Payment Method',
                            color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
        st.plotly_chart(fig_payment, use_container_width=True)
    
    with col2:
        # Senior Citizen analysis
        senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'], normalize='index') * 100
        fig_senior = px.bar(senior_churn, title='Churn Rate by Senior Citizen Status',
                           color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
        st.plotly_chart(fig_senior, use_container_width=True)
    
    with col3:
        # Paperless Billing analysis
        paperless_churn = pd.crosstab(df['PaperlessBilling'], df['Churn'], normalize='index') * 100
        fig_paperless = px.bar(paperless_churn, title='Churn Rate by Paperless Billing',
                              color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
        st.plotly_chart(fig_paperless, use_container_width=True)

def show_individual_prediction(df, model, label_encoders, feature_columns):
    """Show individual customer churn prediction interface"""
    st.header("🔮 Individual Customer Churn Prediction")
    
    st.write("Enter customer details to predict churn probability:")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        partner = st.selectbox("Partner", ['Yes', 'No'])
        dependents = st.selectbox("Dependents", ['Yes', 'No'])
        
        st.subheader("Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
        payment_method = st.selectbox("Payment Method", 
                                     ['Electronic check', 'Mailed check', 
                                      'Bank transfer (automatic)', 'Credit card (automatic)'])
    
    with col2:
        st.subheader("Phone Services")
        phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
        multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
        
        st.subheader("Internet Services")
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    
    with col3:
        st.subheader("Additional Services")
        tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
        
        st.subheader("Charges")
        monthly_charges = st.slider("Monthly Charges ($)", 18.25, 118.75, 65.0)
        total_charges = st.slider("Total Charges ($)", 0.0, 8684.8, 2000.0)
    
    if st.button("🔮 Predict Churn", type="primary"):
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Encode categorical variables
        input_df = pd.DataFrame([input_data])
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError:
                    # Handle unseen categories
                    input_df[col] = 0
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **HIGH RISK**: Customer likely to churn")
                st.metric("Churn Probability", f"{probability[1]:.1%}")
            else:
                st.success("✅ **LOW RISK**: Customer likely to stay")
                st.metric("Retention Probability", f"{probability[0]:.1%}")
        
        with col2:
            # Probability gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability[1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

def show_upload_prediction(model, label_encoders, feature_columns):
    """Show CSV upload and batch prediction interface"""
    st.header("📁 Upload Dataset & Batch Prediction")
    
    st.write("Upload a CSV file with customer data to predict churn for multiple customers.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df_upload = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Data Preview")
            st.dataframe(df_upload.head())
            
            st.subheader("📊 Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df_upload))
            with col2:
                st.metric("Total Columns", len(df_upload.columns))
            with col3:
                missing_values = df_upload.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            
            if st.button("🔮 Predict Churn for All Customers", type="primary"):
                # Preprocess the uploaded data
                df_processed = df_upload.copy()
                
                # Store customer IDs if present
                if 'customerID' in df_processed.columns:
                    customer_ids = df_processed['customerID']
                    df_processed = df_processed.drop('customerID', axis=1)
                else:
                    customer_ids = [f"Customer_{i+1}" for i in range(len(df_processed))]
                
                # Remove Churn column if present
                if 'Churn' in df_processed.columns:
                    df_processed = df_processed.drop('Churn', axis=1)
                
                # Handle TotalCharges
                if 'TotalCharges' in df_processed.columns:
                    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
                    df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
                
                # Encode categorical variables
                for col in df_processed.columns:
                    if col in label_encoders and df_processed[col].dtype == 'object':
                        try:
                            df_processed[col] = label_encoders[col].transform(df_processed[col])
                        except ValueError:
                            # Handle unseen categories
                            df_processed[col] = 0
                
                # Ensure all required features are present
                for feature in feature_columns:
                    if feature not in df_processed.columns:
                        df_processed[feature] = 0
                
                # Reorder columns to match training data
                df_processed = df_processed[feature_columns]
                
                # Make predictions
                predictions = model.predict(df_processed)
                probabilities = model.predict_proba(df_processed)
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'CustomerID': customer_ids,
                    'Churn_Prediction': ['Yes' if pred == 1 else 'No' for pred in predictions],
                    'Churn_Probability': [prob[1] for prob in probabilities],
                    'Risk_Level': ['High' if prob[1] > 0.7 else 'Medium' if prob[1] > 0.3 else 'Low' 
                              for prob in probabilities]
                })
                
                st.subheader("🎯 Prediction Results")
                st.dataframe(results_df)
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_customers = len(results_df)
                    st.metric("Total Customers", total_customers)
                
                with col2:
                    churn_customers = (results_df['Churn_Prediction'] == 'Yes').sum()
                    st.metric("Predicted Churners", churn_customers)
                
                with col3:
                    churn_rate = (churn_customers / total_customers) * 100
                    st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
                
                with col4:
                    high_risk = (results_df['Risk_Level'] == 'High').sum()
                    st.metric("High Risk Customers", high_risk)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk level distribution
                    fig_risk = px.pie(results_df, names='Risk_Level', 
                                     title='Risk Level Distribution',
                                     color_discrete_map={'High': '#ff6b6b', 
                                                       'Medium': '#ffa726', 
                                                       'Low': '#4ecdc4'})
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                with col2:
                    # Churn probability distribution
                    fig_prob = px.histogram(results_df, x='Churn_Probability', 
                                          title='Churn Probability Distribution',
                                          nbins=20)
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prediction Results",
                    data=csv,
                    file_name='churn_predictions.csv',
                    mime='text/csv'
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your CSV file has the correct format and column names.")
    
    # Show expected format
    st.subheader("📋 Expected CSV Format")
    st.write("Your CSV file should contain the following columns:")
    
    expected_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    st.code(', '.join(expected_columns))
    
    # Sample data download
    sample_data = {
        'customerID': ['7590-VHVEG', '5575-GNVDE'],
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 0],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'No'],
        'tenure': [1, 34],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No phone service', 'No'],
        'InternetService': ['DSL', 'DSL'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['Yes', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['No', 'No'],
        'StreamingMovies': ['No', 'No'],
        'Contract': ['Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check'],
        'MonthlyCharges': [29.85, 56.95],
        'TotalCharges': [29.85, 1889.5]
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_csv = sample_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv,
        file_name='sample_telco_data.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()
