#!/usr/bin/env python3
"""
Streamlit Dashboard for Customer Churn Prediction

A comprehensive web interface for the Customer Churn Prediction API.
Provides single predictions, batch uploads, SHAP explanations, and customer segmentation.

Author: Bank Churn Analysis Team
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import time

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "https://customer-churn-api-omgg.onrender.com"
LOCAL_API_URL = "http://localhost:8000"

# Custom CSS for professional styling
st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.75rem;
    border-left: 4px solid #1f77b4;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
.high-risk {
    border-left-color: #dc3545 !important;
    background-color: #fff5f5;
}
.medium-risk {
    border-left-color: #fd7e14 !important;
    background-color: #fff8f0;
}
.low-risk {
    border-left-color: #198754 !important;
    background-color: #f0fff4;
}
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}
.status-connected {
    background-color: #198754;
    animation: pulse 2s infinite;
}
.status-disconnected {
    background-color: #dc3545;
}
.status-checking {
    background-color: #ffc107;
    animation: blink 1s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0.3; }
}
.professional-header {
    background: linear-gradient(90deg, #1f77b4 0%, #2e86ab 100%);
    color: white;
    padding: 2rem;
    border-radius: 0.75rem;
    margin-bottom: 2rem;
    text-align: center;
}
.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    margin: 1rem 0;
    transition: transform 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.error-container {
    background-color: #fff5f5;
    border: 1px solid #fed7d7;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
.warning-container {
    background-color: #fffbf0;
    border: 1px solid #feebc8;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
.info-container {
    background-color: #f0f9ff;
    border: 1px solid #bfdbfe;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class ChurnAPI:
    """API client for churn prediction service."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make single customer prediction."""
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=customer_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, customers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions."""
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json={"customers": customers},
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            response = self.session.get(f"{self.base_url}/model/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'api_client' not in st.session_state:
        st.session_state.api_client = ChurnAPI()
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []
    if 'api_status' not in st.session_state:
        st.session_state.api_status = None

def check_api_status():
    """Check and display API status with professional error handling."""
    status_placeholder = st.empty()
    
    with status_placeholder.container():
        st.markdown("""
        <div class="info-container">
            <span class="status-indicator status-checking"></span>
            <strong>Checking API connectivity...</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Check primary API
    health = st.session_state.api_client.health_check()
    st.session_state.api_status = health
    
    if health.get("status") == "healthy":
        status_placeholder.markdown("""
        <div class="info-container">
            <span class="status-indicator status-connected"></span>
            <strong>API Connected Successfully</strong><br>
            <small>Model Version: {}</small>
        </div>
        """.format(health.get('version', 'Unknown')), unsafe_allow_html=True)
        return True
    else:
        # Show professional error message
        status_placeholder.markdown("""
        <div class="warning-container">
            <span class="status-indicator status-disconnected"></span>
            <strong>Primary API Unavailable</strong><br>
            <small>Attempting to connect to local development server...</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Try local API as fallback
        time.sleep(1)  # Brief pause for better UX
        st.session_state.api_client = ChurnAPI(LOCAL_API_URL)
        local_health = st.session_state.api_client.health_check()
        
        if local_health.get("status") == "healthy":
            status_placeholder.markdown("""
            <div class="info-container">
                <span class="status-indicator status-connected"></span>
                <strong>Connected to Local Development Server</strong><br>
                <small>Model Version: {}</small>
            </div>
            """.format(local_health.get('version', 'Unknown')), unsafe_allow_html=True)
            return True
        else:
            status_placeholder.markdown("""
            <div class="error-container">
                <span class="status-indicator status-disconnected"></span>
                <strong>Service Temporarily Unavailable</strong><br>
                <small>Please try again later or contact support if the issue persists.</small><br>
                <details style="margin-top: 0.5rem;">
                    <summary style="cursor: pointer; color: #6c757d;">Technical Details</summary>
                    <code style="font-size: 0.8em; color: #6c757d;">
                        Primary API: Connection failed<br>
                        Local API: Connection failed
                    </code>
                </details>
            </div>
            """, unsafe_allow_html=True)
            return False

def render_sidebar():
    """Render sidebar with navigation and API status."""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1f77b4, #2e86ab); border-radius: 0.5rem; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">🏦 Churn Predictor</h2>
        <p style="color: #e3f2fd; margin: 0; font-size: 0.9em;">AI-Powered Customer Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status
    st.sidebar.subheader("System Status")
    
    if st.session_state.api_status:
        status = st.session_state.api_status.get("status", "unknown")
        if status == "healthy":
            st.sidebar.markdown("""
            <div style="background: #f0fff4; padding: 0.5rem; border-radius: 0.25rem; border-left: 3px solid #198754;">
                <span class="status-indicator status-connected"></span>
                <strong style="color: #198754;">API Connected</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <div style="background: #fff5f5; padding: 0.5rem; border-radius: 0.25rem; border-left: 3px solid #dc3545;">
                <span class="status-indicator status-disconnected"></span>
                <strong style="color: #dc3545;">API Disconnected</strong>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style="background: #fffbf0; padding: 0.5rem; border-radius: 0.25rem; border-left: 3px solid #ffc107;">
            <span class="status-indicator status-checking"></span>
            <strong style="color: #856404;">Checking Status...</strong>
        </div>
        """, unsafe_allow_html=True)
    
    if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
        check_api_status()
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Home",
            "👤 Single Prediction",
            "📊 Batch Predictions",
            "🔍 Model Insights",
            "📈 Analytics Dashboard"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Model Information
    if st.sidebar.button("ℹ️ Model Info"):
        model_info = st.session_state.api_client.get_model_info()
        if "error" not in model_info:
            st.sidebar.json(model_info)
        else:
            st.sidebar.error(f"Error: {model_info['error']}")
    
    return page.split(" ", 1)[1]  # Remove emoji from page name

def render_customer_form() -> Dict[str, Any]:
    """Render customer data input form."""
    st.subheader("Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            help="Customer's credit score (300-850)"
        )
        
        geography = st.selectbox(
            "Geography",
            ["France", "Spain", "Germany"],
            help="Customer's country"
        )
        
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            help="Customer's gender"
        )
        
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            help="Customer's age"
        )
        
        tenure = st.number_input(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=5,
            help="Years with the bank"
        )
    
    with col2:
        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            max_value=1000000.0,
            value=50000.0,
            step=1000.0,
            help="Current account balance"
        )
        
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2,
            help="Number of bank products"
        )
        
        has_cr_card = st.selectbox(
            "Has Credit Card",
            ["Yes", "No"],
            help="Does customer have a credit card?"
        )
        
        is_active = st.selectbox(
            "Active Member",
            ["Yes", "No"],
            help="Is customer an active member?"
        )
        
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            max_value=500000.0,
            value=75000.0,
            step=1000.0,
            help="Customer's estimated annual salary"
        )
    
    return {
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": float(estimated_salary)
    }

def render_prediction_result(prediction: Dict[str, Any]):
    """Render prediction results with visualizations."""
    if "error" in prediction:
        st.error(f"Prediction Error: {prediction['error']}")
        return
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    churn_prob = prediction.get("churn_probability", 0)
    risk_level = prediction.get("risk_level", "Unknown")
    confidence = prediction.get("confidence", 0)
    will_churn = prediction.get("churn_prediction", False)
    
    with col1:
        st.metric(
            "Churn Probability",
            f"{churn_prob:.1%}",
            delta=None
        )
    
    with col2:
        risk_color = {
            "Low": "🟢",
            "Medium": "🟡", 
            "High": "🔴"
        }.get(risk_level, "⚪")
        st.metric(
            "Risk Level",
            f"{risk_color} {risk_level}"
        )
    
    with col3:
        st.metric(
            "Confidence",
            f"{confidence:.1%}"
        )
    
    with col4:
        prediction_text = "Will Churn" if will_churn else "Will Stay"
        prediction_color = "🔴" if will_churn else "🟢"
        st.metric(
            "Prediction",
            f"{prediction_color} {prediction_text}"
        )
    
    # Probability gauge chart
    st.subheader("Churn Probability Gauge")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=churn_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk interpretation
    st.subheader("Risk Interpretation")
    
    if risk_level == "High":
        st.error(
            "🚨 **High Risk Customer**: This customer has a high probability of churning. "
            "Consider immediate retention strategies such as personalized offers, "
            "account manager contact, or loyalty programs."
        )
    elif risk_level == "Medium":
        st.warning(
            "⚠️ **Medium Risk Customer**: This customer shows some signs of potential churn. "
            "Monitor their activity closely and consider proactive engagement strategies."
        )
    else:
        st.success(
            "✅ **Low Risk Customer**: This customer is likely to stay. "
            "Continue providing excellent service to maintain their loyalty."
        )

def render_home_page():
    """Render the professional home page."""
    # Professional header
    st.markdown("""
    <div class="professional-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🏦 Customer Churn Prediction Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">AI-Powered Customer Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    Upload a CSV file with customer data to get AI-powered churn predictions. Our machine 
    learning model analyzes customer behavior patterns to identify who is likely to leave and 
    who will stay.
    """)
    
    # File upload section
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1f77b4; margin-top: 0;">📊 Upload Customer Data</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drop your CSV file here",
        type="csv",
        help="Upload a CSV file with customer data for batch predictions",
        label_visibility="collapsed"
    )
    
    if uploaded_file is None:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; border: 2px dashed #ccc; border-radius: 0.5rem; margin: 1rem 0;">
            <p style="color: #666; margin: 0;">or click to browse files</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File requirements
    with st.expander("📋 File Requirements", expanded=False):
        st.markdown("""
        • **CSV format** with headers
        • **Maximum file size:** 10MB
        • **Required columns:** CreditScore, Geography, Gender, Age, Tenure, Balance, 
          NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
        """)
        
        # Sample data
        sample_data = {
            "CreditScore": [650, 700, 580],
            "Geography": ["France", "Germany", "Spain"],
            "Gender": ["Female", "Male", "Female"],
            "Age": [35, 42, 28],
            "Tenure": [5, 8, 2],
            "Balance": [50000.0, 75000.0, 0.0],
            "NumOfProducts": [2, 1, 3],
            "HasCrCard": [1, 1, 0],
            "IsActiveMember": [1, 0, 1],
            "EstimatedSalary": [75000.0, 85000.0, 45000.0]
        }
        
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
    
    # Features overview
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1f77b4; margin-top: 0;">🚀 Platform Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **👤 Single Prediction**  
        Analyze individual customer churn risk
        
        **📊 Batch Predictions**  
        Upload CSV files for bulk analysis
        """)
    
    with col2:
        st.markdown("""
        **🔍 Model Insights**  
        Understand how the model makes decisions
        
        **📈 Analytics Dashboard**  
        View trends and patterns in your data
        """)
    
    # System status check
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1f77b4; margin-top: 0;">🔧 System Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Check System Status", use_container_width=True):
        check_api_status()
    
    # Sample data preview
    st.subheader("📋 Sample Customer Data Format")
    
    sample_data = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0
    }
    
    st.json(sample_data)

def render_single_prediction_page():
    """Render single prediction page."""
    # Professional header
    st.markdown("""
    <div class="professional-header">
        <h1 style="margin: 0; font-size: 2rem;">👤 Single Customer Prediction</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">Analyze individual customer churn risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API status first
    if st.session_state.api_status.get("status") != "healthy":
        st.markdown("""
        <div class="warning-container">
            <strong>⚠️ Service Unavailable</strong><br>
            <small>The prediction service is currently unavailable. Please check the system status in the sidebar.</small>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    Enter customer information below to get a real-time churn prediction with risk assessment.
    """)
    
    # Customer form
    customer_data = render_customer_form()
    
    # Prediction button
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing customer data..."):
            prediction = st.session_state.api_client.predict_single(customer_data)
            
            if "error" in prediction:
                st.markdown("""
                <div class="error-container">
                    <strong>❌ Prediction Failed</strong><br>
                    <small>Unable to process the prediction request. Please try again or contact support.</small>
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Store in history
            st.session_state.predictions_history.append({
                "timestamp": datetime.now(),
                "customer_data": customer_data,
                "prediction": prediction
            })
            
            # Display results
            render_prediction_result(prediction)
    
    # Prediction history
    if st.session_state.predictions_history:
        st.subheader("📊 Recent Predictions")
        
        history_df = pd.DataFrame([
            {
                "Timestamp": item["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "Churn Probability": f"{item['prediction'].get('churn_probability', 0):.1%}" if "error" not in item["prediction"] else "Error",
                "Risk Level": item["prediction"].get("risk_level", "Error") if "error" not in item["prediction"] else "Error",
                "Geography": item["customer_data"]["Geography"],
                "Age": item["customer_data"]["Age"]
            }
            for item in st.session_state.predictions_history[-10:]  # Last 10 predictions
        ])
        
        st.dataframe(history_df, use_container_width=True)

def render_batch_prediction_page():
    """Render batch prediction page."""
    st.title("📊 Batch Predictions")
    
    st.markdown("""
    Upload a CSV file with customer data to get predictions for multiple customers at once.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with customer data. See the sample format below."
    )
    
    # Sample CSV download
    st.subheader("📋 Sample CSV Format")
    
    sample_df = pd.DataFrame([
        {
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Female",
            "Age": 35,
            "Tenure": 5,
            "Balance": 50000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 75000.0
        },
        {
            "CreditScore": 720,
            "Geography": "Germany",
            "Gender": "Male",
            "Age": 42,
            "Tenure": 8,
            "Balance": 75000.0,
            "NumOfProducts": 3,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 85000.0
        }
    ])
    
    st.dataframe(sample_df)
    
    # Download sample CSV
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv_buffer.getvalue(),
        file_name="sample_customers.csv",
        mime="text/csv"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.info(f"Loaded {len(df)} customers for prediction.")
            
            # Validate columns
            required_columns = [
                "CreditScore", "Geography", "Gender", "Age", "Tenure",
                "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            # Predict button
            if st.button("🔮 Predict Batch", type="primary"):
                if len(df) > 1000:
                    st.error("Batch size cannot exceed 1000 customers.")
                    return
                
                with st.spinner(f"Making predictions for {len(df)} customers..."):
                    # Convert DataFrame to list of dictionaries
                    customers = df.to_dict('records')
                    
                    # Make batch prediction
                    result = st.session_state.api_client.predict_batch(customers)
                    
                    if "error" in result:
                        st.error(f"Batch prediction error: {result['error']}")
                        return
                    
                    # Process results
                    predictions = result.get("predictions", [])
                    summary = result.get("summary", {})
                    
                    # Display summary
                    st.subheader("📊 Batch Prediction Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Customers", summary.get("total_customers", 0))
                    
                    with col2:
                        st.metric("Predicted Churners", summary.get("predicted_churners", 0))
                    
                    with col3:
                        churn_rate = summary.get("churn_rate", 0)
                        st.metric("Churn Rate", f"{churn_rate:.1%}")
                    
                    with col4:
                        avg_prob = summary.get("avg_churn_probability", 0)
                        st.metric("Avg Probability", f"{avg_prob:.1%}")
                    
                    # Create results DataFrame
                    results_df = df.copy()
                    results_df["Churn_Probability"] = [p["churn_probability"] for p in predictions]
                    results_df["Churn_Prediction"] = [p["churn_prediction"] for p in predictions]
                    results_df["Risk_Level"] = [p["risk_level"] for p in predictions]
                    results_df["Confidence"] = [p["confidence"] for p in predictions]
                    
                    # Display results
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    st.subheader("📈 Batch Analysis Visualizations")
                    
                    # Risk distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        risk_counts = results_df["Risk_Level"].value_counts()
                        fig_pie = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Risk Level Distribution",
                            color_discrete_map={
                                "Low": "green",
                                "Medium": "orange",
                                "High": "red"
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Probability distribution
                        fig_hist = px.histogram(
                            results_df,
                            x="Churn_Probability",
                            title="Churn Probability Distribution",
                            nbins=20
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Geography analysis
                    if "Geography" in results_df.columns:
                        geo_analysis = results_df.groupby("Geography").agg({
                            "Churn_Probability": "mean",
                            "Churn_Prediction": "sum"
                        }).reset_index()
                        geo_analysis["Churn_Rate"] = geo_analysis["Churn_Prediction"] / results_df.groupby("Geography").size().values
                        
                        fig_geo = px.bar(
                            geo_analysis,
                            x="Geography",
                            y="Churn_Rate",
                            title="Churn Rate by Geography",
                            color="Churn_Rate",
                            color_continuous_scale="Reds"
                        )
                        st.plotly_chart(fig_geo, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def render_model_insights_page():
    """Render model insights page."""
    st.title("🔍 Model Insights")
    
    st.markdown("""
    Understand how the machine learning model makes predictions and what factors influence churn risk.
    """)
    
    # Model information
    if st.button("📊 Get Model Information"):
        with st.spinner("Fetching model information..."):
            model_info = st.session_state.api_client.get_model_info()
            
            if "error" not in model_info:
                st.subheader("🤖 Model Details")
                
                metadata = model_info.get("model_metadata", {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Model Name:** {metadata.get('model_name', 'Unknown')}")
                    st.info(f"**Version:** {metadata.get('version', 'Unknown')}")
                    st.info(f"**Training Date:** {metadata.get('training_date', 'Unknown')}")
                
                with col2:
                    performance = metadata.get('performance_metrics', {})
                    if performance:
                        st.info(f"**Accuracy:** {performance.get('accuracy', 'N/A')}")
                        st.info(f"**Precision:** {performance.get('precision', 'N/A')}")
                        st.info(f"**Recall:** {performance.get('recall', 'N/A')}")
                
                # Feature names
                feature_names = model_info.get("feature_names", [])
                if feature_names:
                    st.subheader("📋 Model Features")
                    st.write(f"The model uses {len(feature_names)} features:")
                    
                    # Display features in columns
                    cols = st.columns(3)
                    for i, feature in enumerate(feature_names):
                        with cols[i % 3]:
                            st.write(f"• {feature}")
                
                # Preprocessing components
                preprocessing = model_info.get("preprocessing_components", {})
                if preprocessing:
                    st.subheader("⚙️ Preprocessing Components")
                    st.json(preprocessing)
            
            else:
                st.error(f"Error fetching model info: {model_info['error']}")
    
    # Feature importance explanation
    st.subheader("📊 Feature Importance Guide")
    
    st.markdown("""
    Based on typical churn prediction models, here are the key factors that usually influence customer churn:
    
    ### 🔴 High Impact Features
    - **Age**: Older customers tend to be more stable
    - **Number of Products**: Customers with more products are less likely to churn
    - **Geography**: Different regions may have different churn patterns
    - **Activity Status**: Active members are less likely to churn
    
    ### 🟡 Medium Impact Features
    - **Balance**: Account balance can indicate customer value
    - **Credit Score**: Higher scores may indicate stability
    - **Tenure**: Longer relationships reduce churn probability
    
    ### 🟢 Lower Impact Features
    - **Gender**: Usually has minimal impact on churn
    - **Estimated Salary**: May have some correlation with stability
    - **Credit Card**: Having a credit card may increase stickiness
    """)
    
    # Risk factors explanation
    st.subheader("⚠️ Common Churn Risk Factors")
    
    risk_factors = {
        "High Risk Indicators": [
            "Low account balance with high number of products",
            "Inactive membership status",
            "Very young or very old age groups",
            "Short tenure with the bank",
            "Single product relationship"
        ],
        "Protective Factors": [
            "Multiple product relationships",
            "Active membership engagement",
            "Long-term customer relationship",
            "High account balance",
            "Good credit score"
        ]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("**High Risk Indicators**")
        for factor in risk_factors["High Risk Indicators"]:
            st.write(f"• {factor}")
    
    with col2:
        st.success("**Protective Factors**")
        for factor in risk_factors["Protective Factors"]:
            st.write(f"• {factor}")

def render_analytics_dashboard():
    """Render analytics dashboard page."""
    st.title("📈 Analytics Dashboard")
    
    st.markdown("""
    Analyze patterns and trends in your churn predictions and customer data.
    """)
    
    # Check if we have prediction history
    if not st.session_state.predictions_history:
        st.info("📊 No prediction data available yet. Make some predictions first!")
        return
    
    # Convert history to DataFrame
    history_data = []
    for item in st.session_state.predictions_history:
        if "error" not in item["prediction"]:
            row = item["customer_data"].copy()
            row.update({
                "Timestamp": item["timestamp"],
                "Churn_Probability": item["prediction"]["churn_probability"],
                "Churn_Prediction": item["prediction"]["churn_prediction"],
                "Risk_Level": item["prediction"]["risk_level"],
                "Confidence": item["prediction"]["confidence"]
            })
            history_data.append(row)
    
    if not history_data:
        st.info("📊 No valid prediction data available.")
        return
    
    df = pd.DataFrame(history_data)
    
    # Summary metrics
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(df))
    
    with col2:
        churn_count = df["Churn_Prediction"].sum()
        st.metric("Predicted Churners", churn_count)
    
    with col3:
        churn_rate = churn_count / len(df) if len(df) > 0 else 0
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    
    with col4:
        avg_prob = df["Churn_Probability"].mean()
        st.metric("Avg Probability", f"{avg_prob:.1%}")
    
    # Visualizations
    st.subheader("📈 Trend Analysis")
    
    # Time series of predictions
    df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
    daily_stats = df.groupby("Date").agg({
        "Churn_Probability": "mean",
        "Churn_Prediction": "sum"
    }).reset_index()
    
    if len(daily_stats) > 1:
        fig_time = px.line(
            daily_stats,
            x="Date",
            y="Churn_Probability",
            title="Average Churn Probability Over Time"
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Feature analysis
    st.subheader("🔍 Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age vs Churn Probability
        fig_age = px.scatter(
            df,
            x="Age",
            y="Churn_Probability",
            color="Risk_Level",
            title="Age vs Churn Probability",
            color_discrete_map={
                "Low": "green",
                "Medium": "orange",
                "High": "red"
            }
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Balance vs Churn Probability
        fig_balance = px.scatter(
            df,
            x="Balance",
            y="Churn_Probability",
            color="Risk_Level",
            title="Balance vs Churn Probability",
            color_discrete_map={
                "Low": "green",
                "Medium": "orange",
                "High": "red"
            }
        )
        st.plotly_chart(fig_balance, use_container_width=True)
    
    # Categorical analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Geography analysis
        geo_stats = df.groupby("Geography")["Churn_Probability"].mean().reset_index()
        fig_geo = px.bar(
            geo_stats,
            x="Geography",
            y="Churn_Probability",
            title="Average Churn Probability by Geography"
        )
        st.plotly_chart(fig_geo, use_container_width=True)
    
    with col2:
        # Gender analysis
        gender_stats = df.groupby("Gender")["Churn_Probability"].mean().reset_index()
        fig_gender = px.bar(
            gender_stats,
            x="Gender",
            y="Churn_Probability",
            title="Average Churn Probability by Gender"
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlations")
    
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                   "HasCrCard", "IsActiveMember", "EstimatedSalary", "Churn_Probability"]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Initial API status check (silent)
    if st.session_state.api_status is None:
        # Silent check without showing status messages
        health = st.session_state.api_client.health_check()
        if health.get("status") != "healthy":
            # Try local API silently
            st.session_state.api_client = ChurnAPI(LOCAL_API_URL)
            health = st.session_state.api_client.health_check()
        st.session_state.api_status = health
    
    # Render selected page
    if page == "Home":
        render_home_page()
    elif page == "Single Prediction":
        render_single_prediction_page()
    elif page == "Batch Predictions":
        render_batch_prediction_page()
    elif page == "Model Insights":
        render_model_insights_page()
    elif page == "Analytics Dashboard":
        render_analytics_dashboard()
    
    # Professional footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6c757d; padding: 1rem;'>"
        "<small>🏦 Customer Churn Prediction Dashboard | "
        "Powered by Machine Learning & Advanced Analytics</small>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()