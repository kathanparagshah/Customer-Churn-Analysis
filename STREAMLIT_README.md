# 🏦 Customer Churn Prediction Dashboard

A comprehensive Streamlit web application for predicting customer churn using machine learning. This dashboard provides an intuitive interface to interact with the FastAPI backend for real-time churn predictions, batch processing, and analytics.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Access to the FastAPI backend (either deployed or running locally)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r streamlit_requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run streamlit_dashboard.py
   ```

3. **Open your browser:**
   The dashboard will automatically open at `http://localhost:8501`

## 📊 Features

### 🏠 Home Page
- Overview of dashboard capabilities
- API status monitoring
- Sample data format reference

### 👤 Single Prediction
- Interactive form for customer data input
- Real-time churn probability calculation
- Risk level assessment (Low/Medium/High)
- Visual probability gauge
- Prediction history tracking

### 📊 Batch Predictions
- CSV file upload for bulk processing
- Sample CSV template download
- Batch prediction summary statistics
- Results visualization and export
- Geographic and demographic analysis

### 🔍 Model Insights
- Model metadata and performance metrics
- Feature importance explanations
- Risk factor analysis
- Preprocessing component details

### 📈 Analytics Dashboard
- Prediction trends over time
- Feature correlation analysis
- Geographic and demographic breakdowns
- Interactive visualizations

## 🔧 Configuration

### API Endpoints
The dashboard automatically tries to connect to:
1. **Production API:** `https://customer-churn-api-omgg.onrender.com`
2. **Local API (fallback):** `http://localhost:8000`

To use a different API endpoint, modify the `API_BASE_URL` constant in `streamlit_dashboard.py`:

```python
API_BASE_URL = "https://your-api-endpoint.com"
```

### Environment Variables
You can also set the API URL using environment variables:

```bash
export CHURN_API_URL="https://your-api-endpoint.com"
streamlit run streamlit_dashboard.py
```

## 📋 Data Format

### Required Fields
The dashboard expects customer data with the following fields:

| Field | Type | Description | Range/Values |
|-------|------|-------------|-------------|
| CreditScore | Integer | Customer's credit score | 300-850 |
| Geography | String | Customer's country | France, Spain, Germany |
| Gender | String | Customer's gender | Male, Female |
| Age | Integer | Customer's age | 18-100 |
| Tenure | Integer | Years with the bank | 0-10 |
| Balance | Float | Account balance | 0.0+ |
| NumOfProducts | Integer | Number of bank products | 1-4 |
| HasCrCard | Integer | Has credit card | 0 (No), 1 (Yes) |
| IsActiveMember | Integer | Active member status | 0 (No), 1 (Yes) |
| EstimatedSalary | Float | Estimated annual salary | 0.0+ |

### Sample CSV Format
```csv
CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
650,France,Female,35,5,50000.0,2,1,1,75000.0
720,Germany,Male,42,8,75000.0,3,1,0,85000.0
```

## 🎨 User Interface

### Navigation
- **Sidebar:** Contains navigation menu, API status, and model information
- **Main Area:** Displays the selected page content
- **Status Indicators:** Real-time API connectivity status

### Visualizations
- **Probability Gauge:** Interactive gauge showing churn probability
- **Risk Distribution:** Pie charts showing risk level breakdown
- **Trend Analysis:** Time series plots of prediction patterns
- **Correlation Heatmaps:** Feature relationship analysis
- **Geographic Analysis:** Regional churn pattern visualization

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check if the FastAPI backend is running
   - Verify the API URL in the configuration
   - Check network connectivity

2. **CSV Upload Errors**
   - Ensure all required columns are present
   - Check data types match the expected format
   - Verify file size is under the limit (1000 rows)

3. **Prediction Errors**
   - Validate input data ranges
   - Check for missing or invalid values
   - Ensure API backend model is loaded

### Debug Mode
Run with debug information:
```bash
streamlit run streamlit_dashboard.py --logger.level=debug
```

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with the `streamlit_requirements.txt` file

### Docker Deployment
Create a `Dockerfile` for the dashboard:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY streamlit_requirements.txt .
RUN pip install -r streamlit_requirements.txt

COPY streamlit_dashboard.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Local Network Access
To access the dashboard from other devices on your network:

```bash
streamlit run streamlit_dashboard.py --server.address=0.0.0.0
```

## 📈 Performance Tips

1. **Batch Processing:** Use batch predictions for multiple customers instead of individual calls
2. **Data Caching:** The dashboard caches API responses for better performance
3. **File Size:** Keep CSV uploads under 1000 rows for optimal performance
4. **Browser:** Use modern browsers for best visualization experience

## 🤝 Contributing

To contribute to the dashboard:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Customer Churn Analysis system. See the main project LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section above
- Review the main project documentation
- Open an issue in the GitHub repository

---

**Built with ❤️ using Streamlit, Plotly, and FastAPI**