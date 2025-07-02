# Customer Churn Analysis
<!-- Trigger deployment with VERCEL_TOKEN -->

A comprehensive machine learning project for predicting customer churn in banking, featuring automated data pipelines, model explainability, and full-stack deployment.

## 🚀 Live Demo

- **Frontend**: [https://customer-churn-analysis-kgz3.vercel.app](https://customer-churn-analysis-kgz3.vercel.app)
- **API**: Deploy your own backend using the [Deployment Guide](DEPLOYMENT_GUIDE.md)
- **API Documentation**: Available at `/docs` endpoint of your deployed backend

> **Note**: The frontend currently uses mock data. Follow the [Deployment Guide](DEPLOYMENT_GUIDE.md) to connect it to real APIs.

## Project Overview

This project implements an end-to-end machine learning solution to predict customer churn for a bank. It includes data acquisition, preprocessing, exploratory data analysis, customer segmentation, predictive modeling, and full production deployment.

The solution addresses real-world challenges in customer retention by providing:
- **Predictive Analytics**: Identify customers at risk of churning
- **Customer Segmentation**: Understand different customer behaviors
- **Actionable Insights**: Data-driven recommendations for retention strategies
- **Production-Ready**: Full-stack deployment with React frontend and FastAPI backend
- **CI/CD Pipeline**: Automated testing and deployment

## Features

- **Automated Data Pipeline**: Kaggle API integration for data acquisition
- **Robust Data Processing**: Schema validation, encoding fallback, and outlier handling
- **Customer Segmentation**: Unsupervised learning with K-means clustering
- **Churn Prediction**: Supervised learning with multiple algorithms
- **Model Explainability**: SHAP integration for interpretable AI
- **Fairness Analysis**: Bias detection across demographic groups
- **API Deployment**: FastAPI-based prediction service
- **Full-Stack Integration**: React frontend with real API connectivity
- **Comprehensive Testing**: Unit tests with 95%+ coverage
- **Monitoring**: Data drift detection and model performance tracking

## 🔗 API Integration

The application supports both mock data (for demo purposes) and real API integration:

### Current Status
- ✅ **CSV Uploader**: Connected to real API
- 🔄 **Single Prediction**: Updated to use real API (requires backend deployment)
- 🔄 **Batch Predictions**: Updated to use real API (requires backend deployment)
- 🔄 **Google OAuth**: API service ready (requires backend endpoint)

### Quick Setup
1. **Deploy Backend**: Follow the [Deployment Guide](DEPLOYMENT_GUIDE.md)
2. **Configure Vercel**: Set `VITE_API_BASE_URL` environment variable
3. **Test Integration**: Use `python test_api_integration.py` to verify

### API Endpoints
- `POST /predict` - Single customer prediction
- `POST /predict/batch` - Batch customer predictions
- `GET /health` - Health check
- `GET /model/info` - Model information
- `GET /docs` - Interactive API documentation

## Dependencies

### Core Requirements
- **Python**: 3.8+ (tested on 3.8, 3.9, 3.10, 3.11)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Deep Learning**: TensorFlow/Keras (optional)
- **Visualization**: matplotlib, seaborn, plotly
- **API Framework**: FastAPI, uvicorn
- **Model Explainability**: SHAP, LIME
- **Testing**: pytest, pytest-cov

### Optional Dependencies
- **Kaggle API**: For automated data download
- **Docker**: For containerized deployment
- **Jupyter**: For interactive analysis

See `requirements.txt` for complete dependency list with versions.

## Project Structure

```
├── data/
│   ├── raw/                 # Raw data from Kaggle
│   ├── interim/             # Intermediate processed data
│   └── processed/           # Final cleaned datasets
├── notebooks/               # Jupyter notebooks for EDA
├── src/
│   ├── data/               # Data loading and cleaning scripts
│   ├── features/           # Feature engineering modules
│   ├── models/             # Model training and evaluation
│   └── visualization/      # Plotting utilities
├── models/                 # Serialized model artifacts
├── reports/                # Analysis reports and figures
├── deployment/             # API and Docker configuration
├── README.md
└── requirements.txt
```

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
# Install the latest stable version
pip install bank-churn-analysis

# Install with development dependencies
pip install bank-churn-analysis[dev]

# Install with all optional dependencies
pip install bank-churn-analysis[dev,docs]
```

After installation, you'll have access to command-line tools:
```bash
churn-predict    # Train churn prediction models
churn-segment    # Perform customer segmentation
churn-explain    # Generate model explanations
```

### Option 2: Install from TestPyPI (Pre-release)

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bank-churn-analysis
```

### Option 3: Development Installation

#### Prerequisites

- **Python 3.8+**: Ensure you have a compatible Python version
- **Git**: For cloning the repository
- **Kaggle API credentials** (optional): For automated data download
- **Docker** (optional): For containerized deployment

#### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/kathanparagshah/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
```

2. **Create and activate virtual environment**:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install the package and dependencies**:
```bash
# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e .[dev]

# Or install dependencies only
pip install -r requirements.txt
```

4. **Set up project structure**:
```bash
# Run the automated setup script
python setup_project.py

# Or create directories manually
mkdir -p data/{raw,interim,processed} models reports/figures logs
```

### Project Root Configuration

The project uses a flexible `project_root` system that automatically detects the project directory. You can also set it explicitly:

```python
from pathlib import Path
from src.models.train_churn import ChurnPredictor

# Automatic detection (recommended)
predictor = ChurnPredictor()

# Explicit path setting
predictor = ChurnPredictor(project_root=Path('/path/to/your/project'))
```

## Kaggle Credentials Setup

This project includes automated data download from Kaggle. For detailed setup instructions, see [KAGGLE_DOWNLOAD_GUIDE.md](KAGGLE_DOWNLOAD_GUIDE.md).

### Quick Setup

1. **Install Kaggle API**:
```bash
pip install kaggle
```

2. **Get API credentials**:
   - Go to Kaggle → Account → API → Create New API Token
   - Download `kaggle.json`

3. **Configure credentials** (choose one method):

   **Option A: Global setup (recommended)**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **Option B: Environment variables**
   ```bash
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   ```

   **Option C: Project-level (for development)**
   ```bash
   # Place kaggle.json in project root (already in .gitignore)
   cp ~/Downloads/kaggle.json ./kaggle.json
   ```

4. **Test download**:
```bash
python3 simple_download_test.py
```

### CI/CD Integration

For GitHub Actions, set repository secrets:
- `KAGGLE_USERNAME`: Your Kaggle username
- `KAGGLE_KEY`: Your Kaggle API key

The CI workflow will automatically handle credential setup and data download testing.

## Usage

### Running with Docker Compose

```bash
docker-compose up --build -d
```

**Endpoints:**
- Healthcheck: http://localhost:8000/health
- Swagger UI: http://localhost:8000/docs
- Predict: POST JSON to http://localhost:8000/predict

### Running the Full Pipeline End-to-End

The complete machine learning pipeline can be executed in sequence:

#### 1. Data Pipeline

```bash
# Step 1: Data Acquisition (if using Kaggle API)
python src/data/load_data.py

# Step 2: Data Cleaning and Validation
python src/data/clean_data.py

# Step 3: Feature Engineering
python src/features/create_features.py
```

#### 2. Model Training and Analysis

```bash
# Step 4: Customer Segmentation
python src/models/segment.py

# Step 5: Churn Prediction Model Training
python src/models/train_churn.py

# Step 6: Model Explanation and Interpretability
python src/models/explain.py
```

#### 3. One-Command Pipeline

For convenience, you can run the entire pipeline with:

```bash
# Run complete pipeline
python setup_project.py --full-pipeline

# Or use the Makefile
make pipeline
```

### Interactive Analysis

#### Exploratory Data Analysis

```bash
# Start Jupyter notebook
jupyter notebook notebooks/eda.ipynb

# Or use JupyterLab
jupyter lab
```

#### Python API Usage

```python
from src.models.train_churn import ChurnPredictor
from src.models.segment import CustomerSegmentation

# Initialize models
predictor = ChurnPredictor()
segmentation = CustomerSegmentation()

# Load and prepare data
data = predictor.load_and_prepare_data()

# Train churn prediction model
metrics = predictor.train_baseline_model(data)
print(f"Model F1 Score: {metrics['f1_score']:.3f}")

# Perform customer segmentation
features = segmentation.select_features(data)
scaled_features = segmentation.preprocess_features(features)
optimal_k = segmentation.find_optimal_clusters(scaled_features)
model, labels = segmentation.fit_final_model(scaled_features, optimal_k)

# Analyze segments
segment_profiles = segmentation.analyze_segments(data, features, labels)
```

### Testing

The project includes comprehensive unit tests with 95%+ coverage.

#### Running Unit Tests

```bash
# Run all tests
pytest src/tests/

# Run with verbose output
pytest src/tests/ -v

# Run with coverage report
pytest src/tests/ --cov=src --cov-report=html

# Run specific test modules
pytest src/tests/test_models.py
pytest src/tests/test_data_pipeline.py

# Run specific test classes
pytest src/tests/test_models.py::TestChurnPredictor
pytest src/tests/test_models.py::TestCustomerSegmentation

# Run tests matching a pattern
pytest src/tests/ -k "test_train"
```

#### Test Categories

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test end-to-end workflows (requires data)
- **API Tests**: Test FastAPI endpoints and responses
- **Data Pipeline Tests**: Test data loading, cleaning, and feature engineering

#### Test Configuration

Tests use temporary directories and mock data to ensure isolation:

```python
# Tests automatically handle project_root setup
@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)
```

#### Continuous Integration

Tests run automatically on:
- **Push to main branch**
- **Pull requests**
- **Multiple Python versions** (3.8, 3.9, 3.10, 3.11)
- **Different operating systems** (Ubuntu, macOS, Windows)

### Deployment

#### Local API Server

```bash
# Start development server
cd deployment
python app.py

# Or use uvicorn directly
uvicorn deployment.app:app --reload --host 0.0.0.0 --port 8000
```

#### Docker Deployment

**Model Information**: The churn prediction model is now saved with NumPy 1.26.3 compatibility and gzip compression at `deployment/models/churn_model.pkl`.

```bash
# Build Docker image
cd deployment
docker build -t bank-churn-api .

# Run container
docker run -p 8000:8000 bank-churn-api

# Or use docker-compose (recommended)
docker-compose up -d

# Check API health
curl http://localhost:8000/health
```

#### Production Deployment

```bash
# Using gunicorn for production
gunicorn deployment.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## API Usage

### Predict Churn Probability

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "credit_score": 650,
       "geography": "France",
       "gender": "Female",
       "age": 42,
       "tenure": 2,
       "balance": 83807.86,
       "num_of_products": 1,
       "has_cr_card": 1,
       "is_active_member": 1,
       "estimated_salary": 112542.58
     }'
```}'

## 🚀 Deployment Guide

### Backend Deployment (FastAPI)

#### Local Development

1. **Using Docker Compose (Recommended)**:
   ```bash
   # Build and run the entire stack
   docker-compose up --build
   
   # API will be available at http://localhost:8000
   # API docs at http://localhost:8000/docs
   # Health check at http://localhost:8000/health
   ```

   **Note**: The model is now saved with NumPy 1.26.3 compatibility and gzip compression for improved performance and reliability.

2. **Manual Setup**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the API server
   uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Production Deployment

1. **Build Docker Image**:
   ```bash
   docker build -t churn-api .
   docker tag churn-api ghcr.io/yourusername/customer-churn-analysis/churn-api:latest
   docker push ghcr.io/yourusername/customer-churn-analysis/churn-api:latest
   ```

2. **Deploy to Render.com**:
   - Connect your GitHub repository to Render
   - Create a new Web Service
   - Set build command: `docker build -t churn-api .`
   - Set start command: `uvicorn deployment.app:app --host 0.0.0.0 --port $PORT`
   - Configure environment variables as needed

3. **Deploy to Heroku**:
   ```bash
   # Install Heroku CLI and login
   heroku create your-app-name
   heroku container:push web
   heroku container:release web
   ```

#### CORS Configuration

The FastAPI backend is configured with CORS middleware to allow requests from:
- `https://customer-churn-analysis-kgz3.vercel.app` (production frontend)
- `http://localhost:3000` (local development)
- `http://127.0.0.1:3000` (alternative local)

**Why CORS is needed**: Cross-Origin Resource Sharing (CORS) is a security feature implemented by web browsers that blocks requests from one domain to another unless explicitly allowed. Since our frontend (Vercel) and backend (Render/Heroku) are on different domains, we need to configure CORS to allow the frontend to make API calls to the backend.

### Frontend Deployment (React + Vite)

#### Local Development

1. **Setup Environment**:
   ```bash
   cd frontend
   cp .env.example .env.local
   # Edit .env.local to set VITE_API_BASE_URL=http://localhost:8000
   ```

2. **Install and Run**:
   ```bash
   npm install
   npm run dev
   # Or use the convenience script:
   npm run dev:local
   ```

#### Production Deployment to Vercel

1. **Manual Deployment**:
   ```bash
   cd frontend
   npm install -g vercel
   vercel login
   vercel --prod
   ```

2. **Environment Variables in Vercel**:
   - Go to your Vercel project dashboard
   - Navigate to Settings → Environment Variables
   - Add: `VITE_API_BASE_URL` = `https://customer-churn-api-omgg.onrender.com`
   - Redeploy the project

3. **Automatic Deployment**:
   - Connect your GitHub repository to Vercel
   - Set root directory to `frontend`
   - Configure build settings:
     - Build Command: `npm run build`
     - Output Directory: `dist`
     - Install Command: `npm install`
   - Set environment variable `VITE_API_BASE_URL`

### CI/CD Pipeline

The project includes GitHub Actions workflows for automated deployment:

#### Backend Workflow (`.github/workflows/backend.yml`)
- Runs tests on every push/PR
- Builds and pushes Docker image to GitHub Container Registry
- Deploys to Render.com on main branch pushes
- **Required Secrets**:
  - `RENDER_SERVICE_ID`: Your Render service ID
  - `RENDER_API_KEY`: Your Render API key

#### Frontend Workflow (`.github/workflows/frontend.yml`)
- Runs linting and builds on every push/PR
- Creates preview deployments for PRs
- Deploys to production on main branch pushes
- **Required Secrets**:
  - `VERCEL_TOKEN`: Your Vercel API token
  - `VERCEL_ORG_ID`: Your Vercel organization ID
  - `VERCEL_PROJECT_ID`: Your Vercel project ID

### Environment Variables Reference

#### Backend
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)
- `PYTHONPATH`: Python path (set to /app in Docker)

#### Frontend
- `VITE_API_BASE_URL`: Backend API URL
  - Local: `http://localhost:8000`
  - Production: `https://customer-churn-api-omgg.onrender.com`

### Verification Steps

1. **Backend Health Check**:
   ```bash
   curl https://customer-churn-api-omgg.onrender.com/health
   ```

2. **Frontend Functionality**:
   - Visit https://customer-churn-analysis-kgz3.vercel.app
   - Upload the sample CSV file (`frontend/sample_data.csv`)
   - Verify predictions are returned without CORS errors

3. **End-to-End Test**:
   ```bash
   curl -X POST https://customer-churn-api-omgg.onrender.com/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"customers": [{"CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 35, "Tenure": 5, "Balance": 50000, "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 75000}]}'
   ```

## Key Findings

- **Customer Segments**: Identified 4 distinct customer segments with varying churn rates
- **Top Churn Drivers**: Age, number of products, and account activity level
- **Model Performance**: Achieved 85% accuracy with 0.89 ROC-AUC score
- **Fairness**: No significant bias detected across gender or geographic regions

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 85.2% |
| Precision | 82.1% |
| Recall | 79.8% |
| F1-Score | 80.9% |
| ROC-AUC | 0.89 |

## Business Impact

- **Retention Strategy**: Target high-risk segments with personalized offers
- **Cost Savings**: Reduce customer acquisition costs by 15-20%
- **Revenue Protection**: Prevent $2M+ annual revenue loss from churn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions or support, please contact [your-email@domain.com]

## Roadmap

- [ ] Monthly automated retraining pipeline
- [ ] A/B testing framework for retention strategies
- [ ] Integration with behavioral data sources
- [ ] Real-time prediction dashboard
- [ ] Advanced ensemble models