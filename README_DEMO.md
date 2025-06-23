# Customer Churn Predictor Demo

An interactive end-to-end demo for customer churn analysis featuring a React frontend with TailwindCSS that communicates with a FastAPI backend to display churn predictions.

## 🚀 Features

- **Interactive CSV Upload**: Drag-and-drop or click to upload customer data
- **Real-time Predictions**: Get churn predictions via FastAPI backend
- **Dual View Modes**: Switch between table and card views
- **Advanced Filtering**: Filter by risk level and churn status
- **Data Export**: Export filtered results to CSV
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Beautiful UI**: Modern design with smooth animations

## 🛠️ Tech Stack

### Frontend
- **React 18** with Vite for fast development
- **TailwindCSS** for styling
- **Framer Motion** for animations
- **PapaParse** for CSV parsing
- **Lucide React** for icons

### Backend
- **FastAPI** for API endpoints
- **Machine Learning** model for churn prediction
- **CORS** enabled for frontend communication

## 📋 Prerequisites

- Node.js 16+ and npm
- Python 3.8+ (for backend)
- Git

## 🚀 Quick Start

### 1. Backend Setup

```bash
# Navigate to project root
cd "Customer Churn Analysis"

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI backend
cd deployment
python app.py
```

The backend will be available at `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📊 Usage

1. **Upload CSV**: Use the drag-and-drop zone or click to select a CSV file with customer data
2. **View Predictions**: Once processed, view results in either table or card format
3. **Filter & Sort**: Use the filtering options to analyze specific customer segments
4. **Export Results**: Download filtered results as CSV for further analysis

### Expected CSV Format

Your CSV file should contain the following columns:

```csv
CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
650,France,Female,35,5,50000.0,2,1,1,75000.0
720,Spain,Male,42,8,125000.0,1,1,0,85000.0
```

**Required Columns:**
- `CreditScore`: Customer's credit score (numeric)
- `Geography`: Country (France, Spain, Germany)
- `Gender`: Customer gender (Male, Female)
- `Age`: Customer age (numeric)
- `Tenure`: Years with the bank (numeric)
- `Balance`: Account balance (numeric)
- `NumOfProducts`: Number of bank products (numeric)
- `HasCrCard`: Has credit card (0 or 1)
- `IsActiveMember`: Is active member (0 or 1)
- `EstimatedSalary`: Estimated salary (numeric)

## 🎨 Design System

### Color Palette
- **Primary**: Indigo (#1E3A8A, #6366F1)
- **Success**: Emerald (#10B981)
- **Danger**: Red (#EF4444)
- **Background**: Gray (#F3F4F6)
- **Surface**: White (#FFFFFF)

### Components
- **Cards**: Rounded corners with soft shadows
- **Buttons**: Primary and secondary variants
- **Animations**: Smooth transitions with Framer Motion

## 📁 Project Structure

```
Customer Churn Analysis/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── context/         # Global state management
│   │   ├── styles/          # TailwindCSS styles
│   │   ├── App.jsx          # Main app component
│   │   └── main.jsx         # Entry point
│   ├── package.json         # Dependencies
│   └── vite.config.js       # Vite configuration
├── deployment/              # FastAPI backend
│   └── app.py              # API endpoints
├── src/                     # ML pipeline
├── models/                  # Trained models
└── data/                    # Dataset
```

## 🔧 Available Scripts

### Frontend
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
```

### Backend
```bash
python app.py        # Start FastAPI server
```

## 🌐 API Endpoints

- `POST /predict/batch` - Batch prediction for multiple customers
- `POST /predict` - Single customer prediction
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## 🎯 Key Features Explained

### CSV Upload & Processing
- Validates file format and required columns
- Parses CSV data using PapaParse
- Sends data to backend for prediction
- Handles errors gracefully

### Prediction Display
- **Table View**: Sortable columns with pagination
- **Card View**: Individual customer cards with visual indicators
- **Risk Indicators**: Color-coded risk levels (Low, Medium, High)
- **Probability Bars**: Visual representation of churn probability

### Filtering & Export
- Filter by churn prediction (Yes/No)
- Filter by risk level
- Sort by any column
- Export filtered results to CSV

## 🔍 Troubleshooting

### Common Issues

1. **Backend not running**: Ensure FastAPI server is running on port 8000
2. **CORS errors**: Backend includes CORS middleware for localhost:3000
3. **CSV format errors**: Check that your CSV matches the expected format
4. **Port conflicts**: Frontend uses port 3000, backend uses port 8000

### Error Messages
- **"Invalid CSV format"**: Check column names and data types
- **"Backend connection failed"**: Ensure backend is running
- **"File too large"**: Maximum file size is 10MB

## 🚀 Deployment

### Frontend (Production Build)
```bash
npm run build
npm run preview
```

### Backend (Production)
```bash
# Use uvicorn for production
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🎉 Demo

Try the demo with the included `sample_data.csv` file in the frontend directory!

---

**Happy Predicting! 🎯**