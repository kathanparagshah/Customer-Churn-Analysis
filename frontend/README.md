# Customer Churn Predictor Demo

An interactive React frontend for the Customer Churn Analysis project. This application provides a beautiful, modern interface for uploading customer data and visualizing churn predictions powered by machine learning.

## Features

- 🎯 **Interactive CSV Upload**: Drag-and-drop or click to upload customer data files
- 📊 **Dual View Modes**: Switch between table and card views for prediction results
- 🔍 **Advanced Filtering**: Filter by risk level and churn prediction
- 📈 **Real-time Analytics**: Summary statistics and visual probability indicators
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- ⚡ **Fast Performance**: Built with Vite for lightning-fast development and builds
- 🎨 **Modern UI**: Beautiful interface using TailwindCSS and Framer Motion

## Tech Stack

- **React 18** - Modern React with hooks and context
- **Vite** - Next-generation frontend tooling
- **TailwindCSS** - Utility-first CSS framework
- **Framer Motion** - Production-ready motion library
- **Papa Parse** - Powerful CSV parser
- **Lucide React** - Beautiful & consistent icons

## Prerequisites

- Node.js 16+ and npm
- Backend API running on `http://localhost:8000`

## Setup

```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:3000`.

## Backend Requirements

Ensure your FastAPI backend is running with the following endpoints:

- `POST /predict/batch` - Batch prediction endpoint
- `GET /health` - Health check endpoint

The backend must be running on `http://localhost:8000` for the frontend to connect properly.

## CSV File Format

Your CSV file must include the following columns:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| CreditScore | Integer | 300-850 | Customer credit score |
| Geography | String | France/Spain/Germany | Customer location |
| Gender | String | Male/Female | Customer gender |
| Age | Integer | 18-100 | Customer age |
| Tenure | Integer | 0-10 | Years with bank |
| Balance | Float | ≥0 | Account balance |
| NumOfProducts | Integer | 1-4 | Number of products |
| HasCrCard | Integer | 0/1 | Has credit card |
| IsActiveMember | Integer | 0/1 | Is active member |
| EstimatedSalary | Float | ≥0 | Estimated salary |

### Example CSV Structure

```csv
CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
650,France,Female,35,5,50000.0,2,1,1,75000.0
720,Spain,Male,42,8,125000.0,1,1,0,85000.0
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── CSVUploader.jsx      # File upload with validation
│   │   ├── PredictionTable.jsx  # Tabular view with sorting/filtering
│   │   ├── PredictionCard.jsx   # Card view for individual customers
│   │   └── Loader.jsx           # Loading spinner component
│   ├── context/
│   │   └── PredictionsContext.jsx # Global state management
│   ├── styles/
│   │   └── tailwind.css         # Global styles and utilities
│   ├── App.jsx                  # Main application component
│   └── main.jsx                 # Application entry point
├── tailwind.config.js           # TailwindCSS configuration
├── postcss.config.js            # PostCSS configuration
├── vite.config.js               # Vite configuration
└── package.json                 # Dependencies and scripts
```

## Key Components

### CSVUploader
- Drag-and-drop file upload
- CSV parsing and validation
- Real-time error feedback
- Automatic API integration

### PredictionTable
- Sortable columns
- Advanced filtering
- Pagination
- CSV export functionality

### PredictionCard
- Visual probability indicators
- Risk level badges
- Customer information display
- Responsive card layout

## API Integration

The frontend communicates with the FastAPI backend using the following flow:

1. **File Upload**: User uploads CSV file
2. **Parsing**: Papa Parse converts CSV to JSON
3. **Validation**: Client-side validation of data format
4. **API Call**: POST request to `/predict/batch` endpoint
5. **Results**: Display predictions with summary statistics

## Customization

### Colors
Update the color palette in `tailwind.config.js`:

```javascript
colors: {
  primary: {
    DEFAULT: '#1E3A8A', // Your brand color
    // ... other shades
  }
}
```

### API Endpoint
Change the API URL in `CSVUploader.jsx`:

```javascript
const response = await fetch('YOUR_API_URL/predict/batch', {
  // ... request configuration
});
```

## Performance Considerations

- **File Size Limit**: 10MB maximum for CSV files
- **Batch Size**: API limited to 1000 customers per request
- **Virtual Scrolling**: Consider implementing for very large datasets
- **Lazy Loading**: Images and components load on demand

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure backend has proper CORS configuration
2. **API Connection**: Verify backend is running on port 8000
3. **CSV Format**: Check that all required columns are present
4. **File Size**: Ensure CSV file is under 10MB

### Debug Mode

Enable debug logging by adding to your environment:

```bash
VITE_DEBUG=true npm run dev
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Customer Churn Analysis system. See the main project README for license information.