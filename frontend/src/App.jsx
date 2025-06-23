import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart3, Table, Grid3X3, AlertCircle, TrendingUp, Users, Target } from 'lucide-react';
import { PredictionsProvider, usePredictions } from './context/PredictionsContext';
import CSVUploader from './components/CSVUploader';
import PredictionTable from './components/PredictionTable';
import PredictionCard from './components/PredictionCard';
import Loader from './components/Loader';
import './styles/tailwind.css';

const Header = () => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary-600 rounded-lg">
              <BarChart3 className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Churn Predictor Demo</h1>
              <p className="text-sm text-gray-500">AI-Powered Customer Analytics</p>
            </div>
          </div>
          
          <div className="hidden sm:flex items-center space-x-6 text-sm text-gray-600">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>API Connected</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

const SummaryStats = ({ summary }) => {
  if (!summary) return null;

  const stats = [
    {
      label: 'Total Customers',
      value: summary.total_customers,
      icon: Users,
      color: 'text-blue-600 bg-blue-100'
    },
    {
      label: 'Predicted Churners',
      value: summary.predicted_churners,
      icon: AlertCircle,
      color: 'text-red-600 bg-red-100'
    },
    {
      label: 'Churn Rate',
      value: `${(summary.churn_rate * 100).toFixed(1)}%`,
      icon: TrendingUp,
      color: 'text-yellow-600 bg-yellow-100'
    },
    {
      label: 'High Risk',
      value: summary.high_risk_customers,
      icon: Target,
      color: 'text-red-600 bg-red-100'
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6"
    >
      {stats.map((stat, index) => {
        const Icon = stat.icon;
        return (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            className="card p-4"
          >
            <div className="flex items-center space-x-3">
              <div className={`p-2 rounded-lg ${stat.color}`}>
                <Icon size={20} />
              </div>
              <div>
                <p className="text-2xl font-semibold text-gray-900">{stat.value}</p>
                <p className="text-sm text-gray-600">{stat.label}</p>
              </div>
            </div>
          </motion.div>
        );
      })}
    </motion.div>
  );
};

const ViewToggle = ({ viewMode, onViewModeChange }) => {
  return (
    <div className="flex items-center space-x-2 mb-6">
      <span className="text-sm font-medium text-gray-700">View:</span>
      <div className="flex bg-gray-100 rounded-lg p-1">
        <button
          onClick={() => onViewModeChange('table')}
          className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
            viewMode === 'table'
              ? 'bg-white text-primary-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <Table size={16} />
          <span>Table</span>
        </button>
        <button
          onClick={() => onViewModeChange('cards')}
          className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
            viewMode === 'cards'
              ? 'bg-white text-primary-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <Grid3X3 size={16} />
          <span>Cards</span>
        </button>
      </div>
    </div>
  );
};

const ErrorMessage = ({ error, onDismiss }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3"
    >
      <AlertCircle className="text-red-500 mt-0.5" size={20} />
      <div className="flex-1">
        <h4 className="font-medium text-red-800">Error</h4>
        <p className="text-red-700 text-sm">{error}</p>
      </div>
      <button
        onClick={onDismiss}
        className="text-red-400 hover:text-red-600"
      >
        ×
      </button>
    </motion.div>
  );
};

const MainContent = () => {
  const {
    predictions,
    loading,
    error,
    summary,
    viewMode,
    setViewMode,
    clearError,
    resetState
  } = usePredictions();

  const hasResults = predictions.length > 0;

  const handleNewUpload = () => {
    resetState();
  };

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <AnimatePresence>
        {error && (
          <ErrorMessage error={error} onDismiss={clearError} />
        )}
      </AnimatePresence>

      {!hasResults && !loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center mb-8"
        >
          <h2 className="heading-xl mb-4">Welcome to Churn Predictor</h2>
          <p className="body-text max-w-2xl mx-auto mb-8">
            Upload a CSV file with customer data to get AI-powered churn predictions. 
            Our machine learning model analyzes customer behavior patterns to identify 
            who is likely to leave and who will stay.
          </p>
        </motion.div>
      )}

      {!hasResults && !loading && <CSVUploader />}

      {hasResults && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {/* Summary Statistics */}
          <SummaryStats summary={summary} />

          {/* Action Bar */}
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 space-y-4 sm:space-y-0">
            <ViewToggle viewMode={viewMode} onViewModeChange={setViewMode} />
            <button
              onClick={handleNewUpload}
              className="btn-secondary"
            >
              Upload New File
            </button>
          </div>

          {/* Results */}
          {viewMode === 'table' ? (
            <PredictionTable predictions={predictions} />
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {predictions.map((prediction, index) => (
                <PredictionCard
                  key={prediction.id}
                  prediction={prediction}
                  index={index}
                />
              ))}
            </div>
          )}
        </motion.div>
      )}

      {loading && <Loader />}
    </main>
  );
};

const App = () => {
  return (
    <PredictionsProvider>
      <div className="min-h-screen bg-surface-gray">
        <Header />
        <MainContent />
        
        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center text-gray-600">
              <p className="text-sm">
                Built with React, TailwindCSS, and FastAPI • 
                <a href="#" className="text-primary-600 hover:text-primary-700">Documentation</a> • 
                <a href="#" className="text-primary-600 hover:text-primary-700">API Reference</a>
              </p>
            </div>
          </div>
        </footer>
      </div>
    </PredictionsProvider>
  );
};

export default App;