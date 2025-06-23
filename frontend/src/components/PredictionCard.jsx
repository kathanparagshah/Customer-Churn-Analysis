import React from 'react';
import { motion } from 'framer-motion';
import { User, TrendingUp, TrendingDown, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

const PredictionCard = ({ prediction, index }) => {
  const {
    id,
    CreditScore,
    Geography,
    Gender,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    HasCrCard,
    IsActiveMember,
    EstimatedSalary,
    churn_probability,
    churn_prediction,
    risk_level,
    confidence
  } = prediction;

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'High': return 'text-red-600 bg-red-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'Low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getRiskIcon = (risk) => {
    switch (risk) {
      case 'High': return <AlertTriangle size={16} />;
      case 'Medium': return <TrendingUp size={16} />;
      case 'Low': return <CheckCircle size={16} />;
      default: return <TrendingDown size={16} />;
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ scale: 1.02, y: -2 }}
      className="card hover:shadow-lg transition-all duration-200"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-primary-100 rounded-lg">
            <User className="text-primary-600" size={20} />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">Customer #{id}</h3>
            <p className="text-sm text-gray-500">{Geography} • {Gender} • {Age}y</p>
          </div>
        </div>
        
        {/* Risk Badge */}
        <div className={`px-3 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${getRiskColor(risk_level)}`}>
          {getRiskIcon(risk_level)}
          <span>{risk_level} Risk</span>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Credit Score</p>
          <p className="text-lg font-semibold text-gray-900">{CreditScore}</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Balance</p>
          <p className="text-lg font-semibold text-gray-900">{formatCurrency(Balance)}</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Tenure</p>
          <p className="text-lg font-semibold text-gray-900">{Tenure} years</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Products</p>
          <p className="text-lg font-semibold text-gray-900">{NumOfProducts}</p>
        </div>
      </div>

      {/* Additional Info */}
      <div className="grid grid-cols-3 gap-2 mb-4 text-xs">
        <div className="text-center">
          <p className="text-gray-500">Salary</p>
          <p className="font-medium">{formatCurrency(EstimatedSalary)}</p>
        </div>
        <div className="text-center">
          <p className="text-gray-500">Credit Card</p>
          <p className="font-medium">{HasCrCard ? 'Yes' : 'No'}</p>
        </div>
        <div className="text-center">
          <p className="text-gray-500">Active</p>
          <p className="font-medium">{IsActiveMember ? 'Yes' : 'No'}</p>
        </div>
      </div>

      {/* Churn Probability Bar */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Churn Probability</span>
          <span className="text-sm font-semibold text-gray-900">{formatPercentage(churn_probability)}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${churn_probability * 100}%` }}
            transition={{ duration: 1, delay: index * 0.05 + 0.5 }}
            className={`h-3 rounded-full ${
              churn_probability > 0.7 ? 'bg-red-500' :
              churn_probability > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
            }`}
          />
        </div>
      </div>

      {/* Confidence Score */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Confidence</span>
          <span className="text-sm font-semibold text-gray-900">{formatPercentage(confidence)}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ duration: 1, delay: index * 0.05 + 0.7 }}
            className="h-2 rounded-full bg-blue-500"
          />
        </div>
      </div>

      {/* Footer - Prediction Result */}
      <div className={`flex items-center justify-center space-x-2 p-3 rounded-lg ${
        churn_prediction 
          ? 'bg-red-50 text-red-700 border border-red-200' 
          : 'bg-green-50 text-green-700 border border-green-200'
      }`}>
        {churn_prediction ? (
          <>
            <XCircle size={20} />
            <span className="font-semibold">Likely to Churn</span>
          </>
        ) : (
          <>
            <CheckCircle size={20} />
            <span className="font-semibold">Likely to Stay</span>
          </>
        )}
      </div>
    </motion.div>
  );
};

export default PredictionCard;