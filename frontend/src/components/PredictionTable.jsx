import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, ArrowUpDown, ArrowUp, ArrowDown, Download, Filter } from 'lucide-react';

const PredictionTable = ({ predictions }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [filterConfig, setFilterConfig] = useState({ risk: 'all', churn: 'all' });

  // Filter predictions
  const filteredPredictions = useMemo(() => {
    return predictions.filter(prediction => {
      const riskMatch = filterConfig.risk === 'all' || prediction.risk_level === filterConfig.risk;
      const churnMatch = filterConfig.churn === 'all' || 
        (filterConfig.churn === 'churn' && prediction.churn_prediction) ||
        (filterConfig.churn === 'stay' && !prediction.churn_prediction);
      return riskMatch && churnMatch;
    });
  }, [predictions, filterConfig]);

  // Sort predictions
  const sortedPredictions = useMemo(() => {
    if (!sortConfig.key) return filteredPredictions;

    return [...filteredPredictions].sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];

      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, [filteredPredictions, sortConfig]);

  // Pagination
  const totalPages = Math.ceil(sortedPredictions.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentPredictions = sortedPredictions.slice(startIndex, endIndex);

  const handleSort = (key) => {
    setSortConfig(prevConfig => ({
      key,
      direction: prevConfig.key === key && prevConfig.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const getSortIcon = (columnKey) => {
    if (sortConfig.key !== columnKey) {
      return <ArrowUpDown size={14} className="text-gray-400" />;
    }
    return sortConfig.direction === 'asc' 
      ? <ArrowUp size={14} className="text-primary-600" />
      : <ArrowDown size={14} className="text-primary-600" />;
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

  const getRiskBadge = (risk) => {
    const colors = {
      'High': 'bg-red-100 text-red-800',
      'Medium': 'bg-yellow-100 text-yellow-800',
      'Low': 'bg-green-100 text-green-800'
    };
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[risk] || 'bg-gray-100 text-gray-800'}`}>
        {risk}
      </span>
    );
  };

  const getChurnBadge = (churnPrediction) => {
    return churnPrediction ? (
      <span className="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
        Churn
      </span>
    ) : (
      <span className="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
        Stay
      </span>
    );
  };

  const exportToCSV = () => {
    const headers = [
      'Customer ID', 'Credit Score', 'Geography', 'Gender', 'Age', 'Tenure',
      'Balance', 'Products', 'Credit Card', 'Active Member', 'Salary',
      'Churn Probability', 'Churn Prediction', 'Risk Level', 'Confidence'
    ];
    
    const csvContent = [
      headers.join(','),
      ...sortedPredictions.map(p => [
        p.id,
        p.CreditScore,
        p.Geography,
        p.Gender,
        p.Age,
        p.Tenure,
        p.Balance,
        p.NumOfProducts,
        p.HasCrCard ? 'Yes' : 'No',
        p.IsActiveMember ? 'Yes' : 'No',
        p.EstimatedSalary,
        p.churn_probability,
        p.churn_prediction ? 'Churn' : 'Stay',
        p.risk_level,
        p.confidence
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'churn_predictions.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card"
    >
      {/* Header with filters and export */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 space-y-4 sm:space-y-0">
        <h2 className="heading-xl">Prediction Results</h2>
        
        <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-2 sm:space-y-0 sm:space-x-4">
          {/* Filters */}
          <div className="flex items-center space-x-2">
            <Filter size={16} className="text-gray-500" />
            <select
              value={filterConfig.risk}
              onChange={(e) => setFilterConfig(prev => ({ ...prev, risk: e.target.value }))}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="all">All Risk Levels</option>
              <option value="High">High Risk</option>
              <option value="Medium">Medium Risk</option>
              <option value="Low">Low Risk</option>
            </select>
            
            <select
              value={filterConfig.churn}
              onChange={(e) => setFilterConfig(prev => ({ ...prev, churn: e.target.value }))}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="all">All Predictions</option>
              <option value="churn">Will Churn</option>
              <option value="stay">Will Stay</option>
            </select>
          </div>
          
          {/* Export button */}
          <button
            onClick={exportToCSV}
            className="btn-secondary flex items-center space-x-2 text-sm"
          >
            <Download size={16} />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-3 px-2 font-medium text-gray-700">ID</th>
              <th 
                className="text-left py-3 px-2 font-medium text-gray-700 cursor-pointer hover:bg-gray-50"
                onClick={() => handleSort('CreditScore')}
              >
                <div className="flex items-center space-x-1">
                  <span>Credit Score</span>
                  {getSortIcon('CreditScore')}
                </div>
              </th>
              <th className="text-left py-3 px-2 font-medium text-gray-700">Geography</th>
              <th className="text-left py-3 px-2 font-medium text-gray-700">Gender</th>
              <th 
                className="text-left py-3 px-2 font-medium text-gray-700 cursor-pointer hover:bg-gray-50"
                onClick={() => handleSort('Age')}
              >
                <div className="flex items-center space-x-1">
                  <span>Age</span>
                  {getSortIcon('Age')}
                </div>
              </th>
              <th 
                className="text-left py-3 px-2 font-medium text-gray-700 cursor-pointer hover:bg-gray-50"
                onClick={() => handleSort('Balance')}
              >
                <div className="flex items-center space-x-1">
                  <span>Balance</span>
                  {getSortIcon('Balance')}
                </div>
              </th>
              <th className="text-left py-3 px-2 font-medium text-gray-700">Products</th>
              <th 
                className="text-left py-3 px-2 font-medium text-gray-700 cursor-pointer hover:bg-gray-50"
                onClick={() => handleSort('churn_probability')}
              >
                <div className="flex items-center space-x-1">
                  <span>Churn Prob.</span>
                  {getSortIcon('churn_probability')}
                </div>
              </th>
              <th className="text-left py-3 px-2 font-medium text-gray-700">Prediction</th>
              <th className="text-left py-3 px-2 font-medium text-gray-700">Risk Level</th>
            </tr>
          </thead>
          <tbody>
            {currentPredictions.map((prediction, index) => (
              <motion.tr
                key={prediction.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.02 }}
                className="border-b border-gray-100 hover:bg-gray-50"
              >
                <td className="py-3 px-2 font-medium">{prediction.id}</td>
                <td className="py-3 px-2">{prediction.CreditScore}</td>
                <td className="py-3 px-2">{prediction.Geography}</td>
                <td className="py-3 px-2">{prediction.Gender}</td>
                <td className="py-3 px-2">{prediction.Age}</td>
                <td className="py-3 px-2">{formatCurrency(prediction.Balance)}</td>
                <td className="py-3 px-2">{prediction.NumOfProducts}</td>
                <td className="py-3 px-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium">{formatPercentage(prediction.churn_probability)}</span>
                    <div className="w-16 bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          prediction.churn_probability > 0.7 ? 'bg-red-500' :
                          prediction.churn_probability > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${prediction.churn_probability * 100}%` }}
                      />
                    </div>
                  </div>
                </td>
                <td className="py-3 px-2">{getChurnBadge(prediction.churn_prediction)}</td>
                <td className="py-3 px-2">{getRiskBadge(prediction.risk_level)}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex flex-col sm:flex-row justify-between items-center mt-6 space-y-4 sm:space-y-0">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Show</span>
            <select
              value={itemsPerPage}
              onChange={(e) => {
                setItemsPerPage(Number(e.target.value));
                setCurrentPage(1);
              }}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value={10}>10</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
            <span className="text-sm text-gray-600">per page</span>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">
              Showing {startIndex + 1} to {Math.min(endIndex, sortedPredictions.length)} of {sortedPredictions.length} results
            </span>
          </div>
          
          <div className="flex items-center space-x-1">
            <button
              onClick={() => setCurrentPage(1)}
              disabled={currentPage === 1}
              className="p-2 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronsLeft size={16} />
            </button>
            <button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className="p-2 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft size={16} />
            </button>
            
            {/* Page numbers */}
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              const pageNum = Math.max(1, Math.min(currentPage - 2 + i, totalPages - 4 + i));
              return (
                <button
                  key={pageNum}
                  onClick={() => setCurrentPage(pageNum)}
                  className={`px-3 py-1 rounded text-sm ${
                    currentPage === pageNum
                      ? 'bg-primary-600 text-white'
                      : 'hover:bg-gray-100'
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}
            
            <button
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
              className="p-2 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRight size={16} />
            </button>
            <button
              onClick={() => setCurrentPage(totalPages)}
              disabled={currentPage === totalPages}
              className="p-2 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronsRight size={16} />
            </button>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default PredictionTable;