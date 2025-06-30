import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, AlertCircle, CheckCircle } from 'lucide-react';
import Papa from 'papaparse';
import { usePredictions } from '../hooks/usePredictions';
import apiService from '../services/apiService';

const CSVUploader = () => {
  const [dragActive, setDragActive] = useState(false);
  const [parseError, setParseError] = useState(null);
  const [parseSuccess, setParseSuccess] = useState(false);
  const fileInputRef = useRef(null);
  
  const { setLoading, setPredictions, setError, setUploadedFile } = usePredictions();

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Reset states
    setParseError(null);
    setParseSuccess(false);
    setError(null);
    
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setParseError('Please select a CSV file.');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setParseError('File size must be less than 10MB.');
      return;
    }

    setUploadedFile(file);
    setParseSuccess(true);

    // Parse CSV
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (header) => header.trim(),
      complete: (results) => {
        if (results.errors.length > 0) {
          setParseError(`CSV parsing error: ${results.errors[0].message}`);
          setParseSuccess(false);
          return;
        }

        if (results.data.length === 0) {
          setParseError('CSV file is empty or contains no valid data.');
          setParseSuccess(false);
          return;
        }

        // Validate required columns
        const requiredColumns = [
          'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
          'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ];
        
        const headers = Object.keys(results.data[0]);
        const missingColumns = requiredColumns.filter(col => !headers.includes(col));
        
        if (missingColumns.length > 0) {
          setParseError(`Missing required columns: ${missingColumns.join(', ')}`);
          setParseSuccess(false);
          return;
        }

        // Validate data types and ranges
        const validationErrors = validateData(results.data);
        if (validationErrors.length > 0) {
          setParseError(`Data validation errors: ${validationErrors.slice(0, 3).join(', ')}${validationErrors.length > 3 ? '...' : ''}`);
          setParseSuccess(false);
          return;
        }

        // Send to API
        sendPredictionRequest(results.data);
      },
      error: (error) => {
        setParseError(`Failed to parse CSV: ${error.message}`);
        setParseSuccess(false);
      }
    });
  };

  const validateData = (data) => {
    const errors = [];
    
    data.slice(0, 10).forEach((row, index) => { // Validate first 10 rows for performance
      // Credit Score validation
      const creditScore = parseInt(row.CreditScore);
      if (isNaN(creditScore) || creditScore < 300 || creditScore > 850) {
        errors.push(`Row ${index + 1}: Invalid CreditScore`);
      }
      
      // Geography validation
      if (!['France', 'Spain', 'Germany'].includes(row.Geography)) {
        errors.push(`Row ${index + 1}: Invalid Geography`);
      }
      
      // Gender validation
      if (!['Male', 'Female'].includes(row.Gender)) {
        errors.push(`Row ${index + 1}: Invalid Gender`);
      }
      
      // Age validation
      const age = parseInt(row.Age);
      if (isNaN(age) || age < 18 || age > 100) {
        errors.push(`Row ${index + 1}: Invalid Age`);
      }
      
      // Tenure validation
      const tenure = parseInt(row.Tenure);
      if (isNaN(tenure) || tenure < 0 || tenure > 10) {
        errors.push(`Row ${index + 1}: Invalid Tenure`);
      }
    });
    
    return errors;
  };

  const sendPredictionRequest = async (data) => {
    setLoading(true);
    
    try {
      // Transform data to match API format
      const customers = data.map(row => ({
        credit_score: parseInt(row.CreditScore),
        geography: row.Geography,
        gender: row.Gender,
        age: parseInt(row.Age),
        tenure: parseInt(row.Tenure),
        balance: parseFloat(row.Balance),
        num_of_products: parseInt(row.NumOfProducts),
        has_cr_card: parseInt(row.HasCrCard),
        is_active_member: parseInt(row.IsActiveMember),
        estimated_salary: parseFloat(row.EstimatedSalary)
      }));

      // Use API service for batch prediction
      const result = await apiService.predictBatch(customers);
      
      // Combine original data with predictions
      const enrichedPredictions = result.predictions.map((prediction, index) => ({
        ...data[index], // Original CSV data
        ...prediction,  // Prediction results
        id: index + 1,  // Add unique ID
      }));
      
      setPredictions({
        predictions: enrichedPredictions,
        summary: result.summary
      });
      
    } catch (error) {
      console.error('Prediction request failed:', error);
      setError(`Failed to get predictions: ${error.message}`);
      setLoading(false);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-2xl mx-auto"
    >
      <div className="card">
        <h2 className="heading-xl mb-6 text-center">Upload Customer Data</h2>
        
        <div
          className={`upload-zone ${dragActive ? 'dragover' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleChange}
            className="hidden"
          />
          
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="cursor-pointer"
            onClick={onButtonClick}
          >
            <Upload size={48} className="mx-auto mb-4 text-primary-400" />
            <h3 className="heading-lg mb-2">Drop your CSV file here</h3>
            <p className="body-text mb-4">or click to browse files</p>
            <button className="btn-primary">
              Choose File
            </button>
          </motion.div>
        </div>
        
        {/* File requirements */}
        <div className="mt-6 p-4 bg-primary-50 rounded-lg">
          <h4 className="font-medium text-primary-800 mb-2">File Requirements:</h4>
          <ul className="text-sm text-primary-700 space-y-1">
            <li>• CSV format with headers</li>
            <li>• Maximum file size: 10MB</li>
            <li>• Required columns: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary</li>
          </ul>
        </div>
        
        {/* Status messages */}
        {parseError && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3"
          >
            <AlertCircle className="text-red-500 mt-0.5" size={20} />
            <div>
              <h4 className="font-medium text-red-800">Upload Error</h4>
              <p className="text-red-700 text-sm">{parseError}</p>
            </div>
          </motion.div>
        )}
        
        {parseSuccess && !parseError && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg flex items-start space-x-3"
          >
            <CheckCircle className="text-green-500 mt-0.5" size={20} />
            <div>
              <h4 className="font-medium text-green-800">File Uploaded Successfully</h4>
              <p className="text-green-700 text-sm">Processing predictions...</p>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default CSVUploader;