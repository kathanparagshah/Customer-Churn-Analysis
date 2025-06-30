import { useContext } from 'react';
import { PredictionsContext } from '../context/PredictionsContext';

export const usePredictions = () => {
  const context = useContext(PredictionsContext);
  if (!context) {
    throw new Error('usePredictions must be used within a PredictionsProvider');
  }
  return context;
};