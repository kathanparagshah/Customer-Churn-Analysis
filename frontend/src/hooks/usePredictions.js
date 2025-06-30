import { useContext } from 'react';
import { PredictionsContext } from '../contexts/PredictionsContextDefinition';

export const usePredictions = () => {
  const context = useContext(PredictionsContext);
  if (!context) {
    throw new Error('usePredictions must be used within a PredictionsProvider');
  }
  return context;
};