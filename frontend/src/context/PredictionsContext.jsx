import { useReducer } from 'react';
import { PredictionsContext } from '../contexts/PredictionsContextDefinition';

// Initial state
const initialState = {
  predictions: [],
  loading: false,
  error: null,
  summary: null,
  viewMode: 'table', // 'table' or 'cards'
  uploadedFile: null,
};

// Action types
const ACTIONS = {
  SET_LOADING: 'SET_LOADING',
  SET_PREDICTIONS: 'SET_PREDICTIONS',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',
  SET_VIEW_MODE: 'SET_VIEW_MODE',
  SET_UPLOADED_FILE: 'SET_UPLOADED_FILE',
  RESET_STATE: 'RESET_STATE',
};

// Reducer function
function predictionsReducer(state, action) {
  switch (action.type) {
    case ACTIONS.SET_LOADING:
      return {
        ...state,
        loading: action.payload,
        error: action.payload ? null : state.error, // Clear error when starting new request
      };
    
    case ACTIONS.SET_PREDICTIONS:
      return {
        ...state,
        predictions: action.payload.predictions || [],
        summary: action.payload.summary || null,
        loading: false,
        error: null,
      };
    
    case ACTIONS.SET_ERROR:
      return {
        ...state,
        error: action.payload,
        loading: false,
      };
    
    case ACTIONS.CLEAR_ERROR:
      return {
        ...state,
        error: null,
      };
    
    case ACTIONS.SET_VIEW_MODE:
      return {
        ...state,
        viewMode: action.payload,
      };
    
    case ACTIONS.SET_UPLOADED_FILE:
      return {
        ...state,
        uploadedFile: action.payload,
      };
    
    case ACTIONS.RESET_STATE:
      return {
        ...initialState,
        viewMode: state.viewMode, // Preserve view mode preference
      };
    
    default:
      return state;
  }
}

// Provider component
export function PredictionsProvider({ children }) {
  const [state, dispatch] = useReducer(predictionsReducer, initialState);

  // Action creators
  const actions = {
    setLoading: (loading) => {
      dispatch({ type: ACTIONS.SET_LOADING, payload: loading });
    },
    
    setPredictions: (data) => {
      dispatch({ type: ACTIONS.SET_PREDICTIONS, payload: data });
    },
    
    setError: (error) => {
      dispatch({ type: ACTIONS.SET_ERROR, payload: error });
    },
    
    clearError: () => {
      dispatch({ type: ACTIONS.CLEAR_ERROR });
    },
    
    setViewMode: (mode) => {
      dispatch({ type: ACTIONS.SET_VIEW_MODE, payload: mode });
    },
    
    setUploadedFile: (file) => {
      dispatch({ type: ACTIONS.SET_UPLOADED_FILE, payload: file });
    },
    
    resetState: () => {
      dispatch({ type: ACTIONS.RESET_STATE });
    },
  };

  const value = {
    ...state,
    ...actions,
  };

  return (
    <PredictionsContext.Provider value={value}>
      {children}
    </PredictionsContext.Provider>
  );
}