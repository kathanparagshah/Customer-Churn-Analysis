import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { PredictionsProvider } from './context/PredictionsContext';
import { AuthProvider } from './contexts/AuthContext';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './pages/Login';
import Home from './pages/Home.tsx';
import SinglePrediction from './pages/SinglePrediction.tsx';
import BatchPredictions from './pages/BatchPredictions.tsx';
import ModelInsights from './pages/ModelInsights.tsx';
import AnalyticsDashboard from './pages/AnalyticsDashboard.tsx';

function App() {
  const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || 'your-google-client-id';
   
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <AuthProvider>
        <PredictionsProvider>
        <Router>
          <Routes>
            {/* Public route */}
            <Route path="/login" element={<Login />} />
            
            {/* Protected routes */}
            <Route path="/" element={
              <ProtectedRoute>
                <Layout>
                  <Home />
                </Layout>
              </ProtectedRoute>
            } />
            <Route path="/single-prediction" element={
              <ProtectedRoute>
                <Layout>
                  <SinglePrediction />
                </Layout>
              </ProtectedRoute>
            } />
            <Route path="/batch-predictions" element={
              <ProtectedRoute>
                <Layout>
                  <BatchPredictions />
                </Layout>
              </ProtectedRoute>
            } />
            <Route path="/model-insights" element={
              <ProtectedRoute>
                <Layout>
                  <ModelInsights />
                </Layout>
              </ProtectedRoute>
            } />
            <Route path="/analytics-dashboard" element={
              <ProtectedRoute>
                <Layout>
                  <AnalyticsDashboard />
                </Layout>
              </ProtectedRoute>
            } />
          </Routes>
        </Router>
        </PredictionsProvider>
      </AuthProvider>
    </GoogleOAuthProvider>
  );
}

export default App;