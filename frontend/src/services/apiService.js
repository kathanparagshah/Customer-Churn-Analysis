const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

class ApiService {
  async makeRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    const response = await fetch(url, config);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      // Throw the full error data as JSON string so frontend can parse it
      throw new Error(JSON.stringify(errorData));
    }
    
    return response.json();
  }

  // Single prediction
  async predictSingle(customerData) {
    return this.makeRequest('/predict', {
      method: 'POST',
      body: JSON.stringify(customerData),
    });
  }

  // Batch prediction
  async predictBatch(customers) {
    return this.makeRequest('/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ customers }),
    });
  }

  // Health check
  async healthCheck() {
    return this.makeRequest('/health');
  }

  // Model info
  async getModelInfo() {
    return this.makeRequest('/model/info');
  }

  // Analytics endpoints
  async getDailyMetrics(days = 30) {
    return this.makeRequest(`/analytics/daily-metrics?days=${days}`);
  }

  async getPredictionTrends(days = 30) {
    return this.makeRequest(`/analytics/prediction-trends?days=${days}`);
  }

  async getRiskDistribution(days = 30) {
    return this.makeRequest(`/analytics/risk-distribution?days=${days}`);
  }

  async getAnalyticsDashboard(days = 30) {
    return this.makeRequest(`/analytics/dashboard?days=${days}`);
  }

  // Google OAuth (if backend endpoint exists)
  async saveUserToBackend(userData) {
    return this.makeRequest('/auth/google', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }
}

export default new ApiService();