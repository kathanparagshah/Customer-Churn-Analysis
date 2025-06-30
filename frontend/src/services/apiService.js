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
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
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

  // Google OAuth (if backend endpoint exists)
  async saveUserToBackend(userData) {
    return this.makeRequest('/auth/google', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }
}

export default new ApiService();