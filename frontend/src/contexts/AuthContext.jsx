import { createContext, useState, useEffect } from 'react';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);



  useEffect(() => {
    // Check if user is already logged in (from localStorage)
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      try {
        const userData = JSON.parse(savedUser);
        setUser(userData);
        setIsAuthenticated(true);
      } catch (error) {
        console.error('Error parsing saved user data:', error);
        localStorage.removeItem('user');
      }
    }
    setLoading(false);
  }, []);

  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      // Decode the JWT token to get user information
      const decoded = JSON.parse(atob(credentialResponse.credential.split('.')[1]));
      
      // Create user object
      const userData = {
        id: decoded.sub,
        name: decoded.name,
        email: decoded.email,
        avatar: decoded.picture,
        tokenId: credentialResponse.credential,
        loginTime: new Date().toISOString()
      };

      // Save user to backend (mock API call)
      await saveUserToBackend(userData);
      
      // Save to localStorage and state
      localStorage.setItem('user', JSON.stringify(userData));
      setUser(userData);
      setIsAuthenticated(true);
      
      console.log('Google login successful:', userData);
    } catch (error) {
      console.error('Error handling Google login:', error);
    }
  };

  const handleGoogleFailure = (error) => {
    console.error('Google login failed:', error);
  };

  const logout = () => {
    localStorage.removeItem('user');
    setUser(null);
    setIsAuthenticated(false);
    
    // Note: With @react-oauth/google, logout is handled by the GoogleLogin component
    console.log('User logged out successfully');
  };

  // Mock function to save user to backend
  const saveUserToBackend = async (userData) => {
    try {
      // In a real application, this would be an API call to your backend
      const response = await fetch('/api/auth/google', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });
      
      if (!response.ok) {
        throw new Error('Failed to save user to backend');
      }
      
      return await response.json();
    } catch (error) {
      // For demo purposes, we'll just log the error and continue
      console.log('Backend not available, continuing with local auth:', error.message);
      return { success: true, message: 'Local authentication successful' };
    }
  };

  const value = {
    user,
    isAuthenticated,
    loading,
    handleGoogleSuccess,
    handleGoogleFailure,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthProvider;