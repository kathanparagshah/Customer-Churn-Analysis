import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Box, Spinner, VStack, Text } from '@chakra-ui/react';

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();

  // Show loading spinner while checking authentication
  if (loading) {
    return (
      <Box
        minH="100vh"
        display="flex"
        alignItems="center"
        justifyContent="center"
        bg="background.100"
      >
        <VStack spacing={4}>
          <Spinner
            size="xl"
            color="primary.500"
            thickness="4px"
          />
          <Text color="secondary.600" fontSize="lg">
            Loading...
          </Text>
        </VStack>
      </Box>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Render protected content if authenticated
  return children;
};

export default ProtectedRoute;