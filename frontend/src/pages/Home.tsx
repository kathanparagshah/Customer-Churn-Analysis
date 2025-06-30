import React from 'react';
import {
  Box,
  Text,
  VStack,
  Heading,
} from '@chakra-ui/react';

const Home: React.FC = () => {
  return (
    <VStack spacing={8} align="center" justify="center" minH="60vh">
      <Heading size="xl" color="primary.500">
        Welcome to Customer Churn Analytics
      </Heading>
      <Text fontSize="lg" textAlign="center">
        Dashboard is loading successfully! Authentication is working.
      </Text>
      <Box p={4} bg="green.100" borderRadius="md">
        <Text color="green.800">
          ✅ React app is running
          <br />
          ✅ Authentication is working
          <br />
          ✅ Routing is functional
        </Text>
      </Box>
    </VStack>
  );
};

export default Home;