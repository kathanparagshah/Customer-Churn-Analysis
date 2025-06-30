import {
  Box,
  Container,
  VStack,
  Heading,
  Text,
  Card,
  CardBody,
  useColorModeValue,
  Flex,
  Icon,
} from '@chakra-ui/react';
import { GoogleLogin } from '@react-oauth/google';
import { useAuth } from '../contexts/AuthContext';
import { Navigate } from 'react-router-dom';
import { FaShieldAlt, FaChartLine, FaUsers } from 'react-icons/fa';

const Login = () => {
  const { isAuthenticated, handleGoogleSuccess, handleGoogleFailure } = useAuth();
  const bgGradient = useColorModeValue(
    'linear(to-br, primary.500, primary.700)',
    'linear(to-br, primary.600, primary.800)'
  );
  const cardBg = useColorModeValue('white', 'gray.800');

  // Redirect if already authenticated
  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  const features = [
    {
      icon: FaChartLine,
      title: 'Advanced Analytics',
      description: 'Comprehensive customer churn prediction with ML insights'
    },
    {
      icon: FaShieldAlt,
      title: 'Enterprise Security',
      description: 'Bank-grade security with Google OAuth authentication'
    },
    {
      icon: FaUsers,
      title: 'Customer Intelligence',
      description: 'Deep insights into customer behavior and retention'
    }
  ];

  return (
    <Box minH="100vh" bgGradient={bgGradient} position="relative">
      {/* Background Pattern */}
      <Box
        position="absolute"
        top={0}
        left={0}
        right={0}
        bottom={0}
        opacity={0.1}
        backgroundImage="url('data:image/svg+xml,%3Csvg width=%2260%22 height=%2260%22 viewBox=%220 0 60 60%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cg fill=%22none%22 fill-rule=%22evenodd%22%3E%3Cg fill=%22%23ffffff%22 fill-opacity=%220.4%22%3E%3Ccircle cx=%2230%22 cy=%2230%22 r=%222%22/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')"
      />
      
      <Container maxW="6xl" py={20} position="relative">
        <VStack spacing={12} align="center">
          {/* Header */}
          <VStack spacing={4} textAlign="center">
            <Heading
              size="2xl"
              color="white"
              fontFamily="heading"
              fontWeight="900"
            >
              Customer Churn Prediction
            </Heading>
            <Text fontSize="xl" color="whiteAlpha.900" maxW="2xl">
              Professional analytics platform for predicting and preventing customer churn
              with enterprise-grade machine learning insights.
            </Text>
          </VStack>

          {/* Main Login Card */}
          <Card
            maxW="md"
            w="full"
            bg={cardBg}
            shadow="2xl"
            borderRadius="xl"
            overflow="hidden"
          >
            <CardBody p={8}>
              <VStack spacing={6}>
                <VStack spacing={2} textAlign="center">
                  <Heading size="lg" color="primary.500" fontFamily="heading">
                    Welcome Back
                  </Heading>
                  <Text color="secondary.600">
                    Sign in with your Google account to access the dashboard
                  </Text>
                </VStack>

                {/* Google Login Button */}
                <Box w="full">
                  <GoogleLogin
                    onSuccess={handleGoogleSuccess}
                    onError={handleGoogleFailure}
                    theme="outline"
                    size="large"
                    text="continue_with"
                    shape="rectangular"
                    width="100%"
                  />
                </Box>

                <Text fontSize="sm" color="secondary.500" textAlign="center">
                  By signing in, you agree to our Terms of Service and Privacy Policy
                </Text>
              </VStack>
            </CardBody>
          </Card>

          {/* Features Grid */}
          <Box w="full" maxW="4xl">
            <VStack spacing={8}>
              <Heading
                size="lg"
                color="white"
                textAlign="center"
                fontFamily="heading"
              >
                Why Choose Our Platform?
              </Heading>
              
              <Flex
                direction={{ base: 'column', md: 'row' }}
                gap={6}
                w="full"
              >
                {features.map((feature, index) => (
                  <Card
                    key={index}
                    flex={1}
                    bg="whiteAlpha.100"
                    backdropFilter="blur(10px)"
                    border="1px"
                    borderColor="whiteAlpha.200"
                    _hover={{
                      bg: 'whiteAlpha.200',
                      transform: 'translateY(-4px)'
                    }}
                    transition="all 0.3s"
                  >
                    <CardBody p={6} textAlign="center">
                      <VStack spacing={4}>
                        <Icon
                          as={feature.icon}
                          boxSize={8}
                          color="accent.400"
                        />
                        <Heading
                          size="md"
                          color="white"
                          fontFamily="heading"
                        >
                          {feature.title}
                        </Heading>
                        <Text color="whiteAlpha.900" fontSize="sm">
                          {feature.description}
                        </Text>
                      </VStack>
                    </CardBody>
                  </Card>
                ))}
              </Flex>
            </VStack>
          </Box>
        </VStack>
      </Container>
    </Box>
  );
};

export default Login;