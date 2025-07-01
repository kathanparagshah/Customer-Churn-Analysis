import React from 'react';
import {
  Box,
  Text,
  VStack,
  HStack,
  Heading,
  Container,
  Grid,
  GridItem,
  Card,
  CardBody,
  Icon,
  Button,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  useColorModeValue,
} from '@chakra-ui/react';
import { FaChartLine, FaUsers, FaShieldAlt, FaBrain, FaArrowRight, FaDatabase } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';

const Home: React.FC = () => {
  const navigate = useNavigate();
  const cardBg = useColorModeValue('white', 'gray.800');
  const statBg = useColorModeValue('gray.50', 'gray.700');
  
  return (
    <Box>
      {/* Hero Section */}
      <Container maxW="7xl" py={16}>
        <VStack spacing={8} textAlign="center">
          <Heading 
            size="2xl" 
            color="primary.500"
            fontWeight="bold"
            lineHeight="shorter"
          >
            Advanced Customer Analytics
          </Heading>
          <Text 
            fontSize="xl" 
            color="gray.600" 
            maxW="2xl"
            lineHeight="tall"
          >
            Leverage machine learning to predict customer behavior, reduce churn, 
            and optimize retention strategies with enterprise-grade analytics.
          </Text>
          <HStack spacing={4} pt={4}>
            <Button 
              colorScheme="primary" 
              size="lg" 
              rightIcon={<FaArrowRight />}
              onClick={() => navigate('/predict')}
            >
              Start Analysis
            </Button>
            <Button 
              variant="outline" 
              size="lg"
              onClick={() => navigate('/analytics')}
            >
              View Dashboard
            </Button>
          </HStack>
        </VStack>
      </Container>

      {/* Key Metrics */}
      <Box bg={statBg} py={12}>
        <Container maxW="7xl">
          <Grid templateColumns={{ base: '1fr', md: 'repeat(4, 1fr)' }} gap={8}>
            <Stat textAlign="center">
              <StatNumber fontSize="3xl" color="primary.500">95%</StatNumber>
              <StatLabel fontSize="md">Prediction Accuracy</StatLabel>
              <StatHelpText>Machine Learning Model</StatHelpText>
            </Stat>
            <Stat textAlign="center">
              <StatNumber fontSize="3xl" color="primary.500">24/7</StatNumber>
              <StatLabel fontSize="md">Real-time Monitoring</StatLabel>
              <StatHelpText>Continuous Analysis</StatHelpText>
            </Stat>
            <Stat textAlign="center">
              <StatNumber fontSize="3xl" color="primary.500">10+</StatNumber>
              <StatLabel fontSize="md">Key Indicators</StatLabel>
              <StatHelpText>Comprehensive Metrics</StatHelpText>
            </Stat>
            <Stat textAlign="center">
              <StatNumber fontSize="3xl" color="primary.500">∞</StatNumber>
              <StatLabel fontSize="md">Scalable Processing</StatLabel>
              <StatHelpText>Enterprise Ready</StatHelpText>
            </Stat>
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxW="7xl" py={16}>
        <VStack spacing={12}>
          <VStack spacing={4} textAlign="center">
            <Heading size="xl" color="gray.800">
              Comprehensive Analytics Suite
            </Heading>
            <Text fontSize="lg" color="gray.600" maxW="2xl">
              Our platform provides end-to-end customer analytics with advanced 
              machine learning capabilities designed for financial institutions.
            </Text>
          </VStack>
          
          <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }} gap={8}>
            <Card bg={cardBg} shadow="lg" _hover={{ transform: 'translateY(-4px)', transition: 'all 0.2s' }}>
              <CardBody>
                <VStack spacing={4} align="start">
                  <Icon as={FaBrain} w={8} h={8} color="primary.500" />
                  <Heading size="md">Predictive Modeling</Heading>
                  <Text color="gray.600">
                    Advanced machine learning algorithms to predict customer churn 
                    with high accuracy and actionable insights.
                  </Text>
                </VStack>
              </CardBody>
            </Card>

            <Card bg={cardBg} shadow="lg" _hover={{ transform: 'translateY(-4px)', transition: 'all 0.2s' }}>
              <CardBody>
                <VStack spacing={4} align="start">
                  <Icon as={FaChartLine} w={8} h={8} color="primary.500" />
                  <Heading size="md">Real-time Analytics</Heading>
                  <Text color="gray.600">
                    Monitor customer behavior patterns and trends in real-time 
                    with interactive dashboards and visualizations.
                  </Text>
                </VStack>
              </CardBody>
            </Card>

            <Card bg={cardBg} shadow="lg" _hover={{ transform: 'translateY(-4px)', transition: 'all 0.2s' }}>
              <CardBody>
                <VStack spacing={4} align="start">
                  <Icon as={FaUsers} w={8} h={8} color="primary.500" />
                  <Heading size="md">Customer Segmentation</Heading>
                  <Text color="gray.600">
                    Identify distinct customer segments and tailor retention 
                    strategies for maximum effectiveness.
                  </Text>
                </VStack>
              </CardBody>
            </Card>

            <Card bg={cardBg} shadow="lg" _hover={{ transform: 'translateY(-4px)', transition: 'all 0.2s' }}>
              <CardBody>
                <VStack spacing={4} align="start">
                  <Icon as={FaDatabase} w={8} h={8} color="primary.500" />
                  <Heading size="md">Batch Processing</Heading>
                  <Text color="gray.600">
                    Process large datasets efficiently with bulk prediction 
                    capabilities and automated reporting.
                  </Text>
                </VStack>
              </CardBody>
            </Card>

            <Card bg={cardBg} shadow="lg" _hover={{ transform: 'translateY(-4px)', transition: 'all 0.2s' }}>
              <CardBody>
                <VStack spacing={4} align="start">
                  <Icon as={FaShieldAlt} w={8} h={8} color="primary.500" />
                  <Heading size="md">Enterprise Security</Heading>
                  <Text color="gray.600">
                    Bank-grade security with encrypted data transmission 
                    and secure authentication protocols.
                  </Text>
                </VStack>
              </CardBody>
            </Card>

            <Card bg={cardBg} shadow="lg" _hover={{ transform: 'translateY(-4px)', transition: 'all 0.2s' }}>
              <CardBody>
                <VStack spacing={4} align="start">
                  <Icon as={FaArrowRight} w={8} h={8} color="primary.500" />
                  <Heading size="md">Actionable Insights</Heading>
                  <Text color="gray.600">
                    Transform data into strategic recommendations with 
                    clear action items and performance metrics.
                  </Text>
                </VStack>
              </CardBody>
            </Card>
          </Grid>
        </VStack>
      </Container>

      {/* Call to Action */}
      <Box bg="primary.500" color="white" py={16}>
        <Container maxW="4xl" textAlign="center">
          <VStack spacing={6}>
            <Heading size="xl">
              Ready to Transform Your Customer Analytics?
            </Heading>
            <Text fontSize="lg" opacity={0.9}>
              Start making data-driven decisions today with our comprehensive 
              customer churn prediction platform.
            </Text>
            <Button 
              size="lg" 
              bg="white" 
              color="primary.500" 
              _hover={{ bg: 'gray.100' }}
              rightIcon={<FaArrowRight />}
              onClick={() => navigate('/predict')}
            >
              Get Started Now
            </Button>
          </VStack>
        </Container>
      </Box>
    </Box>
  );
};

export default Home;