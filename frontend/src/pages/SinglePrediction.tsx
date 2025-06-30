import React, { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Button,
  Grid,
  GridItem,
  Card,
  CardBody,
  CardHeader,
  FormControl,
  FormLabel,
  Input,
  Select,
  NumberInput,
  NumberInputField,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Progress,
  Badge,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Spinner,
  Icon,
  CircularProgress,
  CircularProgressLabel,
} from '@chakra-ui/react';
import {
  User,
  CreditCard,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Target,
  Brain,
} from 'lucide-react';
import apiService from '../services/apiService';

interface CustomerData {
  creditScore: string;
  geography: string;
  gender: string;
  age: string;
  tenure: string;
  balance: string;
  numOfProducts: string;
  hasCrCard: string;
  isActiveMember: string;
  estimatedSalary: string;
}

interface PredictionResult {
  churnProbability: number;
  riskLevel: string;
  confidence: number;
  prediction: string;
  shapValues?: Record<string, number>;
}

const SinglePrediction: React.FC = () => {
  const [customerData, setCustomerData] = useState<CustomerData>({
    creditScore: '',
    geography: '',
    gender: '',
    age: '',
    tenure: '',
    balance: '',
    numOfProducts: '',
    hasCrCard: '',
    isActiveMember: '',
    estimatedSalary: '',
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: keyof CustomerData, value: string) => {
    setCustomerData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Convert form data to API format
      const apiData = {
        credit_score: parseInt(customerData.creditScore),
        geography: customerData.geography,
        gender: customerData.gender,
        age: parseInt(customerData.age),
        tenure: parseInt(customerData.tenure),
        balance: parseFloat(customerData.balance),
        num_of_products: parseInt(customerData.numOfProducts),
        has_cr_card: parseInt(customerData.hasCrCard),
        is_active_member: parseInt(customerData.isActiveMember),
        estimated_salary: parseFloat(customerData.estimatedSalary),
      };
      
      // Make API call
      const response = await apiService.predictSingle(apiData);
      
      // Transform API response to match component interface
      const result: PredictionResult = {
        churnProbability: response.churn_probability,
        riskLevel: response.risk_level,
        confidence: response.confidence,
        prediction: response.prediction === 1 ? 'Will Churn' : 'Will Stay',
        shapValues: response.feature_importance || {},
      };
      
      setResult(result);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err instanceof Error ? err.message : 'Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const isFormValid = Object.values(customerData).every(value => value !== '');

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'High': return 'red';
      case 'Medium': return 'yellow';
      case 'Low': return 'green';
      default: return 'gray';
    }
  };

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'High': return AlertTriangle;
      case 'Medium': return TrendingUp;
      case 'Low': return CheckCircle;
      default: return TrendingDown;
    }
  };

  return (
    <VStack spacing={8} align="stretch">
      {/* Header */}
      <Box>
        <Text fontSize="3xl" fontWeight="bold" fontFamily="heading" color="primary.500" mb={2}>
          Single Customer Prediction
        </Text>
        <Text color="secondary.600" fontSize="lg">
          Enter customer information to get real-time churn prediction and risk assessment
        </Text>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert status="error" borderRadius="lg">
          <AlertIcon />
          <AlertTitle>Prediction Failed!</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Grid templateColumns={{ base: '1fr', lg: '1fr 1fr' }} gap={8}>
        {/* Personal Information */}
        <GridItem>
          <Card bg="white">
            <CardHeader>
              <HStack>
                <Icon as={User} color="primary.500" boxSize={5} />
                <Text fontSize="xl" fontWeight="bold" color="primary.500">
                  Personal Information
                </Text>
              </HStack>
            </CardHeader>
            <CardBody>
              <VStack spacing={4}>
                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Credit Score</FormLabel>
                  <NumberInput>
                    <NumberInputField
                      placeholder="Enter credit score (300-850)"
                      value={customerData.creditScore}
                      onChange={(e) => handleInputChange('creditScore', e.target.value)}
                    />
                  </NumberInput>
                </FormControl>

                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Geography</FormLabel>
                  <Select
                    placeholder="Select country"
                    value={customerData.geography}
                    onChange={(e) => handleInputChange('geography', e.target.value)}
                  >
                    <option value="France">France</option>
                    <option value="Germany">Germany</option>
                    <option value="Spain">Spain</option>
                  </Select>
                </FormControl>

                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Gender</FormLabel>
                  <Select
                    placeholder="Select gender"
                    value={customerData.gender}
                    onChange={(e) => handleInputChange('gender', e.target.value)}
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </Select>
                </FormControl>

                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Age</FormLabel>
                  <NumberInput>
                    <NumberInputField
                      placeholder="Enter age"
                      value={customerData.age}
                      onChange={(e) => handleInputChange('age', e.target.value)}
                    />
                  </NumberInput>
                </FormControl>

                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Estimated Salary</FormLabel>
                  <NumberInput>
                    <NumberInputField
                      placeholder="Enter estimated salary"
                      value={customerData.estimatedSalary}
                      onChange={(e) => handleInputChange('estimatedSalary', e.target.value)}
                    />
                  </NumberInput>
                </FormControl>
              </VStack>
            </CardBody>
          </Card>
        </GridItem>

        {/* Account Information */}
        <GridItem>
          <Card bg="white">
            <CardHeader>
              <HStack>
                <Icon as={CreditCard} color="primary.500" boxSize={5} />
                <Text fontSize="xl" fontWeight="bold" color="primary.500">
                  Account Information
                </Text>
              </HStack>
            </CardHeader>
            <CardBody>
              <VStack spacing={4}>
                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Tenure (Years)</FormLabel>
                  <NumberInput>
                    <NumberInputField
                      placeholder="Years with bank"
                      value={customerData.tenure}
                      onChange={(e) => handleInputChange('tenure', e.target.value)}
                    />
                  </NumberInput>
                </FormControl>

                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Account Balance</FormLabel>
                  <NumberInput>
                    <NumberInputField
                      placeholder="Current balance"
                      value={customerData.balance}
                      onChange={(e) => handleInputChange('balance', e.target.value)}
                    />
                  </NumberInput>
                </FormControl>

                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Number of Products</FormLabel>
                  <Select
                    placeholder="Select number of products"
                    value={customerData.numOfProducts}
                    onChange={(e) => handleInputChange('numOfProducts', e.target.value)}
                  >
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                  </Select>
                </FormControl>

                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Has Credit Card</FormLabel>
                  <Select
                    placeholder="Select option"
                    value={customerData.hasCrCard}
                    onChange={(e) => handleInputChange('hasCrCard', e.target.value)}
                  >
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                  </Select>
                </FormControl>

                <FormControl>
                  <FormLabel color="secondary.600" fontWeight="medium">Is Active Member</FormLabel>
                  <Select
                    placeholder="Select option"
                    value={customerData.isActiveMember}
                    onChange={(e) => handleInputChange('isActiveMember', e.target.value)}
                  >
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                  </Select>
                </FormControl>
              </VStack>
            </CardBody>
          </Card>
        </GridItem>
      </Grid>

      {/* Predict Button */}
      <Box textAlign="center">
        <Button
          size="lg"
          onClick={handlePredict}
          isDisabled={!isFormValid || loading}
          isLoading={loading}
          loadingText="Analyzing..."
          bg="accent.500"
          color="white"
          _hover={{ bg: 'accent.600' }}
          px={12}
        >
          {loading ? (
            <HStack>
              <Spinner size="sm" />
              <Text>Predicting...</Text>
            </HStack>
          ) : (
            'Predict Churn Risk'
          )}
        </Button>
      </Box>

      {/* Results */}
      {result && (
        <Box>
          <Text fontSize="2xl" fontWeight="bold" fontFamily="heading" color="primary.500" mb={6}>
            Prediction Results
          </Text>
          
          <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }} gap={6} mb={8}>
            {/* Churn Probability */}
            <GridItem>
              <Card bg="white" textAlign="center">
                <CardBody>
                  <VStack spacing={4}>
                    <CircularProgress
                      value={result.churnProbability * 100}
                      size="120px"
                      color={result.churnProbability > 0.7 ? 'red.500' : result.churnProbability > 0.4 ? 'yellow.500' : 'green.500'}
                      thickness="8px"
                    >
                      <CircularProgressLabel fontSize="lg" fontWeight="bold">
                        {(result.churnProbability * 100).toFixed(1)}%
                      </CircularProgressLabel>
                    </CircularProgress>
                    <VStack spacing={1}>
                      <Text fontSize="lg" fontWeight="bold" color="primary.500">
                        Churn Probability
                      </Text>
                      <Text fontSize="sm" color="secondary.600">
                        Likelihood to leave
                      </Text>
                    </VStack>
                  </VStack>
                </CardBody>
              </Card>
            </GridItem>

            {/* Risk Level */}
            <GridItem>
              <Card bg="white" textAlign="center">
                <CardBody>
                  <VStack spacing={4}>
                    <Box p={4} bg={`${getRiskColor(result.riskLevel)}.100`} borderRadius="full">
                      <Icon as={getRiskIcon(result.riskLevel)} color={`${getRiskColor(result.riskLevel)}.500`} boxSize={12} />
                    </Box>
                    <VStack spacing={1}>
                      <Badge colorScheme={getRiskColor(result.riskLevel)} fontSize="lg" px={3} py={1}>
                        {result.riskLevel} Risk
                      </Badge>
                      <Text fontSize="sm" color="secondary.600">
                        Risk classification
                      </Text>
                    </VStack>
                  </VStack>
                </CardBody>
              </Card>
            </GridItem>

            {/* Confidence */}
            <GridItem>
              <Card bg="white">
                <CardBody>
                  <VStack spacing={4}>
                    <Stat textAlign="center">
                      <StatNumber fontSize="3xl" color="primary.500">
                        {(result.confidence * 100).toFixed(1)}%
                      </StatNumber>
                      <StatLabel color="secondary.600" fontWeight="medium">
                        Model Confidence
                      </StatLabel>
                      <StatHelpText>
                        <Progress
                          value={result.confidence * 100}
                          colorScheme="blue"
                          size="sm"
                          borderRadius="full"
                        />
                      </StatHelpText>
                    </Stat>
                  </VStack>
                </CardBody>
              </Card>
            </GridItem>

            {/* Prediction */}
            <GridItem>
              <Card bg="white" textAlign="center">
                <CardBody>
                  <VStack spacing={4}>
                    <Box p={4} bg={result.prediction === 'Will Churn' ? 'red.100' : 'green.100'} borderRadius="full">
                      <Icon
                        as={result.prediction === 'Will Churn' ? TrendingDown : TrendingUp}
                        color={result.prediction === 'Will Churn' ? 'red.500' : 'green.500'}
                        boxSize={12}
                      />
                    </Box>
                    <VStack spacing={1}>
                      <Text fontSize="lg" fontWeight="bold" color="primary.500">
                        {result.prediction}
                      </Text>
                      <Text fontSize="sm" color="secondary.600">
                        Final prediction
                      </Text>
                    </VStack>
                  </VStack>
                </CardBody>
              </Card>
            </GridItem>
          </Grid>

          {/* SHAP Explanation */}
          {result.shapValues && (
            <Accordion allowToggle>
              <AccordionItem border="1px" borderColor="gray.200" borderRadius="lg">
                <AccordionButton bg="white" _hover={{ bg: 'background.100' }}>
                  <Box flex="1" textAlign="left">
                    <HStack>
                      <Icon as={Brain} color="primary.500" boxSize={5} />
                      <Text fontSize="lg" fontWeight="bold" color="primary.500">
                        Feature Importance (SHAP Values)
                      </Text>
                    </HStack>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={4} bg="background.50">
                  <VStack spacing={3} align="stretch">
                    {Object.entries(result.shapValues).map(([feature, value]) => (
                      <HStack key={feature} justify="space-between">
                        <Text fontWeight="medium">{feature}</Text>
                        <HStack>
                          <Progress
                            value={Math.abs(value) * 500}
                            colorScheme={value > 0 ? 'red' : 'green'}
                            size="sm"
                            w="100px"
                          />
                          <Text
                            fontSize="sm"
                            fontWeight="bold"
                            color={value > 0 ? 'red.500' : 'green.500'}
                            w="60px"
                            textAlign="right"
                          >
                            {value > 0 ? '+' : ''}{value.toFixed(3)}
                          </Text>
                        </HStack>
                      </HStack>
                    ))}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          )}
        </Box>
      )}
    </VStack>
  );
};

export default SinglePrediction;