import React, { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Grid,
  GridItem,
  Card,
  CardBody,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  Icon,
  Progress,
  Code,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Divider,
} from '@chakra-ui/react';
import {
  Brain,
  TrendingUp,
  Target,
  Calendar,
  BarChart3,
  Settings,
  FileText,
  Zap,
} from 'lucide-react';
import Plot from 'react-plotly.js';

interface FeatureImportance {
  feature: string;
  importance: number;
  description: string;
}

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
}

interface ModelInfo {
  name: string;
  version: string;
  lastUpdated: string;
  trainingDate: string;
  datasetSize: number;
  features: number;
}

const ModelInsights: React.FC = () => {
  const [selectedFeature, setSelectedFeature] = useState<string>('CreditScore');

  // Mock data
  const modelInfo: ModelInfo = {
    name: 'Customer Churn Prediction Model',
    version: 'v2.1.0',
    lastUpdated: '2025-01-08',
    trainingDate: '2025-01-01',
    datasetSize: 10000,
    features: 10,
  };

  const modelMetrics: ModelMetrics = {
    accuracy: 0.942,
    precision: 0.891,
    recall: 0.876,
    f1Score: 0.883,
    auc: 0.958,
  };

  const featureImportances: FeatureImportance[] = [
    { feature: 'Age', importance: 0.234, description: 'Customer age in years' },
    { feature: 'NumOfProducts', importance: 0.187, description: 'Number of bank products used' },
    { feature: 'IsActiveMember', importance: 0.156, description: 'Active membership status' },
    { feature: 'Balance', importance: 0.143, description: 'Account balance amount' },
    { feature: 'CreditScore', importance: 0.098, description: 'Credit score rating' },
    { feature: 'EstimatedSalary', importance: 0.087, description: 'Estimated annual salary' },
    { feature: 'Geography', importance: 0.065, description: 'Customer location' },
    { feature: 'Tenure', importance: 0.054, description: 'Years with the bank' },
    { feature: 'HasCrCard', importance: 0.043, description: 'Credit card ownership' },
    { feature: 'Gender', importance: 0.033, description: 'Customer gender' },
  ];

  const preprocessingSteps = [
    {
      step: 'Data Cleaning',
      description: 'Removed duplicates and handled missing values',
      details: ['Removed 127 duplicate records', 'Imputed missing values using median/mode', 'Validated data types'],
    },
    {
      step: 'Feature Engineering',
      description: 'Created new features and transformations',
      details: ['Created age groups', 'Balance-to-salary ratio', 'Product usage intensity score'],
    },
    {
      step: 'Encoding',
      description: 'Converted categorical variables to numerical',
      details: ['One-hot encoding for Geography', 'Label encoding for Gender', 'Binary encoding for boolean features'],
    },
    {
      step: 'Scaling',
      description: 'Normalized numerical features',
      details: ['StandardScaler for continuous variables', 'MinMaxScaler for bounded features'],
    },
    {
      step: 'Feature Selection',
      description: 'Selected most relevant features',
      details: ['Correlation analysis', 'Recursive feature elimination', 'Statistical significance testing'],
    },
  ];

  const modelMetadata = {
    algorithm: 'Random Forest Classifier',
    hyperparameters: {
      n_estimators: 100,
      max_depth: 10,
      min_samples_split: 5,
      min_samples_leaf: 2,
      random_state: 42,
    },
    training: {
      train_size: 0.8,
      validation_size: 0.2,
      cross_validation_folds: 5,
      early_stopping: true,
    },
    performance: {
      training_time: '2.3 minutes',
      inference_time: '0.05 seconds',
      model_size: '45.2 MB',
    },
  };

  // Generate partial dependence data
  const generatePartialDependenceData = (feature: string) => {
    const points = 50;
    let xValues: number[];
    let yValues: number[];

    switch (feature) {
      case 'Age':
        xValues = Array.from({ length: points }, (_, i) => 18 + (i * (80 - 18)) / (points - 1));
        yValues = xValues.map(x => 0.1 + 0.8 * (1 / (1 + Math.exp(-(x - 45) / 10))));
        break;
      case 'Balance':
        xValues = Array.from({ length: points }, (_, i) => (i * 250000) / (points - 1));
        yValues = xValues.map(x => 0.3 + 0.4 * Math.sin(x / 50000) * Math.exp(-x / 200000));
        break;
      case 'CreditScore':
        xValues = Array.from({ length: points }, (_, i) => 300 + (i * (850 - 300)) / (points - 1));
        yValues = xValues.map(x => 0.8 - 0.6 * (1 / (1 + Math.exp(-(x - 650) / 50))));
        break;
      default:
        xValues = Array.from({ length: points }, (_, i) => i);
        yValues = xValues.map(x => 0.5 + 0.3 * Math.sin(x / 10));
    }

    return { x: xValues, y: yValues };
  };

  const partialDependenceData = generatePartialDependenceData(selectedFeature);

  return (
    <VStack spacing={8} align="stretch">
      {/* Header */}
      <Box>
        <Text fontSize="3xl" fontWeight="bold" fontFamily="heading" color="primary.500" mb={2}>
          Model Insights
        </Text>
        <Text color="secondary.600" fontSize="lg">
          Explore model performance, feature importance, and understand how predictions are made
        </Text>
      </Box>

      {/* Top Metrics Panel */}
      <Grid templateColumns={{ base: '1fr', lg: '2fr 3fr' }} gap={8}>
        {/* Model Info */}
        <GridItem>
          <Card bg="white">
            <CardBody>
              <VStack spacing={4} align="start">
                <HStack>
                  <Icon as={Brain} color="primary.500" boxSize={6} />
                  <Text fontSize="xl" fontWeight="bold" color="primary.500">
                    Model Information
                  </Text>
                </HStack>
                
                <VStack spacing={3} align="start" w="full">
                  <HStack justify="space-between" w="full">
                    <Text fontSize="sm" color="secondary.600">Model Name:</Text>
                    <Text fontSize="sm" fontWeight="medium">{modelInfo.name}</Text>
                  </HStack>
                  
                  <HStack justify="space-between" w="full">
                    <Text fontSize="sm" color="secondary.600">Version:</Text>
                    <Badge colorScheme="blue">{modelInfo.version}</Badge>
                  </HStack>
                  
                  <HStack justify="space-between" w="full">
                    <Text fontSize="sm" color="secondary.600">Last Updated:</Text>
                    <Text fontSize="sm" fontWeight="medium">{modelInfo.lastUpdated}</Text>
                  </HStack>
                  
                  <HStack justify="space-between" w="full">
                    <Text fontSize="sm" color="secondary.600">Training Date:</Text>
                    <Text fontSize="sm" fontWeight="medium">{modelInfo.trainingDate}</Text>
                  </HStack>
                  
                  <HStack justify="space-between" w="full">
                    <Text fontSize="sm" color="secondary.600">Dataset Size:</Text>
                    <Text fontSize="sm" fontWeight="medium">{modelInfo.datasetSize.toLocaleString()}</Text>
                  </HStack>
                  
                  <HStack justify="space-between" w="full">
                    <Text fontSize="sm" color="secondary.600">Features:</Text>
                    <Text fontSize="sm" fontWeight="medium">{modelInfo.features}</Text>
                  </HStack>
                </VStack>
              </VStack>
            </CardBody>
          </Card>
        </GridItem>

        {/* Key Metrics */}
        <GridItem>
          <Grid templateColumns="repeat(3, 1fr)" gap={4}>
            <GridItem>
              <Stat
                bg="white"
                p={4}
                borderRadius="lg"
                boxShadow="sm"
                borderLeft="4px solid"
                borderLeftColor="green.500"
                textAlign="center"
              >
                <StatLabel color="secondary.600" fontSize="xs">Accuracy</StatLabel>
                <StatNumber fontSize="xl" color="primary.500">
                  {(modelMetrics.accuracy * 100).toFixed(1)}%
                </StatNumber>
                <StatHelpText fontSize="xs">Overall correctness</StatHelpText>
              </Stat>
            </GridItem>
            
            <GridItem>
              <Stat
                bg="white"
                p={4}
                borderRadius="lg"
                boxShadow="sm"
                borderLeft="4px solid"
                borderLeftColor="blue.500"
                textAlign="center"
              >
                <StatLabel color="secondary.600" fontSize="xs">Precision</StatLabel>
                <StatNumber fontSize="xl" color="primary.500">
                  {(modelMetrics.precision * 100).toFixed(1)}%
                </StatNumber>
                <StatHelpText fontSize="xs">True positive rate</StatHelpText>
              </Stat>
            </GridItem>
            
            <GridItem>
              <Stat
                bg="white"
                p={4}
                borderRadius="lg"
                boxShadow="sm"
                borderLeft="4px solid"
                borderLeftColor="purple.500"
                textAlign="center"
              >
                <StatLabel color="secondary.600" fontSize="xs">Recall</StatLabel>
                <StatNumber fontSize="xl" color="primary.500">
                  {(modelMetrics.recall * 100).toFixed(1)}%
                </StatNumber>
                <StatHelpText fontSize="xs">Sensitivity</StatHelpText>
              </Stat>
            </GridItem>
            
            <GridItem>
              <Stat
                bg="white"
                p={4}
                borderRadius="lg"
                boxShadow="sm"
                borderLeft="4px solid"
                borderLeftColor="orange.500"
                textAlign="center"
              >
                <StatLabel color="secondary.600" fontSize="xs">F1-Score</StatLabel>
                <StatNumber fontSize="xl" color="primary.500">
                  {(modelMetrics.f1Score * 100).toFixed(1)}%
                </StatNumber>
                <StatHelpText fontSize="xs">Harmonic mean</StatHelpText>
              </Stat>
            </GridItem>
            
            <GridItem>
              <Stat
                bg="white"
                p={4}
                borderRadius="lg"
                boxShadow="sm"
                borderLeft="4px solid"
                borderLeftColor="red.500"
                textAlign="center"
              >
                <StatLabel color="secondary.600" fontSize="xs">AUC-ROC</StatLabel>
                <StatNumber fontSize="xl" color="primary.500">
                  {(modelMetrics.auc * 100).toFixed(1)}%
                </StatNumber>
                <StatHelpText fontSize="xs">Area under curve</StatHelpText>
              </Stat>
            </GridItem>
          </Grid>
        </GridItem>
      </Grid>

      {/* Tabs */}
      <Card bg="white">
        <CardBody>
          <Tabs variant="line" colorScheme="primary">
            <TabList>
              <Tab _selected={{ color: 'primary.500', borderBottomColor: 'accent.500' }}>
                <HStack>
                  <Icon as={BarChart3} boxSize={4} />
                  <Text>Feature Importance</Text>
                </HStack>
              </Tab>
              <Tab _selected={{ color: 'primary.500', borderBottomColor: 'accent.500' }}>
                <HStack>
                  <Icon as={TrendingUp} boxSize={4} />
                  <Text>Partial Dependence</Text>
                </HStack>
              </Tab>
              <Tab _selected={{ color: 'primary.500', borderBottomColor: 'accent.500' }}>
                <HStack>
                  <Icon as={Settings} boxSize={4} />
                  <Text>Preprocessing</Text>
                </HStack>
              </Tab>
              <Tab _selected={{ color: 'primary.500', borderBottomColor: 'accent.500' }}>
                <HStack>
                  <Icon as={FileText} boxSize={4} />
                  <Text>Metadata</Text>
                </HStack>
              </Tab>
            </TabList>

            <TabPanels>
              {/* Feature Importance */}
              <TabPanel>
                <Grid templateColumns={{ base: '1fr', lg: '1fr 1fr' }} gap={8}>
                  <GridItem>
                    <VStack spacing={4} align="stretch">
                      <Text fontSize="lg" fontWeight="bold" color="primary.500">
                        Feature Importance Ranking
                      </Text>
                      {featureImportances.map((feature, index) => (
                        <Box key={feature.feature} p={4} bg="background.100" borderRadius="lg">
                          <HStack justify="space-between" mb={2}>
                            <Text fontWeight="medium">{feature.feature}</Text>
                            <Badge colorScheme="blue">
                              {(feature.importance * 100).toFixed(1)}%
                            </Badge>
                          </HStack>
                          <Progress
                            value={feature.importance * 100}
                            colorScheme="blue"
                            size="sm"
                            borderRadius="full"
                            mb={2}
                          />
                          <Text fontSize="sm" color="secondary.600">
                            {feature.description}
                          </Text>
                        </Box>
                      ))}
                    </VStack>
                  </GridItem>
                  
                  <GridItem>
                    <VStack spacing={4}>
                      <Text fontSize="lg" fontWeight="bold" color="primary.500">
                        Importance Distribution
                      </Text>
                      <Plot
                        data={[
                          {
                            x: featureImportances.map(f => f.importance),
                            y: featureImportances.map(f => f.feature),
                            type: 'bar',
                            orientation: 'h',
                            marker: {
                              color: '#4299E1',
                            },
                          },
                        ]}
                        layout={{
                          width: 500,
                          height: 400,
                          margin: { t: 20, b: 40, l: 120, r: 20 },
                          xaxis: { title: 'Importance Score' },
                          yaxis: { title: '' },
                        }}
                        config={{ displayModeBar: false }}
                      />
                    </VStack>
                  </GridItem>
                </Grid>
              </TabPanel>

              {/* Partial Dependence */}
              <TabPanel>
                <Grid templateColumns={{ base: '1fr', lg: '1fr 2fr' }} gap={8}>
                  <GridItem>
                    <VStack spacing={4} align="stretch">
                      <Text fontSize="lg" fontWeight="bold" color="primary.500">
                        Select Feature
                      </Text>
                      <VStack spacing={2} align="stretch">
                        {['Age', 'Balance', 'CreditScore', 'NumOfProducts'].map((feature) => (
                          <Box
                            key={feature}
                            p={3}
                            bg={selectedFeature === feature ? 'primary.50' : 'background.100'}
                            borderRadius="lg"
                            cursor="pointer"
                            border={selectedFeature === feature ? '2px solid' : '1px solid'}
                            borderColor={selectedFeature === feature ? 'primary.500' : 'gray.200'}
                            onClick={() => setSelectedFeature(feature)}
                            _hover={{ bg: 'primary.50' }}
                          >
                            <Text fontWeight="medium" color={selectedFeature === feature ? 'primary.500' : 'gray.700'}>
                              {feature}
                            </Text>
                          </Box>
                        ))}
                      </VStack>
                    </VStack>
                  </GridItem>
                  
                  <GridItem>
                    <VStack spacing={4}>
                      <Text fontSize="lg" fontWeight="bold" color="primary.500">
                        Partial Dependence Plot - {selectedFeature}
                      </Text>
                      <Plot
                        data={[
                          {
                            x: partialDependenceData.x,
                            y: partialDependenceData.y,
                            type: 'scatter',
                            mode: 'lines',
                            line: {
                              color: '#4299E1',
                              width: 3,
                            },
                          },
                        ]}
                        layout={{
                          width: 600,
                          height: 400,
                          margin: { t: 20, b: 60, l: 60, r: 20 },
                          xaxis: { title: selectedFeature },
                          yaxis: { title: 'Partial Dependence' },
                        }}
                        config={{ displayModeBar: false }}
                      />
                      <Text fontSize="sm" color="secondary.600" textAlign="center">
                        Shows how the model's prediction changes as {selectedFeature} varies,
                        while keeping other features at their average values.
                      </Text>
                    </VStack>
                  </GridItem>
                </Grid>
              </TabPanel>

              {/* Preprocessing Steps */}
              <TabPanel>
                <VStack spacing={6} align="stretch">
                  <Text fontSize="lg" fontWeight="bold" color="primary.500">
                    Data Preprocessing Pipeline
                  </Text>
                  
                  <Accordion allowMultiple>
                    {preprocessingSteps.map((step, index) => (
                      <AccordionItem key={index} border="1px" borderColor="gray.200" borderRadius="lg" mb={4}>
                        <AccordionButton bg="background.100" _hover={{ bg: 'background.200' }}>
                          <Box flex="1" textAlign="left">
                            <HStack>
                              <Badge colorScheme="blue" mr={2}>
                                Step {index + 1}
                              </Badge>
                              <Text fontWeight="bold">{step.step}</Text>
                            </HStack>
                            <Text fontSize="sm" color="secondary.600" mt={1}>
                              {step.description}
                            </Text>
                          </Box>
                          <AccordionIcon />
                        </AccordionButton>
                        <AccordionPanel pb={4}>
                          <VStack spacing={2} align="start">
                            {step.details.map((detail, detailIndex) => (
                              <HStack key={detailIndex}>
                                <Icon as={Zap} color="accent.500" boxSize={4} />
                                <Text fontSize="sm">{detail}</Text>
                              </HStack>
                            ))}
                          </VStack>
                        </AccordionPanel>
                      </AccordionItem>
                    ))}
                  </Accordion>
                </VStack>
              </TabPanel>

              {/* Metadata */}
              <TabPanel>
                <VStack spacing={6} align="stretch">
                  <Text fontSize="lg" fontWeight="bold" color="primary.500">
                    Model Configuration & Metadata
                  </Text>
                  
                  <Grid templateColumns={{ base: '1fr', lg: 'repeat(2, 1fr)' }} gap={6}>
                    <GridItem>
                      <Card bg="background.50">
                        <CardBody>
                          <Text fontSize="md" fontWeight="bold" color="primary.500" mb={4}>
                            Algorithm Configuration
                          </Text>
                          <Code display="block" whiteSpace="pre" fontSize="sm" p={4}>
                            {JSON.stringify({
                              algorithm: modelMetadata.algorithm,
                              hyperparameters: modelMetadata.hyperparameters,
                            }, null, 2)}
                          </Code>
                        </CardBody>
                      </Card>
                    </GridItem>
                    
                    <GridItem>
                      <Card bg="background.50">
                        <CardBody>
                          <Text fontSize="md" fontWeight="bold" color="primary.500" mb={4}>
                            Training Configuration
                          </Text>
                          <Code display="block" whiteSpace="pre" fontSize="sm" p={4}>
                            {JSON.stringify(modelMetadata.training, null, 2)}
                          </Code>
                        </CardBody>
                      </Card>
                    </GridItem>
                    
                    <GridItem colSpan={{ base: 1, lg: 2 }}>
                      <Card bg="background.50">
                        <CardBody>
                          <Text fontSize="md" fontWeight="bold" color="primary.500" mb={4}>
                            Performance Metrics
                          </Text>
                          <Code display="block" whiteSpace="pre" fontSize="sm" p={4}>
                            {JSON.stringify(modelMetadata.performance, null, 2)}
                          </Code>
                        </CardBody>
                      </Card>
                    </GridItem>
                  </Grid>
                </VStack>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </CardBody>
      </Card>
    </VStack>
  );
};

export default ModelInsights;