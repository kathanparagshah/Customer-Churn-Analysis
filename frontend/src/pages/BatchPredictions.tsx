import React, { useState, useRef } from 'react';
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
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Icon,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Spinner,
  Progress,
  Flex,
  Divider,
} from '@chakra-ui/react';
import {
  Upload,
  FileText,
  Download,
  Users,
  TrendingDown,
  Percent,
  Target,
  AlertCircle,
} from 'lucide-react';
import Plot from 'react-plotly.js';
import { PlotData, Layout } from 'plotly.js';
import apiService from '../services/apiService';

interface BatchResult {
  id: string;
  customerName: string;
  churnProbability: number;
  riskLevel: string;
  prediction: string;
  confidence: number;
}

interface BatchSummary {
  totalCustomers: number;
  predictedChurners: number;
  churnRate: number;
  averageProbability: number;
}

const BatchPredictions: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BatchResult[]>([]);
  const [summary, setSummary] = useState<BatchSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile: File) => {
    if (!selectedFile.name.toLowerCase().endsWith('.csv')) {
      setError('Please select a CSV file.');
      return;
    }
    setFile(selectedFile);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Parse CSV file
      const text = await file.text();
      const lines = text.split('\n').filter(line => line.trim());
      const headers = lines[0].split(',').map(h => h.trim());
      
      // Convert CSV data to API format
      const customers = lines.slice(1).map((line, index) => {
        const values = line.split(',').map(v => v.trim());
        const customer: any = {};
        
        headers.forEach((header, i) => {
          const value = values[i];
          switch (header.toLowerCase()) {
            case 'creditscore':
              customer.CreditScore = parseInt(value);
              break;
            case 'geography':
              customer.Geography = value;
              break;
            case 'gender':
              customer.Gender = value;
              break;
            case 'age':
              customer.Age = parseInt(value);
              break;
            case 'tenure':
              customer.Tenure = parseInt(value);
              break;
            case 'balance':
              customer.Balance = parseFloat(value);
              break;
            case 'numofproducts':
              customer.NumOfProducts = parseInt(value);
              break;
            case 'hascrcard':
              customer.HasCrCard = parseInt(value);
              break;
            case 'isactivemember':
              customer.IsActiveMember = parseInt(value);
              break;
            case 'estimatedsalary':
              customer.EstimatedSalary = parseFloat(value);
              break;
          }
        });
        
        return customer;
      });
      
      // Make API call
      const response = await apiService.predictBatch(customers);
      
      // Transform API response to match component interface
      const results: BatchResult[] = response.predictions.map((pred: any, index: number) => ({
        id: `CUST-${String(index + 1).padStart(3, '0')}`,
        customerName: `Customer ${index + 1}`,
        churnProbability: pred.churn_probability,
        riskLevel: pred.risk_level,
        prediction: pred.prediction === 1 ? 'Will Churn' : 'Will Stay',
        confidence: pred.confidence,
      }));
      
      const churners = results.filter(r => r.prediction === 'Will Churn').length;
      const summary: BatchSummary = {
        totalCustomers: results.length,
        predictedChurners: churners,
        churnRate: churners / results.length,
        averageProbability: results.reduce((sum, r) => sum + r.churnProbability, 0) / results.length,
      };
      
      setResults(results);
      setSummary(summary);
    } catch (err) {
      console.error('Batch prediction error:', err);
      setError(err instanceof Error ? err.message : 'Failed to process file. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const downloadSampleCSV = () => {
    const sampleData = `CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
619,France,Female,42,2,0.00,1,1,1,101348.88
608,Spain,Female,41,1,83807.86,1,0,1,112542.58
502,France,Female,42,8,159660.80,3,1,0,113931.57
699,France,Female,39,1,0.00,2,0,0,93826.63
850,Spain,Female,43,2,125510.82,1,1,1,79084.10`;
    
    const blob = new Blob([sampleData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_customer_data.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'High': return 'red';
      case 'Medium': return 'yellow';
      case 'Low': return 'green';
      default: return 'gray';
    }
  };

  // Chart data
  const pieData = summary ? {
    values: [summary.predictedChurners, summary.totalCustomers - summary.predictedChurners],
    labels: ['Will Churn', 'Will Stay'],
    type: 'pie' as const,
    marker: {
      colors: ['#E53E3E', '#38A169'],
    },
  } : null;

  const histogramData = {
    x: results.map(r => r.churnProbability),
    type: 'histogram' as const,
    nbinsx: 20,
    marker: {
      color: '#4299E1',
      opacity: 0.7,
    },
  };

  const riskCounts = results.reduce((acc, r) => {
    acc[r.riskLevel] = (acc[r.riskLevel] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const barData = {
    x: Object.keys(riskCounts),
    y: Object.values(riskCounts),
    type: 'bar' as const,
    marker: {
      color: ['#38A169', '#D69E2E', '#E53E3E'],
    },
  };

  return (
    <VStack spacing={8} align="stretch">
      {/* Header */}
      <Box>
        <Text fontSize="3xl" fontWeight="bold" fontFamily="heading" color="primary.500" mb={2}>
          Batch Predictions
        </Text>
        <Text color="secondary.600" fontSize="lg">
          Upload a CSV file to analyze multiple customers at once and get comprehensive insights
        </Text>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert status="error" borderRadius="lg">
          <AlertIcon />
          <AlertTitle>Upload Failed!</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* File Upload Area */}
      {!results.length && (
        <Card bg="white">
          <CardBody>
            <VStack spacing={6}>
              <Box
                border="3px dashed"
                borderColor={dragActive ? 'accent.500' : 'gray.300'}
                borderRadius="xl"
                p={12}
                textAlign="center"
                cursor="pointer"
                transition="all 0.2s"
                _hover={{ borderColor: 'accent.500', bg: 'accent.50' }}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                w="full"
              >
                <VStack spacing={4}>
                  <Box p={4} bg="accent.100" borderRadius="full">
                    <Icon as={Upload} color="accent.500" boxSize={12} />
                  </Box>
                  <VStack spacing={2}>
                    <Text fontSize="xl" fontWeight="bold" color="primary.500">
                      {file ? file.name : 'Drop your CSV file here'}
                    </Text>
                    <Text color="secondary.600">
                      or click to browse files
                    </Text>
                    <Text fontSize="sm" color="secondary.500">
                      Supported format: CSV (max 10MB)
                    </Text>
                  </VStack>
                </VStack>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
              </Box>

              <HStack spacing={4}>
                <Button
                  onClick={handleUpload}
                  isDisabled={!file || loading}
                  isLoading={loading}
                  loadingText="Processing..."
                  size="lg"
                  bg="accent.500"
                  color="white"
                  _hover={{ bg: 'accent.600' }}
                >
                  {loading ? (
                    <HStack>
                      <Spinner size="sm" />
                      <Text>Processing File...</Text>
                    </HStack>
                  ) : (
                    'Upload & Analyze'
                  )}
                </Button>
                
                <Button
                  variant="outline"
                  leftIcon={<Download />}
                  onClick={downloadSampleCSV}
                  size="lg"
                >
                  Download Sample CSV
                </Button>
              </HStack>
            </VStack>
          </CardBody>
        </Card>
      )}

      {/* Results */}
      {summary && results.length > 0 && (
        <>
          {/* Summary Cards */}
          <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }} gap={6}>
            <GridItem>
              <Stat
                bg="white"
                p={6}
                borderRadius="lg"
                boxShadow="sm"
                borderLeft="4px solid"
                borderLeftColor="blue.500"
              >
                <HStack justify="space-between" mb={2}>
                  <StatLabel color="secondary.600" fontSize="sm" fontWeight="medium">
                    Total Customers
                  </StatLabel>
                  <Icon as={Users} color="blue.500" boxSize={5} />
                </HStack>
                <StatNumber fontSize="2xl" fontWeight="bold" color="primary.500">
                  {summary.totalCustomers.toLocaleString()}
                </StatNumber>
                <StatHelpText fontSize="xs" color="secondary.500">
                  Analyzed in this batch
                </StatHelpText>
              </Stat>
            </GridItem>

            <GridItem>
              <Stat
                bg="white"
                p={6}
                borderRadius="lg"
                boxShadow="sm"
                borderLeft="4px solid"
                borderLeftColor="red.500"
              >
                <HStack justify="space-between" mb={2}>
                  <StatLabel color="secondary.600" fontSize="sm" fontWeight="medium">
                    Predicted Churners
                  </StatLabel>
                  <Icon as={TrendingDown} color="red.500" boxSize={5} />
                </HStack>
                <StatNumber fontSize="2xl" fontWeight="bold" color="primary.500">
                  {summary.predictedChurners.toLocaleString()}
                </StatNumber>
                <StatHelpText fontSize="xs" color="secondary.500">
                  High-risk customers
                </StatHelpText>
              </Stat>
            </GridItem>

            <GridItem>
              <Stat
                bg="white"
                p={6}
                borderRadius="lg"
                boxShadow="sm"
                borderLeft="4px solid"
                borderLeftColor="orange.500"
              >
                <HStack justify="space-between" mb={2}>
                  <StatLabel color="secondary.600" fontSize="sm" fontWeight="medium">
                    Churn Rate
                  </StatLabel>
                  <Icon as={Percent} color="orange.500" boxSize={5} />
                </HStack>
                <StatNumber fontSize="2xl" fontWeight="bold" color="primary.500">
                  {(summary.churnRate * 100).toFixed(1)}%
                </StatNumber>
                <StatHelpText fontSize="xs" color="secondary.500">
                  Overall risk percentage
                </StatHelpText>
              </Stat>
            </GridItem>

            <GridItem>
              <Stat
                bg="white"
                p={6}
                borderRadius="lg"
                boxShadow="sm"
                borderLeft="4px solid"
                borderLeftColor="purple.500"
              >
                <HStack justify="space-between" mb={2}>
                  <StatLabel color="secondary.600" fontSize="sm" fontWeight="medium">
                    Avg Probability
                  </StatLabel>
                  <Icon as={Target} color="purple.500" boxSize={5} />
                </HStack>
                <StatNumber fontSize="2xl" fontWeight="bold" color="primary.500">
                  {(summary.averageProbability * 100).toFixed(1)}%
                </StatNumber>
                <StatHelpText fontSize="xs" color="secondary.500">
                  Mean churn probability
                </StatHelpText>
              </Stat>
            </GridItem>
          </Grid>

          {/* Dual Pane: Table and Charts */}
          <Grid templateColumns={{ base: '1fr', xl: '1fr 1fr' }} gap={8}>
            {/* Data Table */}
            <GridItem>
              <Card bg="white">
                <CardBody p={0}>
                  <Box p={4} borderBottom="1px" borderColor="gray.200">
                    <HStack justify="space-between">
                      <Text fontSize="lg" fontWeight="bold" color="primary.500">
                        Customer Predictions
                      </Text>
                      <Button size="sm" leftIcon={<Download />} variant="outline">
                        Export Results
                      </Button>
                    </HStack>
                  </Box>
                  <Box maxH="600px" overflowY="auto">
                    <Table variant="simple" size="sm">
                      <Thead bg="primary.500" position="sticky" top={0}>
                        <Tr>
                          <Th color="white" fontWeight="bold">Customer ID</Th>
                          <Th color="white" fontWeight="bold">Name</Th>
                          <Th color="white" fontWeight="bold">Probability</Th>
                          <Th color="white" fontWeight="bold">Risk</Th>
                          <Th color="white" fontWeight="bold">Prediction</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {results.slice(0, 20).map((result, index) => (
                          <Tr key={result.id} bg={index % 2 === 0 ? 'background.100' : 'white'}>
                            <Td fontWeight="medium" color="primary.500">{result.id}</Td>
                            <Td>{result.customerName}</Td>
                            <Td>
                              <VStack spacing={1} align="start">
                                <Text fontWeight="bold">
                                  {(result.churnProbability * 100).toFixed(1)}%
                                </Text>
                                <Progress
                                  value={result.churnProbability * 100}
                                  size="xs"
                                  colorScheme={result.churnProbability > 0.7 ? 'red' : result.churnProbability > 0.4 ? 'yellow' : 'green'}
                                  w="60px"
                                />
                              </VStack>
                            </Td>
                            <Td>
                              <Badge colorScheme={getRiskColor(result.riskLevel)} variant="subtle">
                                {result.riskLevel}
                              </Badge>
                            </Td>
                            <Td>
                              <Badge
                                colorScheme={result.prediction === 'Will Churn' ? 'red' : 'green'}
                                variant="outline"
                              >
                                {result.prediction}
                              </Badge>
                            </Td>
                          </Tr>
                        ))}
                      </Tbody>
                    </Table>
                  </Box>
                  {results.length > 20 && (
                    <Box p={4} borderTop="1px" borderColor="gray.200" textAlign="center">
                      <Text fontSize="sm" color="secondary.600">
                        Showing 20 of {results.length} results
                      </Text>
                    </Box>
                  )}
                </CardBody>
              </Card>
            </GridItem>

            {/* Charts */}
            <GridItem>
              <VStack spacing={6}>
                {/* Pie Chart */}
                <Card bg="white" w="full">
                  <CardBody>
                    <Text fontSize="lg" fontWeight="bold" color="primary.500" mb={4}>
                      Churn Distribution
                    </Text>
                    {pieData && (
                      <Plot
                        data={[pieData] as PlotData[]}
                        layout={{
                          width: 400,
                          height: 300,
                          margin: { t: 0, b: 0, l: 0, r: 0 },
                          showlegend: true,
                          legend: { orientation: 'h', y: -0.1 },
                        } as Partial<Layout>}
                        config={{ displayModeBar: false }}
                      />
                    )}
                  </CardBody>
                </Card>

                {/* Histogram */}
                <Card bg="white" w="full">
                  <CardBody>
                    <Text fontSize="lg" fontWeight="bold" color="primary.500" mb={4}>
                      Probability Distribution
                    </Text>
                    <Plot
                      data={[histogramData] as PlotData[]}
                      layout={{
                        width: 400,
                        height: 250,
                        margin: { t: 20, b: 40, l: 40, r: 20 },
                        xaxis: { title: 'Churn Probability' },
                        yaxis: { title: 'Count' },
                      } as Partial<Layout>}
                      config={{ displayModeBar: false }}
                    />
                  </CardBody>
                </Card>

                {/* Bar Chart */}
                <Card bg="white" w="full">
                  <CardBody>
                    <Text fontSize="lg" fontWeight="bold" color="primary.500" mb={4}>
                      Risk Level Distribution
                    </Text>
                    <Plot
                      data={[barData] as PlotData[]}
                      layout={{
                        width: 400,
                        height: 250,
                        margin: { t: 20, b: 40, l: 40, r: 20 },
                        xaxis: { title: 'Risk Level' },
                        yaxis: { title: 'Count' },
                      } as Partial<Layout>}
                      config={{ displayModeBar: false }}
                    />
                  </CardBody>
                </Card>
              </VStack>
            </GridItem>
          </Grid>

          {/* New Upload Button */}
          <Box textAlign="center">
            <Button
              onClick={() => {
                setResults([]);
                setSummary(null);
                setFile(null);
                setError(null);
              }}
              variant="outline"
              size="lg"
              leftIcon={<Upload />}
            >
              Upload New File
            </Button>
          </Box>
        </>
      )}
    </VStack>
  );
};

export default BatchPredictions;