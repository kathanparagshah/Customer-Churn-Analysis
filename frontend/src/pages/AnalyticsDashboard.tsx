import React, { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Grid,
  GridItem,
  Card,
  CardBody,
  Select,
  Button,
  ButtonGroup,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Badge,
  Icon,
  Divider,
  Spinner,
  Alert,
  AlertIcon,
} from '@chakra-ui/react';
import {
  Calendar,
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  Map,
  Users,
  Target,
  AlertTriangle,
} from 'lucide-react';
import Plot from 'react-plotly.js';
import apiService from '../services/apiService';

interface TimeRange {
  label: string;
  value: string;
  days: number;
}

interface DailyMetrics {
  date: string;
  churnRate: number;
  predictions: number;
  accuracy: number;
}

interface GeographyData {
  country: string;
  churnRate: number;
  customers: number;
}

const AnalyticsDashboard: React.FC = () => {
  const [selectedTimeRange, setSelectedTimeRange] = useState<string>('30');
  const [selectedMetric, setSelectedMetric] = useState<string>('churnRate');
  const [analyticsData, setAnalyticsData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const timeRanges: TimeRange[] = [
    { label: 'Last 7 Days', value: '7', days: 7 },
    { label: 'Last 30 Days', value: '30', days: 30 },
    { label: 'Last 90 Days', value: '90', days: 90 },
    { label: 'Last 6 Months', value: '180', days: 180 },
    { label: 'Last Year', value: '365', days: 365 },
  ];

  // Fetch analytics data from API
  const fetchAnalyticsData = async (days: number) => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getAnalyticsDashboard(days);
      setAnalyticsData(data);
    } catch (err) {
      console.error('Error fetching analytics data:', err);
      setError('Failed to load analytics data. Using demo data instead.');
      // Fallback to mock data if API fails
      setAnalyticsData(generateMockData(days));
    } finally {
      setLoading(false);
    }
  };

  // Load data on component mount and when time range changes
  useEffect(() => {
    fetchAnalyticsData(parseInt(selectedTimeRange));
  }, [selectedTimeRange]);

  // Handle time range change
  const handleTimeRangeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedTimeRange(event.target.value);
  };

  // Generate mock data as fallback
  const generateMockData = (days: number) => {
    const dailyData: DailyMetrics[] = [];
    const today = new Date();
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      
      dailyData.push({
        date: date.toISOString().split('T')[0],
        churnRate: 0.15 + 0.1 * Math.sin(i / 10) + 0.05 * Math.random(),
        predictions: 50 + Math.floor(30 * Math.random()),
        accuracy: 0.85 + 0.1 * Math.random(),
      });
    }
    
    return {
      daily_metrics: dailyData.map(d => ({
        date: d.date,
        total_predictions: d.predictions,
        total_churners: Math.floor(d.predictions * d.churnRate),
        avg_churn_rate: d.churnRate,
        avg_confidence: d.accuracy,
        high_risk_count: Math.floor(d.predictions * 0.2),
        medium_risk_count: Math.floor(d.predictions * 0.3),
        low_risk_count: Math.floor(d.predictions * 0.5)
      })),
      prediction_trends: {
        daily_data: dailyData.map(d => ({
          date: d.date,
          predictions: d.predictions,
          avg_probability: d.churnRate,
          churners: Math.floor(d.predictions * d.churnRate)
        })),
        overall_stats: {
          total_predictions: dailyData.reduce((sum, d) => sum + d.predictions, 0),
          avg_churn_rate: dailyData.reduce((sum, d) => sum + d.churnRate, 0) / dailyData.length,
          total_churners: dailyData.reduce((sum, d) => sum + Math.floor(d.predictions * d.churnRate), 0),
          avg_confidence: dailyData.reduce((sum, d) => sum + d.accuracy, 0) / dailyData.length
        }
      },
      risk_distribution: {
        High: Math.floor(Math.random() * 100) + 50,
        Medium: Math.floor(Math.random() * 150) + 100,
        Low: Math.floor(Math.random() * 200) + 150
      }
    };
  };

  // Get processed data for charts
  const getChartData = () => {
    if (!analyticsData) return { dailyData: [], overallStats: {}, riskDistribution: {} };
    
    const dailyData = analyticsData.daily_metrics || [];
    const trends = analyticsData.prediction_trends || { daily_data: [], overall_stats: {} };
    const riskDistribution = analyticsData.risk_distribution || { High: 0, Medium: 0, Low: 0 };
    
    return {
      dailyData: dailyData.map((item: any) => ({
        date: item.date,
        churnRate: item.avg_churn_rate || 0,
        predictions: item.total_predictions || 0,
        accuracy: item.avg_confidence || 0,
        churners: item.total_churners || 0
      })),
      overallStats: trends.overall_stats,
      riskDistribution
    };
  };

  const { dailyData, overallStats, riskDistribution } = getChartData();

  // Generate correlation matrix data
  const features = ['Age', 'Balance', 'CreditScore', 'NumOfProducts', 'Tenure'];
  const correlationMatrix = features.map((feature1, i) =>
    features.map((feature2, j) => {
      if (i === j) return 1;
      return Math.random() * 0.8 - 0.4; // Random correlation between -0.4 and 0.4
    })
  );

  // Generate scatter plot data
  const generateScatterData = (xFeature: string, yFeature: string): Array<{x: number, y: number, churn: boolean}> => {
    const n = 200;
    const data: Array<{x: number, y: number, churn: boolean}> = [];
    
    for (let i = 0; i < n; i++) {
      let x: number, y: number;
      
      if (xFeature === 'Age') {
        x = 18 + Math.random() * 62;
      } else if (xFeature === 'Balance') {
        x = Math.random() * 250000;
      } else {
        x = Math.random() * 100;
      }
      
      if (yFeature === 'Churn Probability') {
        y = Math.random();
      } else {
        y = Math.random() * 100;
      }
      
      data.push({ x, y, churn: Math.random() > 0.8 });
    }
    
    return data;
  };

  const ageScatterData = generateScatterData('Age', 'Churn Probability');
  const balanceScatterData = generateScatterData('Balance', 'Churn Probability');

  // Geography data
  const geographyData: GeographyData[] = [
    { country: 'France', churnRate: 0.165, customers: 5014 },
    { country: 'Germany', churnRate: 0.201, customers: 2509 },
    { country: 'Spain', churnRate: 0.143, customers: 2477 },
  ];

  // Calculate summary statistics
  const currentPeriodData = dailyData.slice(-7);
  const previousPeriodData = dailyData.slice(-14, -7);
  
  const currentAvgChurnRate = currentPeriodData.length > 0 
    ? currentPeriodData.reduce((sum, d) => sum + (d.churnRate || 0), 0) / currentPeriodData.length 
    : 0;
  const previousAvgChurnRate = previousPeriodData.length > 0 
    ? previousPeriodData.reduce((sum, d) => sum + (d.churnRate || 0), 0) / previousPeriodData.length 
    : 0;
  const churnRateChange = previousAvgChurnRate > 0 
    ? ((currentAvgChurnRate - previousAvgChurnRate) / previousAvgChurnRate) * 100 
    : 0;
  
  const totalPredictions = dailyData.reduce((sum, d) => sum + (d.predictions || 0), 0);
  const avgAccuracy = dailyData.length > 0 
    ? dailyData.reduce((sum, d) => sum + (d.accuracy || 0), 0) / dailyData.length 
    : 0;
  const totalChurners = overallStats?.total_churners || dailyData.reduce((sum, d) => sum + (d.churners || 0), 0);

  return (
    <VStack spacing={8} align="stretch">
      {/* Header */}
      <Box>
        <Text fontSize="3xl" fontWeight="bold" fontFamily="heading" color="primary.500" mb={2}>
          Analytics Dashboard
        </Text>
        <Text color="secondary.600" fontSize="lg">
          Real-time analytics and trends for customer churn predictions
        </Text>
      </Box>

      {/* Loading State */}
      {loading && (
        <Card bg="white">
          <CardBody>
            <VStack spacing={4}>
              <Spinner size="lg" color="primary.500" />
              <Text>Loading analytics data...</Text>
            </VStack>
          </CardBody>
        </Card>
      )}

      {/* Error State */}
      {error && (
        <Alert status="warning">
          <AlertIcon />
          {error}
        </Alert>
      )}

      {/* Main Content - Only show when not loading */}
      {!loading && (
        <>
          {/* Time Range Picker */}
          <Card bg="white">
        <CardBody>
          <HStack justify="space-between" align="center">
            <HStack>
              <Icon as={Calendar} color="primary.500" boxSize={5} />
              <Text fontSize="lg" fontWeight="bold" color="primary.500">
                Time Period
              </Text>
            </HStack>
            
            <ButtonGroup size="sm" isAttached variant="outline">
              {timeRanges.map((range) => (
                <Button
                  key={range.value}
                  onClick={() => setSelectedTimeRange(range.value)}
                  bg={selectedTimeRange === range.value ? 'primary.500' : 'white'}
                  color={selectedTimeRange === range.value ? 'white' : 'primary.500'}
                  _hover={{
                    bg: selectedTimeRange === range.value ? 'primary.600' : 'primary.50',
                  }}
                >
                  {range.label}
                </Button>
              ))}
            </ButtonGroup>
          </HStack>
        </CardBody>
      </Card>

      {/* Summary Statistics */}
      <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }} gap={6}>
        <GridItem>
          <Stat
            bg="white"
            p={6}
            borderRadius="lg"
            boxShadow="sm"
            borderLeft="4px solid"
            borderLeftColor="red.500"
          >
            <StatLabel color="secondary.600" fontSize="sm">Average Churn Rate</StatLabel>
            <StatNumber fontSize="2xl" color="primary.500">
              {(currentAvgChurnRate * 100).toFixed(1)}%
            </StatNumber>
            <StatHelpText>
              <StatArrow type={churnRateChange > 0 ? 'increase' : 'decrease'} />
              {Math.abs(churnRateChange).toFixed(1)}% vs last period
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
            borderLeftColor="blue.500"
          >
            <StatLabel color="secondary.600" fontSize="sm">Total Predictions</StatLabel>
            <StatNumber fontSize="2xl" color="primary.500">
              {totalPredictions.toLocaleString()}
            </StatNumber>
            <StatHelpText>
              <Icon as={Target} color="green.500" mr={1} />
              Last {selectedTimeRange} days
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
            borderLeftColor="green.500"
          >
            <StatLabel color="secondary.600" fontSize="sm">Model Accuracy</StatLabel>
            <StatNumber fontSize="2xl" color="primary.500">
              {(avgAccuracy * 100).toFixed(1)}%
            </StatNumber>
            <StatHelpText>
              <Icon as={TrendingUp} color="green.500" mr={1} />
              Consistent performance
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
            <StatLabel color="secondary.600" fontSize="sm">Total Churners</StatLabel>
            <StatNumber fontSize="2xl" color="primary.500">
              {totalChurners.toLocaleString()}
            </StatNumber>
            <StatHelpText>
              <Icon as={AlertTriangle} color="orange.500" mr={1} />
              Predicted churners
            </StatHelpText>
          </Stat>
        </GridItem>
      </Grid>

      {/* Main Charts */}
      <Grid templateColumns={{ base: '1fr', xl: '2fr 1fr' }} gap={8}>
        {/* Time Series Chart */}
        <GridItem>
          <Card bg="white">
            <CardBody>
              <VStack spacing={4} align="stretch">
                <HStack justify="space-between">
                  <HStack>
                    <Icon as={TrendingUp} color="primary.500" boxSize={5} />
                    <Text fontSize="lg" fontWeight="bold" color="primary.500">
                      Daily Churn Probability Trend
                    </Text>
                  </HStack>
                  
                  <Select
                    size="sm"
                    width="200px"
                    value={selectedMetric}
                    onChange={(e) => setSelectedMetric(e.target.value)}
                  >
                    <option value="churnRate">Churn Rate</option>
                    <option value="predictions">Predictions Count</option>
                    <option value="accuracy">Model Accuracy</option>
                  </Select>
                </HStack>
                
                <Plot
                  data={[
                    {
                      x: dailyData.map(d => d.date),
                      y: dailyData.map(d => {
                        switch (selectedMetric) {
                          case 'churnRate': return d.churnRate * 100;
                          case 'predictions': return d.predictions;
                          case 'accuracy': return d.accuracy * 100;
                          default: return d.churnRate * 100;
                        }
                      }),
                      type: 'scatter',
                      mode: 'lines+markers',
                      line: {
                        color: '#4299E1',
                        width: 3,
                      },
                      marker: {
                        color: '#4299E1',
                        size: 6,
                      },
                    },
                  ]}
                  layout={{
                    width: 800,
                    height: 400,
                    margin: { t: 20, b: 60, l: 60, r: 20 },
                    xaxis: { title: 'Date' },
                    yaxis: {
                      title: selectedMetric === 'churnRate' ? 'Churn Rate (%)' :
                             selectedMetric === 'predictions' ? 'Number of Predictions' :
                             'Accuracy (%)',
                    },
                  }}
                  config={{ displayModeBar: false }}
                />
              </VStack>
            </CardBody>
          </Card>
        </GridItem>
        
        {/* Geography Distribution */}
        <GridItem>
          <Card bg="white">
            <CardBody>
              <VStack spacing={4} align="stretch">
                <HStack>
                  <Icon as={Map} color="primary.500" boxSize={5} />
                  <Text fontSize="lg" fontWeight="bold" color="primary.500">
                    Churn Rate by Geography
                  </Text>
                </HStack>
                
                <VStack spacing={4} align="stretch">
                  {geographyData.map((geo) => (
                    <Box key={geo.country} p={4} bg="background.100" borderRadius="lg">
                      <HStack justify="space-between" mb={2}>
                        <Text fontWeight="medium">{geo.country}</Text>
                        <Badge
                          colorScheme={geo.churnRate > 0.18 ? 'red' : geo.churnRate > 0.15 ? 'orange' : 'green'}
                        >
                          {(geo.churnRate * 100).toFixed(1)}%
                        </Badge>
                      </HStack>
                      <Text fontSize="sm" color="secondary.600" mb={2}>
                        {geo.customers.toLocaleString()} customers
                      </Text>
                      <Box bg="white" borderRadius="md" p={2}>
                        <Box
                          bg={geo.churnRate > 0.18 ? 'red.500' : geo.churnRate > 0.15 ? 'orange.500' : 'green.500'}
                          height="8px"
                          borderRadius="full"
                          width={`${(geo.churnRate / 0.25) * 100}%`}
                        />
                      </Box>
                    </Box>
                  ))}
                </VStack>
                
                <Plot
                  data={[
                    {
                      values: geographyData.map(g => g.customers),
                      labels: geographyData.map(g => g.country),
                      type: 'pie',
                      marker: {
                        colors: ['#4299E1', '#48BB78', '#ED8936'],
                      },
                    },
                  ]}
                  layout={{
                    width: 350,
                    height: 300,
                    margin: { t: 20, b: 20, l: 20, r: 20 },
                    showlegend: false,
                  }}
                  config={{ displayModeBar: false }}
                />
              </VStack>
            </CardBody>
          </Card>
        </GridItem>
      </Grid>

      {/* Correlation Heatmap */}
      <Card bg="white">
        <CardBody>
          <VStack spacing={4} align="stretch">
            <HStack>
              <Icon as={BarChart3} color="primary.500" boxSize={5} />
              <Text fontSize="lg" fontWeight="bold" color="primary.500">
                Feature Correlation Heatmap
              </Text>
            </HStack>
            
            <Plot
              data={[
                {
                  z: correlationMatrix,
                  x: features,
                  y: features,
                  type: 'heatmap',
                  colorscale: 'RdBu',
                  zmid: 0,
                  showscale: true,
                },
              ]}
              layout={{
                width: 600,
                height: 500,
                margin: { t: 20, b: 80, l: 80, r: 20 },
                xaxis: { title: 'Features' },
                yaxis: { title: 'Features' },
              }}
              config={{ displayModeBar: false }}
            />
          </VStack>
        </CardBody>
      </Card>

      {/* Scatter Plots */}
      <Grid templateColumns={{ base: '1fr', lg: 'repeat(2, 1fr)' }} gap={8}>
        {/* Age vs Probability */}
        <GridItem>
          <Card bg="white">
            <CardBody>
              <VStack spacing={4}>
                <Text fontSize="lg" fontWeight="bold" color="primary.500">
                  Age vs Churn Probability
                </Text>
                
                <Plot
                  data={[
                    {
                      x: ageScatterData.filter(d => !d.churn).map(d => d.x),
                      y: ageScatterData.filter(d => !d.churn).map(d => d.y),
                      mode: 'markers',
                      type: 'scatter',
                      name: 'No Churn',
                      marker: {
                        color: '#48BB78',
                        size: 6,
                        opacity: 0.7,
                      },
                    },
                    {
                      x: ageScatterData.filter(d => d.churn).map(d => d.x),
                      y: ageScatterData.filter(d => d.churn).map(d => d.y),
                      mode: 'markers',
                      type: 'scatter',
                      name: 'Churn',
                      marker: {
                        color: '#F56565',
                        size: 6,
                        opacity: 0.7,
                      },
                    },
                  ]}
                  layout={{
                    width: 500,
                    height: 400,
                    margin: { t: 20, b: 60, l: 60, r: 20 },
                    xaxis: { title: 'Age' },
                    yaxis: { title: 'Churn Probability' },
                    showlegend: true,
                  }}
                  config={{ displayModeBar: false }}
                />
              </VStack>
            </CardBody>
          </Card>
        </GridItem>
        
        {/* Balance vs Probability */}
        <GridItem>
          <Card bg="white">
            <CardBody>
              <VStack spacing={4}>
                <Text fontSize="lg" fontWeight="bold" color="primary.500">
                  Balance vs Churn Probability
                </Text>
                
                <Plot
                  data={[
                    {
                      x: balanceScatterData.filter(d => !d.churn).map(d => d.x),
                      y: balanceScatterData.filter(d => !d.churn).map(d => d.y),
                      mode: 'markers',
                      type: 'scatter',
                      name: 'No Churn',
                      marker: {
                        color: '#48BB78',
                        size: 6,
                        opacity: 0.7,
                      },
                    },
                    {
                      x: balanceScatterData.filter(d => d.churn).map(d => d.x),
                      y: balanceScatterData.filter(d => d.churn).map(d => d.y),
                      mode: 'markers',
                      type: 'scatter',
                      name: 'Churn',
                      marker: {
                        color: '#F56565',
                        size: 6,
                        opacity: 0.7,
                      },
                    },
                  ]}
                  layout={{
                    width: 500,
                    height: 400,
                    margin: { t: 20, b: 60, l: 60, r: 20 },
                    xaxis: { title: 'Balance ($)' },
                    yaxis: { title: 'Churn Probability' },
                    showlegend: true,
                  }}
                  config={{ displayModeBar: false }}
                />
              </VStack>
            </CardBody>
          </Card>
        </GridItem>
      </Grid>
        </>
      )}
    </VStack>
  );
};

export default AnalyticsDashboard;