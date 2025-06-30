import React from 'react';
import {
  Box,
  Flex,
  HStack,
  VStack,
  Text,
  Button,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Avatar,
  Spacer,
  Container,
  useColorModeValue,
  Badge,
  Card,
  CardBody,
  Divider,
} from '@chakra-ui/react';
import { ChevronDownIcon, BarChart3, User, FileText, TrendingUp, Activity, LogOut } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { useNavigate, useLocation } from 'react-router-dom';

interface LayoutProps {
  children: React.ReactNode;
  currentPage?: string;
  showSidebar?: boolean;
  onNavigate?: (page: string) => void;
}

const Header: React.FC<{ currentPage?: string; onNavigate?: (page: string) => void }> = ({ 
  currentPage, 
  onNavigate 
}) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const bg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  const navItems = [
    { label: 'Home', value: 'home', path: '/' },
    { label: 'Single Prediction', value: 'single', path: '/single-prediction' },
    { label: 'Batch', value: 'batch', path: '/batch-predictions' },
    { label: 'Insights', value: 'insights', path: '/model-insights' },
    { label: 'Analytics', value: 'analytics', path: '/analytics-dashboard' },
  ];

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getCurrentPage = () => {
    const currentPath = location.pathname;
    const currentItem = navItems.find(item => item.path === currentPath);
    return currentItem?.value || 'home';
  };

  const activePageValue = getCurrentPage();

  return (
    <Box bg={bg} borderBottom="1px" borderColor={borderColor} position="sticky" top={0} zIndex={1000}>
      <Container maxW="7xl">
        <Flex h={16} alignItems="center" justifyContent="space-between">
          {/* Logo */}
          <HStack spacing={3}>
            <Box p={2} bg="primary.500" borderRadius="lg">
              <BarChart3 color="white" size={24} />
            </Box>
            <VStack spacing={0} align="start">
              <Text fontSize="xl" fontWeight="bold" color="primary.500" fontFamily="heading">
                Customer Churn Prediction
              </Text>
              <Text fontSize="sm" color="secondary.500">
                Bank Analytics Dashboard
              </Text>
            </VStack>
          </HStack>

          {/* Navigation */}
          <HStack spacing={6} display={{ base: 'none', md: 'flex' }}>
            {navItems.map((item) => (
              <Button
                key={item.value}
                variant={activePageValue === item.value ? 'solid' : 'ghost'}
                size="sm"
                onClick={() => handleNavigation(item.path)}
                color={activePageValue === item.value ? 'white' : 'secondary.600'}
                bg={activePageValue === item.value ? 'primary.500' : 'transparent'}
                _hover={{
                  bg: activePageValue === item.value ? 'primary.600' : 'background.200',
                  color: activePageValue === item.value ? 'white' : 'primary.500',
                  transform: 'translateY(-1px)',
                }}
                transition="all 0.2s"
              >
                {item.label}
              </Button>
            ))}
          </HStack>

          {/* User Menu */}
          <Menu>
            <MenuButton 
              as={Button} 
              variant="ghost" 
              rightIcon={<ChevronDownIcon />}
              _hover={{ bg: 'background.200' }}
              transition="all 0.2s"
            >
              <HStack>
                <Avatar 
                  size="sm" 
                  name={user?.name || 'User'} 
                  src={user?.avatar}
                  bg="accent.500" 
                />
                <VStack spacing={0} align="start" display={{ base: 'none', md: 'flex' }}>
                  <Text fontSize="sm" fontWeight="medium">
                    {user?.name || 'User'}
                  </Text>
                  <Text fontSize="xs" color="secondary.500">
                    {user?.email || 'user@example.com'}
                  </Text>
                </VStack>
              </HStack>
            </MenuButton>
            <MenuList>
              <MenuItem icon={<User size={16} />}>Profile</MenuItem>
              <MenuItem icon={<FileText size={16} />}>Reports</MenuItem>
              <Divider />
              <MenuItem icon={<LogOut size={16} />} onClick={handleLogout}>
                Sign Out
              </MenuItem>
            </MenuList>
          </Menu>
        </Flex>
      </Container>
    </Box>
  );
};

const Sidebar: React.FC<{ currentPage?: string }> = ({ currentPage }) => {
  const bg = useColorModeValue('white', 'gray.50');
  
  return (
    <Box w="280px" bg={bg} borderRight="1px" borderColor="gray.200" h="full" p={4}>
      <VStack spacing={4} align="stretch">
        {/* API Status Card */}
        <Card size="sm">
          <CardBody>
            <VStack spacing={3} align="start">
              <HStack>
                <Box w={3} h={3} bg="green.500" borderRadius="full" />
                <Text fontSize="sm" fontWeight="medium">
                  API Status
                </Text>
              </HStack>
              <Text fontSize="xs" color="secondary.600">
                All services operational
              </Text>
              <Badge colorScheme="green" size="sm">
                Healthy
              </Badge>
            </VStack>
          </CardBody>
        </Card>

        {/* Model Info Card */}
        <Card size="sm">
          <CardBody>
            <VStack spacing={3} align="start">
              <HStack>
                <TrendingUp size={16} color="var(--chakra-colors-accent-500)" />
                <Text fontSize="sm" fontWeight="medium">
                  Model Info
                </Text>
              </HStack>
              <VStack spacing={1} align="start">
                <Text fontSize="xs" color="secondary.600">
                  Version: v2.1.0
                </Text>
                <Text fontSize="xs" color="secondary.600">
                  Accuracy: 94.2%
                </Text>
                <Text fontSize="xs" color="secondary.600">
                  Last Updated: Today
                </Text>
              </VStack>
            </VStack>
          </CardBody>
        </Card>

        {/* Quick Links */}
        <Card size="sm">
          <CardBody>
            <VStack spacing={3} align="start">
              <Text fontSize="sm" fontWeight="medium">
                Quick Links
              </Text>
              <VStack spacing={2} align="start" w="full">
                <Button variant="ghost" size="sm" justifyContent="start" w="full">
                  <HStack>
                    <Activity size={14} />
                    <Text fontSize="xs">Real-time Monitor</Text>
                  </HStack>
                </Button>
                <Button variant="ghost" size="sm" justifyContent="start" w="full">
                  <HStack>
                    <FileText size={14} />
                    <Text fontSize="xs">Export Reports</Text>
                  </HStack>
                </Button>
              </VStack>
            </VStack>
          </CardBody>
        </Card>
      </VStack>
    </Box>
  );
};

const Footer: React.FC = () => {
  const bg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box bg={bg} borderTop="1px" borderColor={borderColor} mt={8}>
      <Container maxW="7xl">
        <Flex py={6} justifyContent="space-between" align="center" direction={{ base: 'column', md: 'row' }} gap={4}>
          <HStack spacing={6} fontSize="sm" color="secondary.600">
            <Text>© 2025 Customer Churn Analytics</Text>
            <Text>•</Text>
            <Text>Version 2.1.0</Text>
          </HStack>
          
          <HStack spacing={6} fontSize="sm">
            <Button variant="link" size="sm" color="secondary.600" _hover={{ color: 'primary.500' }}>
              Privacy
            </Button>
            <Text color="secondary.400">•</Text>
            <Button variant="link" size="sm" color="secondary.600" _hover={{ color: 'primary.500' }}>
              Terms
            </Button>
            <Text color="secondary.400">•</Text>
            <Button variant="link" size="sm" color="secondary.600" _hover={{ color: 'primary.500' }}>
              Help
            </Button>
          </HStack>
        </Flex>
      </Container>
    </Box>
  );
};

const Layout: React.FC<LayoutProps> = ({ 
  children, 
  currentPage = 'home', 
  showSidebar = false, 
  onNavigate 
}) => {
  return (
    <Flex direction="column" minH="100vh">
      <Header currentPage={currentPage} onNavigate={onNavigate} />
      
      <Flex flex={1}>
        {showSidebar && <Sidebar currentPage={currentPage} />}
        
        <Box flex={1} overflow="auto">
          <Container maxW="7xl" py={6}>
            {children}
          </Container>
        </Box>
      </Flex>
      
      <Footer />
    </Flex>
  );
};

export default Layout;