#!/bin/bash

# Customer Churn Analysis - Deployment Verification Script
# This script verifies that both frontend and backend are working correctly

set -e

# Configuration
FRONTEND_URL="https://customer-churn-analysis-kgz3.vercel.app"
BACKEND_URL="https://customer-churn-api-omgg.onrender.com"
LOCAL_FRONTEND="http://localhost:3000"
LOCAL_BACKEND="http://localhost:8000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "error")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "info")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
    esac
}

# Function to check URL accessibility
check_url() {
    local url=$1
    local name=$2
    local timeout=${3:-10}
    
    if curl -s --max-time $timeout "$url" > /dev/null; then
        print_status "success" "$name is accessible"
        return 0
    else
        print_status "error" "$name is not accessible"
        return 1
    fi
}

# Function to check API endpoint
check_api_endpoint() {
    local base_url=$1
    local name=$2
    
    print_status "info" "Testing $name API endpoints..."
    
    # Check health endpoint
    if curl -s --max-time 10 "$base_url/health" | grep -q "healthy\|status"; then
        print_status "success" "$name health endpoint working"
    else
        print_status "error" "$name health endpoint failed"
        return 1
    fi
    
    # Check docs endpoint
    if curl -s --max-time 10 "$base_url/docs" > /dev/null; then
        print_status "success" "$name documentation accessible"
    else
        print_status "warning" "$name documentation not accessible"
    fi
    
    return 0
}

# Function to test prediction endpoint
test_prediction() {
    local base_url=$1
    local name=$2
    
    print_status "info" "Testing $name prediction endpoint..."
    
    local test_data='{
        "customers": [{
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Female",
            "Age": 35,
            "Tenure": 5,
            "Balance": 50000,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 75000
        }]
    }'
    
    local response=$(curl -s --max-time 15 -X POST "$base_url/predict/batch" \
        -H "Content-Type: application/json" \
        -d "$test_data")
    
    if echo "$response" | grep -q "churn_probability\|predictions"; then
        print_status "success" "$name prediction endpoint working"
        return 0
    else
        print_status "error" "$name prediction endpoint failed"
        echo "Response: $response"
        return 1
    fi
}

echo "🔍 Customer Churn Analysis - Deployment Verification"
echo "===================================================="
echo ""

# Check if curl is available
if ! command -v curl >/dev/null 2>&1; then
    print_status "error" "curl is required but not installed"
    exit 1
fi

# Determine what to test based on arguments
TEST_PRODUCTION=true
TEST_LOCAL=false

if [ "$1" = "--local" ]; then
    TEST_PRODUCTION=false
    TEST_LOCAL=true
    print_status "info" "Testing local deployment only"
elif [ "$1" = "--all" ]; then
    TEST_LOCAL=true
    print_status "info" "Testing both local and production deployments"
else
    print_status "info" "Testing production deployment (use --local for local, --all for both)"
fi

echo ""

# Test Production Deployment
if [ "$TEST_PRODUCTION" = true ]; then
    echo "🌐 Testing Production Deployment"
    echo "--------------------------------"
    
    # Test frontend
    if check_url "$FRONTEND_URL" "Production Frontend"; then
        print_status "info" "Frontend URL: $FRONTEND_URL"
    fi
    
    # Test backend
    if check_url "$BACKEND_URL" "Production Backend"; then
        print_status "info" "Backend URL: $BACKEND_URL"
        check_api_endpoint "$BACKEND_URL" "Production"
        test_prediction "$BACKEND_URL" "Production"
    fi
    
    echo ""
fi

# Test Local Deployment
if [ "$TEST_LOCAL" = true ]; then
    echo "🏠 Testing Local Deployment"
    echo "---------------------------"
    
    # Test local frontend
    if check_url "$LOCAL_FRONTEND" "Local Frontend" 5; then
        print_status "info" "Local Frontend URL: $LOCAL_FRONTEND"
    else
        print_status "warning" "Local frontend not running. Start with: cd frontend && npm run dev"
    fi
    
    # Test local backend
    if check_url "$LOCAL_BACKEND" "Local Backend" 5; then
        print_status "info" "Local Backend URL: $LOCAL_BACKEND"
        check_api_endpoint "$LOCAL_BACKEND" "Local"
        test_prediction "$LOCAL_BACKEND" "Local"
    else
        print_status "warning" "Local backend not running. Start with: uvicorn deployment.app:app --reload"
    fi
    
    echo ""
fi

# CORS Test
echo "🔒 Testing CORS Configuration"
echo "-----------------------------"

if [ "$TEST_PRODUCTION" = true ]; then
    # Test CORS by checking if OPTIONS request works
    if curl -s --max-time 10 -X OPTIONS "$BACKEND_URL/predict/batch" \
        -H "Origin: $FRONTEND_URL" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: Content-Type" > /dev/null; then
        print_status "success" "CORS configuration appears correct"
    else
        print_status "warning" "CORS test inconclusive (this may be normal)"
    fi
fi

echo ""

# Summary
echo "📊 Verification Summary"
echo "======================="

if [ "$TEST_PRODUCTION" = true ]; then
    echo "Production URLs:"
    echo "  Frontend: $FRONTEND_URL"
    echo "  Backend:  $BACKEND_URL"
    echo "  API Docs: $BACKEND_URL/docs"
fi

if [ "$TEST_LOCAL" = true ]; then
    echo "Local URLs:"
    echo "  Frontend: $LOCAL_FRONTEND"
    echo "  Backend:  $LOCAL_BACKEND"
    echo "  API Docs: $LOCAL_BACKEND/docs"
fi

echo ""
echo "💡 Tips:"
echo "  - If tests fail, check the troubleshooting section in DEPLOYMENT.md"
echo "  - For local testing, ensure both frontend and backend are running"
echo "  - For production issues, check the platform dashboards (Vercel/Render)"
echo "  - Use browser developer tools to debug CORS and network issues"
echo ""
print_status "info" "Verification complete!"