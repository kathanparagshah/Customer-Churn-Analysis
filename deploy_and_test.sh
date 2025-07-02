#!/bin/bash

# deploy_and_test.sh - Complete deployment and testing script
# Exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Variables
BACKEND_URL="https://customer-churn-api-omgg.onrender.com"
RENDER_SERVICE="customer-churn-api-omgg"
FRONTEND_DIR="frontend"

# Summary variables
DEPLOYMENT_SUMMARY=""
TEST_RESULTS=""

echo "=== Customer Churn Analysis - Deploy and Test Script ==="
echo "Starting deployment process..."
echo ""

# Step 1: Git operations
print_step "1. Committing and pushing changes to Git"
echo "Staging all changes..."
git add .

echo "Committing changes..."
git commit -m "🔧 update API URLs & stub /auth/google" || {
    print_warning "No changes to commit or commit failed"
}

echo "Pushing to origin main..."
git push origin main
print_success "Git operations completed"
DEPLOYMENT_SUMMARY+="✅ Git: Changes committed and pushed to main branch\n"
echo ""

# Step 2: Backend deployment on Render
print_step "2. Triggering backend redeploy on Render"

# Check if Render CLI is available
if command -v render &> /dev/null; then
    echo "Render CLI found, triggering manual deploy..."
    render deploy --service "$RENDER_SERVICE"
    print_success "Render deployment triggered via CLI"
    DEPLOYMENT_SUMMARY+="✅ Backend: Deployed via Render CLI\n"
else
    print_warning "Render CLI not found, assuming auto-deploy on push"
    DEPLOYMENT_SUMMARY+="⚠️  Backend: Auto-deploy triggered by Git push\n"
fi
echo ""

# Step 3: Wait and verify auth endpoint
print_step "3. Waiting 10 seconds before testing auth endpoint"
sleep 10

echo "Testing stubbed auth endpoint..."
AUTH_RESPONSE=$(curl -i -s -X POST "$BACKEND_URL/auth/google" \
    -H "Content-Type: application/json" \
    -d '{}' || echo "CURL_FAILED")

if [[ "$AUTH_RESPONSE" == "CURL_FAILED" ]]; then
    print_error "Auth endpoint test failed - curl command failed"
    TEST_RESULTS+="❌ Auth endpoint: Connection failed\n"
else
    AUTH_STATUS=$(echo "$AUTH_RESPONSE" | head -n1 | grep -o 'HTTP/[0-9.]* [0-9]*' | awk '{print $2}')
    print_success "Auth endpoint responded with status: $AUTH_STATUS"
    TEST_RESULTS+="✅ Auth endpoint: HTTP $AUTH_STATUS\n"
fi
echo ""

# Step 4: Frontend deployment on Vercel
print_step "4. Triggering frontend redeploy on Vercel"

if [ ! -d "$FRONTEND_DIR" ]; then
    print_error "Frontend directory not found: $FRONTEND_DIR"
    exit 1
fi

cd "$FRONTEND_DIR"

# Check if Vercel CLI is available
if command -v vercel &> /dev/null; then
    echo "Vercel CLI found, deploying to production..."
    vercel --prod --yes
    print_success "Frontend deployed to Vercel"
    DEPLOYMENT_SUMMARY+="✅ Frontend: Deployed via Vercel CLI\n"
else
    print_error "Vercel CLI not found. Please install it with: npm i -g vercel"
    DEPLOYMENT_SUMMARY+="❌ Frontend: Vercel CLI not available\n"
fi

cd ..
echo ""

# Step 5: Wait and verify backend endpoints
print_step "5. Waiting 10 seconds before testing backend endpoints"
sleep 10

# Test health endpoint
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s "$BACKEND_URL/health" || echo "CURL_FAILED")

if [[ "$HEALTH_RESPONSE" == "CURL_FAILED" ]]; then
    print_error "Health endpoint test failed - curl command failed"
    TEST_RESULTS+="❌ Health endpoint: Connection failed\n"
else
    print_success "Health endpoint response: $HEALTH_RESPONSE"
    TEST_RESULTS+="✅ Health endpoint: Response received\n"
fi

# Test prediction endpoint with sample data
echo "Testing prediction endpoint..."
PREDICTION_PAYLOAD='{
    "SeniorCitizen": 0,
    "MonthlyCharges": 70.0,
    "TotalCharges": 1400.0,
    "gender_Male": 1,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "PhoneService_Yes": 1,
    "MultipleLines_Yes": 0,
    "InternetService_Fiber optic": 1,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaperlessBilling_Yes": 1,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0
}'

PREDICTION_RESPONSE=$(curl -s -X POST "$BACKEND_URL/predict" \
    -H "Content-Type: application/json" \
    -d "$PREDICTION_PAYLOAD" || echo "CURL_FAILED")

if [[ "$PREDICTION_RESPONSE" == "CURL_FAILED" ]]; then
    print_error "Prediction endpoint test failed - curl command failed"
    TEST_RESULTS+="❌ Prediction endpoint: Connection failed\n"
else
    print_success "Prediction endpoint response: $PREDICTION_RESPONSE"
    TEST_RESULTS+="✅ Prediction endpoint: Response received\n"
fi
echo ""

# Step 6: Print summary
print_step "6. Deployment and Testing Summary"
echo "==========================================="
echo ""
echo -e "${BLUE}DEPLOYMENT RESULTS:${NC}"
echo -e "$DEPLOYMENT_SUMMARY"
echo ""
echo -e "${BLUE}ENDPOINT TEST RESULTS:${NC}"
echo -e "$TEST_RESULTS"
echo ""

# Check if any tests failed
if echo "$TEST_RESULTS" | grep -q "❌"; then
    print_error "Some tests failed. Please check the results above."
    exit 1
else
    print_success "All tests passed successfully!"
fi

echo "==========================================="
echo "Deployment and testing completed!"
echo ""