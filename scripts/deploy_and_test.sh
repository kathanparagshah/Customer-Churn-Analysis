#!/bin/bash

# Deploy and Test Script for Customer Churn Analysis API
# This script handles deployment, testing, and validation of the FastAPI application

set -e  # Exit on any error

# Configuration
APP_NAME="customer-churn-api"
APP_MODULE="app.main:app"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
TEST_TIMEOUT="30"
HEALTH_CHECK_RETRIES="10"
HEALTH_CHECK_DELAY="2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    install     Install dependencies
    test        Run tests
    lint        Run code linting
    dev         Start development server
    prod        Start production server
    docker      Build and run Docker container
    health      Check application health
    deploy      Full deployment (install + test + start)
    clean       Clean up temporary files
    help        Show this help message

Options:
    -h, --host HOST     Host to bind to (default: $DEFAULT_HOST)
    -p, --port PORT     Port to bind to (default: $DEFAULT_PORT)
    -e, --env ENV       Environment file to use (default: .env)
    -t, --timeout SEC   Test timeout in seconds (default: $TEST_TIMEOUT)
    -v, --verbose       Verbose output
    --no-reload         Disable auto-reload in dev mode
    --workers NUM       Number of worker processes for production
    --help              Show this help message

Examples:
    $0 install                          # Install dependencies
    $0 test                            # Run all tests
    $0 dev                             # Start development server
    $0 prod --workers 4                # Start production server with 4 workers
    $0 deploy --host 0.0.0.0 --port 8080  # Deploy on custom host/port
    $0 health                          # Check if service is healthy

EOF
}

# Parse command line arguments
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
ENV_FILE=".env"
VERBOSE=false
NO_RELOAD=false
WORKERS="1"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -e|--env)
            ENV_FILE="$2"
            shift 2
            ;;
        -t|--timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --no-reload)
            NO_RELOAD=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Set verbose mode
if [ "$VERBOSE" = true ]; then
    set -x
fi

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    log_error "app/main.py not found. Please run this script from the project root directory."
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Check if Python is available
    if ! command_exists python3; then
        log_error "Python 3 is required but not installed."
        exit 1
    fi
    
    # Check if pip is available
    if ! command_exists pip3; then
        log_error "pip3 is required but not installed."
        exit 1
    fi
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        log_info "Installing Python dependencies from requirements.txt..."
        pip3 install -r requirements.txt
    else
        log_warning "requirements.txt not found. Installing basic dependencies..."
        pip3 install fastapi uvicorn pytest httpx
    fi
    
    log_success "Dependencies installed successfully"
}

# Function to run tests
run_tests() {
    log_info "Running tests..."
    
    if ! command_exists pytest; then
        log_error "pytest is required but not installed. Run 'install' command first."
        exit 1
    fi
    
    # Run pytest with timeout
    timeout "$TEST_TIMEOUT" pytest tests/ -v --tb=short
    
    if [ $? -eq 0 ]; then
        log_success "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

# Function to run linting
run_lint() {
    log_info "Running code linting..."
    
    # Check if flake8 is available
    if command_exists flake8; then
        log_info "Running flake8..."
        flake8 app/ tests/ --max-line-length=100 --ignore=E203,W503
    else
        log_warning "flake8 not found, skipping Python linting"
    fi
    
    # Check if black is available
    if command_exists black; then
        log_info "Checking code formatting with black..."
        black --check app/ tests/
    else
        log_warning "black not found, skipping code formatting check"
    fi
    
    log_success "Linting completed"
}

# Function to start development server
start_dev_server() {
    log_info "Starting development server on $HOST:$PORT..."
    
    if ! command_exists uvicorn; then
        log_error "uvicorn is required but not installed. Run 'install' command first."
        exit 1
    fi
    
    # Set reload option
    RELOAD_OPTION="--reload"
    if [ "$NO_RELOAD" = true ]; then
        RELOAD_OPTION=""
    fi
    
    # Load environment file if it exists
    if [ -f "$ENV_FILE" ]; then
        log_info "Loading environment from $ENV_FILE"
        export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
    fi
    
    # Start the server
    uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" $RELOAD_OPTION --log-level info
}

# Function to start production server
start_prod_server() {
    log_info "Starting production server on $HOST:$PORT with $WORKERS workers..."
    
    if ! command_exists uvicorn; then
        log_error "uvicorn is required but not installed. Run 'install' command first."
        exit 1
    fi
    
    # Load environment file if it exists
    if [ -f "$ENV_FILE" ]; then
        log_info "Loading environment from $ENV_FILE"
        export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
    fi
    
    # Start the server with multiple workers
    uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" --workers "$WORKERS" --log-level info
}

# Function to check application health
check_health() {
    log_info "Checking application health..."
    
    local url="http://$HOST:$PORT/health"
    local retries=0
    
    while [ $retries -lt $HEALTH_CHECK_RETRIES ]; do
        if command_exists curl; then
            if curl -f -s "$url" > /dev/null; then
                log_success "Application is healthy at $url"
                
                # Get detailed health info
                log_info "Health check details:"
                curl -s "$url" | python3 -m json.tool 2>/dev/null || curl -s "$url"
                return 0
            fi
        elif command_exists wget; then
            if wget -q --spider "$url"; then
                log_success "Application is healthy at $url"
                return 0
            fi
        else
            log_error "Neither curl nor wget is available for health check"
            return 1
        fi
        
        retries=$((retries + 1))
        log_info "Health check attempt $retries/$HEALTH_CHECK_RETRIES failed, retrying in $HEALTH_CHECK_DELAY seconds..."
        sleep "$HEALTH_CHECK_DELAY"
    done
    
    log_error "Application health check failed after $HEALTH_CHECK_RETRIES attempts"
    return 1
}

# Function to build and run Docker container
run_docker() {
    log_info "Building and running Docker container..."
    
    if ! command_exists docker; then
        log_error "Docker is required but not installed."
        exit 1
    fi
    
    # Create Dockerfile if it doesn't exist
    if [ ! -f "Dockerfile" ]; then
        log_info "Creating Dockerfile..."
        cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
    fi
    
    # Build Docker image
    log_info "Building Docker image..."
    docker build -t "$APP_NAME" .
    
    # Run Docker container
    log_info "Running Docker container..."
    docker run -p "$PORT:8000" "$APP_NAME"
}

# Function to clean up
clean_up() {
    log_info "Cleaning up temporary files..."
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove test artifacts
    rm -rf .pytest_cache/ 2>/dev/null || true
    rm -rf htmlcov/ 2>/dev/null || true
    rm -f .coverage 2>/dev/null || true
    
    # Remove temporary databases
    rm -f *.db 2>/dev/null || true
    rm -f test_*.db 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Function to run full deployment
full_deploy() {
    log_info "Starting full deployment..."
    
    install_dependencies
    run_tests
    start_prod_server
}

# Main command handling
case "${COMMAND:-help}" in
    install)
        install_dependencies
        ;;
    test)
        run_tests
        ;;
    lint)
        run_lint
        ;;
    dev)
        start_dev_server
        ;;
    prod)
        start_prod_server
        ;;
    docker)
        run_docker
        ;;
    health)
        check_health
        ;;
    deploy)
        full_deploy
        ;;
    clean)
        clean_up
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: ${COMMAND}"
        show_help
        exit 1
        ;;
esac