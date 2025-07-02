# Contributing to Customer Churn Analysis

Thank you for your interest in contributing to the Customer Churn Analysis project! This guide will help you get started with development and ensure code quality standards.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Node.js 18+ (for frontend development)
- Git

### Development Setup

1. **Clone and setup the repository:**
   ```bash
   git clone <repository-url>
   cd "Customer Churn Analysis"
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Install pre-commit hooks:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. **Run tests to verify setup:**
   ```bash
   pytest src/tests/ -v
   ```

## 📋 Development Standards

### Code Style
- **Python**: Follow PEP 8, enforced by Black (line length: 88)
- **Import sorting**: Use isort with Black profile
- **Type hints**: Required for all new functions and methods
- **Docstrings**: Use Google-style docstrings for all public functions

### Code Quality Tools
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security scanning
- **safety**: Dependency vulnerability checking

### Testing Requirements
- **Minimum coverage**: 80% for new code
- **Test types**: Unit, integration, and API tests required
- **Test naming**: `test_<functionality>_<scenario>`
- **Fixtures**: Use pytest fixtures for reusable test data

## 🔧 Development Workflow

### 1. Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/<name>`: New features
- `bugfix/<name>`: Bug fixes
- `hotfix/<name>`: Critical production fixes

### 2. Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes following these guidelines:**
   - Write tests first (TDD approach recommended)
   - Ensure all tests pass: `pytest src/tests/`
   - Run pre-commit checks: `pre-commit run --all-files`
   - Update documentation if needed

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### 3. Commit Message Format
Use conventional commits format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/modifications
- `chore:` Maintenance tasks

### 4. Pull Request Process

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request with:**
   - Clear title and description
   - Link to related issues
   - Screenshots for UI changes
   - Test coverage report

3. **PR Requirements:**
   - All CI checks must pass
   - At least one code review approval
   - No merge conflicts
   - Updated documentation

## 🏗️ Project Structure

```
├── app/                    # FastAPI application
│   ├── models/            # Pydantic schemas
│   ├── routes/            # API endpoints
│   └── services/          # Business logic
├── src/                   # Core ML pipeline
│   ├── data/             # Data processing
│   ├── features/         # Feature engineering
│   ├── models/           # ML model training
│   └── tests/            # Test suite
├── frontend/             # React frontend
├── deployment/           # Docker and deployment configs
└── docs/                 # Documentation
```

## 🧪 Testing Guidelines

### Test Categories
1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **API Tests**: Test HTTP endpoints
4. **End-to-End Tests**: Test complete workflows

### Writing Tests
```python
def test_function_name_scenario():
    """Test description explaining what is being tested."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result.status == "expected_status"
    assert len(result.data) == expected_count
```

### Running Tests
```bash
# Run all tests
pytest src/tests/ -v

# Run with coverage
pytest src/tests/ --cov=src --cov-report=html

# Run specific test file
pytest src/tests/test_api.py -v

# Run tests matching pattern
pytest -k "test_model" -v
```

### Modern Testing Patterns

**✅ Recommended Approach**: Use dependency injection with proper mocking

```python
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_predict_endpoint_success(client):
    """Test successful prediction with loaded model."""
    # ✅ Correct: Mock the model_manager instance directly
    with patch('app.services.model_manager.model_manager') as mock_manager:
        mock_manager.is_loaded = True
        mock_manager.predict_single.return_value = {
            "churn_probability": 0.75,
            "risk_level": "high"
        }
        
        test_data = {
            "credit_score": 650,
            "geography": "France",
            "gender": "Female",
            "age": 42,
            "tenure": 2,
            "balance": 83807.86,
            "num_of_products": 1,
            "has_cr_card": 1,
            "is_active_member": 1,
            "estimated_salary": 112542.58
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        assert response.json()["churn_probability"] == 0.75

def test_predict_endpoint_model_not_loaded(client):
    """Test prediction endpoint when model is not loaded."""
    with patch('app.services.model_manager.model_manager') as mock_manager:
        mock_manager.is_loaded = False
        
        response = client.post("/predict", json={})
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
```

**❌ Deprecated Patterns**: Avoid global state patching

```python
# ❌ Don't do this - global function patching is deprecated
with patch('deployment.app_legacy.model_loaded', return_value=True):
    # This approach is no longer recommended
    pass

# ❌ Don't do this - patching global flags
with patch('deployment.app_legacy.is_model_loaded', return_value=True):
    # This approach is no longer recommended
    pass
```

### Key Testing Guidelines

1. **Always mock `app.services.model_manager.model_manager`** - This is the single source of truth for model state
2. **Use `Depends(get_model_manager)`** - Follow the dependency injection pattern in new endpoints
3. **Avoid global flags** - Don't patch `model_loaded` or `is_model_loaded()` functions
4. **Test both loaded and unloaded states** - Verify 503 responses when `is_loaded = False`
5. **Use proper fixtures** - Leverage pytest fixtures for consistent test setup
6. **Mock external dependencies** - Always mock database connections, file I/O, and external APIs
7. **Test error conditions** - Include tests for invalid inputs and edge cases

### Legacy Test Scripts

⚠️ **Deprecated**: Legacy test scripts have been moved to `legacy_tests/` directory:
- `legacy_tests/debug_test.py`
- `legacy_tests/isolated_test.py` 
- `legacy_tests/minimal_test.py`

These scripts are kept for reference but should **not** be used for new development. See `legacy_tests/README.md` for migration guidance.

**For new API endpoints**, always follow this pattern:

```python
# In your route definition
from app.services.model_manager import get_model_manager, ModelManager
from fastapi import Depends, HTTPException, status

@app.post("/your-endpoint")
async def your_endpoint(
    data: YourSchema,
    model_manager: ModelManager = Depends(get_model_manager)
):
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    result = model_manager.your_method(data)
    return result
```

## 🔒 Security Guidelines

- Never commit secrets, API keys, or passwords
- Use environment variables for configuration
- Validate all user inputs
- Follow OWASP security practices
- Run security scans with bandit

## 📚 Documentation

### Code Documentation
- Use Google-style docstrings
- Include type hints
- Document complex algorithms
- Add inline comments for non-obvious code

### API Documentation
- FastAPI auto-generates OpenAPI docs
- Add examples to Pydantic models
- Include error response documentation

## 🐛 Debugging

### Common Issues
1. **Import errors**: Check PYTHONPATH and virtual environment
2. **Test failures**: Run tests individually to isolate issues
3. **Docker issues**: Rebuild containers with `docker-compose build --no-cache`

### Debugging Tools
- Use `pdb` for Python debugging
- Check logs in `logs/` directory
- Use Docker logs: `docker-compose logs <service>`

## 📞 Getting Help

- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Request reviews from team members

## 🎯 Performance Guidelines

- Profile code before optimizing
- Use async/await for I/O operations
- Implement caching for expensive operations
- Monitor memory usage in ML pipelines
- Use connection pooling for databases

## 📦 Dependency Management

- Pin exact versions in `requirements.txt`
- Use `requirements-dev.txt` for development dependencies
- Run security checks: `safety check`
- Update dependencies regularly but test thoroughly

---

**Happy coding! 🚀**

For questions or suggestions about this contributing guide, please open an issue or start a discussion.