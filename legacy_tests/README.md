# Legacy Test Scripts

⚠️ **DEPRECATED** - These test scripts are legacy and should not be used for new development.

## Purpose

These scripts were created during the transition from global model loading to dependency injection patterns. They are kept for reference but should not be actively maintained.

## Files

- `debug_test.py` - Simple debug script for testing API endpoints
- `isolated_test.py` - Isolated test with comprehensive endpoint testing
- `minimal_test.py` - Minimal test script for basic functionality

## Migration Notes

These scripts have been updated to use the new patching approach:
- ✅ Patch `app.services.model_manager.model_manager` directly
- ✅ Use `deployment.app_legacy` imports for backward compatibility
- ❌ No longer patch global functions like `model_loaded` or `is_model_loaded()`

## Recommended Approach

For new tests, use the modern testing patterns in `src/tests/`:

```python
# Modern approach - use pytest with proper fixtures
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_predict_endpoint(client):
    with patch('app.services.model_manager.model_manager') as mock_manager:
        mock_manager.is_loaded = True
        mock_manager.predict_single.return_value = {...}
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
```

## Deprecation Timeline

- **Current**: Scripts moved to legacy_tests/ directory
- **Future**: These scripts may be removed in a future version
- **Recommendation**: Migrate any custom tests to use the modern patterns in `src/tests/`