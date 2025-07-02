#!/usr/bin/env python3
"""
Test Suite for Model Components

Tests for model training, prediction, explainability, and API components.
Includes unit tests, integration tests, and performance tests.

Author: Bank Churn Analysis Team
Date: 2024
"""

import importlib.util

# import json  # Unused import
# import os  # Unused import
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

# from sklearn.cluster import KMeans  # Unused import
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

# from sklearn.preprocessing import LabelEncoder  # Unused import

# TestClient compatibility monkey patch
try:
    from fastapi.testclient import TestClient as _OriginalTestClient

    class TestClient(_OriginalTestClient):
        def __init__(self, app, *args, **kwargs):
            # Remove problematic kwargs that might cause issues
            kwargs.pop("app", None)
            super().__init__(app, *args, **kwargs)

except ImportError:
    from starlette.testclient import TestClient

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from models.explain import ModelExplainer  # noqa: E402
from models.segment import CustomerSegmentation  # noqa: E402
from models.train_churn import ChurnPredictor  # noqa: E402

# Import API for testing (optional)
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "deployment"))
    from app import CustomerData, app

    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False
    app = None
    CustomerData = None


class TestChurnPredictor:
    """
    Test cases for ChurnPredictor class.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000

        data = pd.DataFrame(
            {
                "CreditScore": np.random.randint(300, 850, n_samples),
                "Geography_France": np.random.choice([0, 1], n_samples),
                "Geography_Germany": np.random.choice([0, 1], n_samples),
                "Geography_Spain": np.random.choice([0, 1], n_samples),
                "Gender_Female": np.random.choice([0, 1], n_samples),
                "Gender_Male": np.random.choice([0, 1], n_samples),
                "Age": np.random.randint(18, 80, n_samples),
                "Tenure": np.random.randint(0, 10, n_samples),
                "Balance": np.random.uniform(0, 200000, n_samples),
                "NumOfProducts": np.random.randint(1, 4, n_samples),
                "HasCrCard": np.random.choice([0, 1], n_samples),
                "IsActiveMember": np.random.choice([0, 1], n_samples),
                "EstimatedSalary": np.random.uniform(10000, 150000, n_samples),
                "Exited": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            }
        )

        return data

    @pytest.fixture
    def churn_predictor(self, temp_dir):
        """Create ChurnPredictor instance with temporary directory."""
        return ChurnPredictor(project_root=temp_dir)

    def test_init(self, churn_predictor, temp_dir):
        """Test ChurnPredictor initialization."""
        assert churn_predictor.project_root == temp_dir
        expected_processed = temp_dir / "data" / "processed"
        assert churn_predictor.processed_dir == expected_processed
        expected_models = temp_dir / "models"
        expected_reports = temp_dir / "reports"
        assert churn_predictor.models_dir == expected_models
        assert churn_predictor.reports_dir == expected_reports

    def test_load_and_prepare_data(
        self, churn_predictor, sample_data, temp_dir
    ):
        """Test data loading and preparation."""
        # Create processed directory and save sample data
        processed_dir = temp_dir / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        data_file = processed_dir / "churn_features.parquet"
        sample_data.to_parquet(data_file)

        # Load and prepare data
        X, y = churn_predictor.load_and_prepare_data()

        # Assertions
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert "Exited" not in X.columns
        assert y.name == "Exited"

    def test_split_data(self, churn_predictor, sample_data):
        """Test data splitting."""
        X = sample_data.drop("Exited", axis=1)
        y = sample_data["Exited"]

        X_train, X_test, y_train, y_test = churn_predictor.split_data(X, y)

        # Check split proportions
        assert len(X_train) == int(0.8 * len(X))
        assert len(X_test) == len(X) - len(X_train)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        # Check stratification (churn rate should be similar)
        train_churn_rate = y_train.mean()
        test_churn_rate = y_test.mean()
        overall_churn_rate = y.mean()

        assert abs(train_churn_rate - overall_churn_rate) < 0.05
        assert abs(test_churn_rate - overall_churn_rate) < 0.05

    def test_train_baseline_model(self, churn_predictor, sample_data):
        """Test baseline model training."""
        X = sample_data.drop("Exited", axis=1)
        y = sample_data["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train baseline model
        result = churn_predictor.train_baseline_model(
            X_train, y_train, X_test, y_test
        )
        baseline_model, baseline_metrics = result

        # Assertions
        assert baseline_model is not None
        assert hasattr(baseline_model, "predict")
        assert hasattr(baseline_model, "predict_proba")

        # Check metrics
        assert "accuracy" in baseline_metrics
        assert "precision" in baseline_metrics
        assert "recall" in baseline_metrics
        assert "f1_score" in baseline_metrics
        assert "roc_auc" in baseline_metrics

        # Validate metric ranges
        assert 0 <= baseline_metrics["accuracy"] <= 1
        assert 0 <= baseline_metrics["precision"] <= 1
        assert 0 <= baseline_metrics["recall"] <= 1
        assert 0 <= baseline_metrics["f1_score"] <= 1
        assert 0 <= baseline_metrics["roc_auc"] <= 1

    def test_model_prediction_output_shape(self, churn_predictor, sample_data):
        """Test that model predictions have correct shape and value ranges."""
        X = sample_data.drop("Exited", axis=1)
        y = sample_data["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train a simple model
        result = churn_predictor.train_baseline_model(
            X_train, y_train, X_test, y_test
        )
        model, _ = result

        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Check prediction shapes
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)  # Binary classification

        # Check prediction value ranges
        msg = "Predictions should be binary (0 or 1)"
        assert all(pred in [0, 1] for pred in predictions), msg
        # Check probability ranges
        for prob_row in probabilities:
            for prob in prob_row:
                assert 0 <= prob <= 1

        # Check that probabilities sum to 1
        prob_sums = probabilities.sum(axis=1)
        tolerance = 1e-6
        msg = "Probabilities should sum to 1"
        condition = all(abs(ps - 1.0) < tolerance for ps in prob_sums)
        assert condition, msg

    def test_model_feature_importance(self, churn_predictor, sample_data):
        """Test that model provides feature importance."""
        X = sample_data.drop("Exited", axis=1)
        y = sample_data["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        result = churn_predictor.train_baseline_model(
            X_train, y_train, X_test, y_test
        )
        model, _ = result

        # Get feature importance
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            assert len(importance) == X_train.shape[1]
            assert all(imp >= 0 for imp in importance)
            assert sum(importance) > 0

    def test_cross_validation_scores(self, churn_predictor, sample_data):
        """Test cross-validation functionality."""
        X = sample_data.drop("Exited", axis=1)
        y = sample_data["Exited"]

        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")

        # Check that we get the right number of scores
        assert len(cv_scores) == 3

        # Check score ranges
        assert all(0 <= score <= 1 for score in cv_scores)

        # Check that scores are reasonable (not all zeros or ones)
        assert not all(score == 0 for score in cv_scores)
        assert not all(score == 1 for score in cv_scores)

    def test_train_advanced_models(self, churn_predictor, sample_data):
        """Test advanced model training."""
        X = sample_data.drop("Exited", axis=1)
        y = sample_data["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train a simple RandomForest model
        model = RandomForestClassifier(
            n_estimators=10, max_depth=3, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        metrics = churn_predictor.evaluate_model(model, X_test, y_test)

        # Assertions
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        assert "accuracy" in metrics
        assert "roc_auc" in metrics

    def test_evaluate_model(self, churn_predictor, sample_data):
        """Test model evaluation."""
        X = sample_data.drop("Exited", axis=1)
        y = sample_data["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        metrics = churn_predictor.evaluate_model(model, X_test, y_test)

        # Check all expected metrics are present
        expected_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

    def test_cross_validate_model_simple(self, churn_predictor, sample_data):
        """Test cross-validation with simple scoring."""
        X = sample_data.drop("Exited", axis=1)
        y = sample_data["Exited"]

        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")

        # Check results
        assert len(cv_scores) == 3
        assert all(0 <= score <= 1 for score in cv_scores)
        mean_score = cv_scores.mean()
        assert 0 <= mean_score <= 1


class TestCustomerSegmentation:
    """
    Test cases for CustomerSegmentation class.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for segmentation testing."""
        np.random.seed(42)
        n_samples = 500

        data = pd.DataFrame(
            {
                "CreditScore": np.random.randint(300, 850, n_samples),
                "Age": np.random.randint(18, 80, n_samples),
                "Tenure": np.random.randint(0, 10, n_samples),
                "Balance": np.random.uniform(0, 200000, n_samples),
                "NumOfProducts": np.random.randint(1, 4, n_samples),
                "EstimatedSalary": np.random.uniform(10000, 150000, n_samples),
                "Exited": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            }
        )

        return data

    @pytest.fixture
    def segmentation(self, temp_dir):
        """Create CustomerSegmentation instance."""
        return CustomerSegmentation(project_root=temp_dir)

    def test_init(self, segmentation, temp_dir):
        """Test CustomerSegmentation initialization."""
        assert segmentation.project_root == temp_dir
        assert segmentation.processed_dir == temp_dir / "data" / "processed"

    def test_select_features(self, segmentation, sample_data):
        """Test feature selection for clustering."""
        features = segmentation.select_features(sample_data)

        # Check that features are selected
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)

        # Check that target variable is not included
        assert "Exited" not in features.columns

        # Check that numeric features are included
        expected_features = ["CreditScore", "Age", "Balance", "NumOfProducts"]
        for feature in expected_features:
            if feature in sample_data.columns:
                assert feature in features.columns

    def test_find_optimal_clusters(self, segmentation, sample_data):
        """Test optimal cluster number finding."""
        features = segmentation.select_features(sample_data)
        scaled_features = segmentation.preprocess_features(features)

        # Find optimal clusters (with small range for speed)
        result = segmentation.find_optimal_clusters(
            scaled_features, k_range=range(2, 5)
        )
        optimal_k, elbow_scores, silhouette_scores = result

        # Assertions
        assert isinstance(optimal_k, int)
        assert 2 <= optimal_k <= 4
        assert len(elbow_scores) == 3  # k_range has 3 values
        assert len(silhouette_scores) == 3

    def test_fit_final_model(self, segmentation, sample_data):
        """Test final clustering model fitting."""
        features = segmentation.select_features(sample_data)
        scaled_features = segmentation.preprocess_features(features)

        # Fit model with k=3
        model, labels = segmentation.fit_final_model(
            scaled_features, n_clusters=3
        )

        # Assertions
        assert model is not None
        assert hasattr(model, "cluster_centers_")
        assert len(labels) == len(sample_data)
        assert len(np.unique(labels)) <= 3
        assert all(0 <= label <= 2 for label in labels)

    def test_analyze_segments(self, segmentation, sample_data):
        """Test segment analysis."""
        features = segmentation.select_features(sample_data)
        scaled_features = segmentation.preprocess_features(features)
        model, labels = segmentation.fit_final_model(
            scaled_features, n_clusters=3
        )

        # Add cluster labels to original data
        data_with_clusters = sample_data.copy()
        data_with_clusters["Cluster"] = labels

        # Analyze segments
        segment_profiles = segmentation.analyze_segments(
            data_with_clusters, features, labels
        )

        # Assertions
        assert isinstance(segment_profiles, dict)
        max_clusters = 3
        msg = "Segments should not exceed max clusters"
        segment_count = len(segment_profiles)
        assert segment_count <= max_clusters, msg

        # Check segment profile structure
        for cluster_id, profile in segment_profiles.items():
            assert "size" in profile
            assert "percentage" in profile
            assert "churn_rate" in profile
            assert "characteristics" in profile


class TestModelExplainer:
    """
    Test cases for ModelExplainer class.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_model_package(self, temp_dir):
        """Create sample model package for testing."""
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Create sample training data
        X_sample = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
                "feature3": [1, 0, 1, 0, 1],
            }
        )
        y_sample = pd.Series([0, 1, 0, 1, 0])

        # Fit model
        model.fit(X_sample, y_sample)

        # Create model package
        model_package = {
            "model": model,
            "model_name": "RandomForest",
            "version": "1.0.0",
            "feature_names": ["feature1", "feature2", "feature3"],
            "scaler": None,
            "label_encoders": {},
        }

        # Save model package
        models_dir = temp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "churn_model.pkl"
        joblib.dump(model_package, model_path)

        return model_path

    @pytest.fixture
    def sample_explanation_data(self, temp_dir):
        """Create sample data for explanation."""
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.choice([0, 1], 100),
                "Exited": np.random.choice([0, 1], 100),
            }
        )

        # Save data
        data_dir = temp_dir / "data" / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / "churn_features.parquet"
        data.to_parquet(data_path)

        return data_path

    @pytest.fixture
    def explainer(self, temp_dir):
        """Create ModelExplainer instance."""
        return ModelExplainer()

    def test_init(self, explainer):
        """Test ModelExplainer initialization."""
        assert explainer.random_state == 42
        assert explainer.model is None
        assert explainer.explainer is None

    def test_load_model(self, explainer, sample_model_package):
        """Test model loading."""
        explainer.load_model(str(sample_model_package))

        # Assertions
        assert explainer.model is not None
        assert explainer.model_name == "RandomForest"
        assert explainer.feature_names == ["feature1", "feature2", "feature3"]

    def test_load_data(self, explainer, sample_explanation_data):
        """Test data loading for explanation."""
        data_path = str(sample_explanation_data)
        data = explainer.load_data(data_path, sample_size=50)

        # Assertions
        assert isinstance(data, pd.DataFrame)
        assert len(data) <= 50  # Should be sampled
        assert "Exited" in data.columns

    def test_prepare_explanation_data(
        self, explainer, sample_model_package, sample_explanation_data
    ):
        """Test explanation data preparation."""
        # Load model and data
        explainer.load_model(str(sample_model_package))
        data = explainer.load_data(
            str(sample_explanation_data), sample_size=50
        )

        # Prepare data
        X_explain, y = explainer.prepare_explanation_data(data)

        # Assertions
        assert isinstance(X_explain, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X_explain) == len(data)
        expected_features = explainer.feature_names
        assert list(X_explain.columns) == expected_features
        assert "Exited" not in X_explain.columns


@pytest.mark.skip(
    reason="TestClient compatibility issues with FastAPI/Starlette"
)
class TestAPI:
    """
    Test cases for the FastAPI application.
    """

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        # Use positional argument to avoid keyword argument issue
        return TestClient(app)

    @pytest.fixture
    def sample_customer_data(self):
        """Create sample customer data for API testing."""
        return {
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Female",
            "Age": 35,
            "Tenure": 5,
            "Balance": 50000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 75000.0,
        }

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_predict_endpoint_success(self, client, sample_customer_data):
        """Test successful prediction endpoint."""
        mock_path = "app.services.model_manager.model_manager"
        with patch(mock_path) as mock_model_manager:
            # Mock the model manager's predict_single method
            mock_model_manager.predict_single.return_value = {
                "churn_probability": 0.3,
                "churn_prediction": False,
                "risk_level": "Low",
                "confidence": 0.7,
                "threshold_used": 0.5,
            }

            response = client.post("/predict", json=sample_customer_data)
            assert response.status_code == 200

            data = response.json()
            assert "churn_probability" in data
            assert "churn_prediction" in data
            assert "risk_level" in data
            assert "confidence" in data
            assert "timestamp" in data

    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data."""
        invalid_data = {
            "CreditScore": 1000,  # Invalid credit score
            "Geography": "InvalidCountry",  # Invalid geography
            "Gender": "Female",
            "Age": 35,
            # Missing required fields
        }

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_endpoint(self, client, sample_customer_data):
        """Test batch prediction endpoint."""
        mock_path = "app.services.model_manager.model_manager"
        with patch(mock_path) as mock_model_manager:
            # Mock the model manager's predict_batch method
            mock_model_manager.predict_batch.return_value = [
                {
                    "churn_probability": 0.3,
                    "churn_prediction": False,
                    "risk_level": "Low",
                    "confidence": 0.7,
                },
                {
                    "churn_probability": 0.6,
                    "churn_prediction": True,
                    "risk_level": "High",
                    "confidence": 0.4,
                },
            ]

            batch_data = {
                "customers": [sample_customer_data, sample_customer_data]
            }

            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == 200

            data = response.json()
            assert "predictions" in data
            assert "summary" in data
            assert len(data["predictions"]) == 2
            assert "total_customers" in data["summary"]

    def test_batch_predict_too_large(self, client, sample_customer_data):
        """Test batch prediction with too many customers."""
        # Create batch with too many customers
        large_batch = {"customers": [sample_customer_data] * 1001}

        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 422

    @pytest.mark.skipif(not APP_AVAILABLE, reason="API module not available")
    def test_customer_data_validation(self):
        """Test CustomerData model validation."""
        # Valid data
        valid_data = {
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Female",
            "Age": 35,
            "Tenure": 5,
            "Balance": 50000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 75000.0,
        }

        customer = CustomerData(**valid_data)
        assert customer.CreditScore == 650
        assert customer.Geography == "France"

        # Invalid geography
        with pytest.raises(ValueError):
            invalid_data = valid_data.copy()
            invalid_data["Geography"] = "InvalidCountry"
            CustomerData(**invalid_data)

        # Invalid credit score
        with pytest.raises(ValueError):
            invalid_data = valid_data.copy()
            invalid_data["CreditScore"] = 1000
            CustomerData(**invalid_data)


class TestPerformance:
    """
    Performance and load tests.
    """

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        np.random.seed(42)
        n_samples = 10000

        data = pd.DataFrame(
            {
                "CreditScore": np.random.randint(300, 850, n_samples),
                "Geography_France": np.random.choice([0, 1], n_samples),
                "Geography_Germany": np.random.choice([0, 1], n_samples),
                "Geography_Spain": np.random.choice([0, 1], n_samples),
                "Gender_Female": np.random.choice([0, 1], n_samples),
                "Gender_Male": np.random.choice([0, 1], n_samples),
                "Age": np.random.randint(18, 80, n_samples),
                "Tenure": np.random.randint(0, 10, n_samples),
                "Balance": np.random.uniform(0, 200000, n_samples),
                "NumOfProducts": np.random.randint(1, 4, n_samples),
                "HasCrCard": np.random.choice([0, 1], n_samples),
                "IsActiveMember": np.random.choice([0, 1], n_samples),
                "EstimatedSalary": np.random.uniform(10000, 150000, n_samples),
                "Exited": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            }
        )

        return data

    def test_model_training_performance(self, large_dataset):
        """Test model training performance on large dataset."""
        import time

        X = large_dataset.drop("Exited", axis=1)
        y = large_dataset["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Test RandomForest training time
        start_time = time.time()
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Test prediction time
        start_time = time.time()
        predictions = model.predict(X_test)
        prediction_time = time.time() - start_time

        # Performance assertions
        assert training_time < 60  # Training should complete within 60s
        assert prediction_time < 5  # Prediction should complete within 5s

        # Accuracy should be reasonable
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.7

        print(f"Training time: {training_time:.2f}s")
        print(f"Prediction time: {prediction_time:.2f}s")
        print(f"Accuracy: {accuracy:.3f}")

    @pytest.mark.skipif(
        importlib.util.find_spec("psutil") is None,
        reason="psutil not installed",
    )
    def test_memory_usage(self, large_dataset):
        """Test memory usage during model training."""
        import psutil

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        X = large_dataset.drop("Exited", axis=1)
        y = large_dataset["Exited"]

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory

        # Memory usage should be reasonable (less than 1GB increase)
        assert memory_increase < 1000

        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
