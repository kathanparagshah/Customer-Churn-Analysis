#!/usr/bin/env python3
"""
Test Suite for Model Components

Comprehensive tests for model training, prediction, explainability, and API components.
Includes unit tests, integration tests, and performance validation tests.

Author: Bank Churn Analysis Team
Date: 2024
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import sys
import os
import importlib.util
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import KMeans

from fastapi.testclient import TestClient
import warnings
warnings.filterwarnings('ignore')

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.train_churn import ChurnPredictor
from models.segment import CustomerSegmentation
from models.explain import ModelExplainer

# Import API for testing (optional)
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / 'deployment'))
    from app import app, CustomerData
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
        
        data = pd.DataFrame({
            'CreditScore': np.random.randint(300, 850, n_samples),
            'Geography_France': np.random.choice([0, 1], n_samples),
            'Geography_Germany': np.random.choice([0, 1], n_samples),
            'Geography_Spain': np.random.choice([0, 1], n_samples),
            'Gender_Female': np.random.choice([0, 1], n_samples),
            'Gender_Male': np.random.choice([0, 1], n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(0, 10, n_samples),
            'Balance': np.random.uniform(0, 200000, n_samples),
            'NumOfProducts': np.random.randint(1, 4, n_samples),
            'HasCrCard': np.random.choice([0, 1], n_samples),
            'IsActiveMember': np.random.choice([0, 1], n_samples),
            'EstimatedSalary': np.random.uniform(10000, 150000, n_samples),
            'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% churn rate
        })
        
        return data
    
    @pytest.fixture
    def churn_predictor(self, temp_dir):
        """Create ChurnPredictor instance with temporary directory."""
        return ChurnPredictor(project_root=temp_dir)
    
    def test_init(self, churn_predictor, temp_dir):
        """Test ChurnPredictor initialization."""
        assert churn_predictor.project_root == temp_dir
        assert churn_predictor.processed_dir == temp_dir / 'data' / 'processed'
        assert churn_predictor.models_dir == temp_dir / 'models'
        assert churn_predictor.reports_dir == temp_dir / 'reports'
    
    def test_load_and_prepare_data(self, churn_predictor, sample_data, temp_dir):
        """Test data loading and preparation."""
        # Create processed directory and save sample data
        processed_dir = temp_dir / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = processed_dir / 'churn_features.parquet'
        sample_data.to_parquet(data_file)
        
        # Load and prepare data
        X, y = churn_predictor.load_and_prepare_data()
        
        # Assertions
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert 'Exited' not in X.columns
        assert y.name == 'Exited'
    
    def test_split_data(self, churn_predictor, sample_data):
        """Test data splitting."""
        X = sample_data.drop('Exited', axis=1)
        y = sample_data['Exited']
        
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
        X = sample_data.drop('Exited', axis=1)
        y = sample_data['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train baseline model
        baseline_model, baseline_metrics = churn_predictor.train_baseline_model(X_train, y_train, X_test, y_test)
        
        # Assertions
        assert baseline_model is not None
        assert hasattr(baseline_model, 'predict')
        assert hasattr(baseline_model, 'predict_proba')
        
        # Check metrics
        assert 'accuracy' in baseline_metrics
        assert 'precision' in baseline_metrics
        assert 'recall' in baseline_metrics
        assert 'f1_score' in baseline_metrics
        assert 'roc_auc' in baseline_metrics
        
        # Validate metric ranges
        assert 0 <= baseline_metrics['accuracy'] <= 1
        assert 0 <= baseline_metrics['precision'] <= 1
        assert 0 <= baseline_metrics['recall'] <= 1
        assert 0 <= baseline_metrics['f1_score'] <= 1
        assert 0 <= baseline_metrics['roc_auc'] <= 1
    
    def test_model_prediction_output_shape(self, churn_predictor, sample_data):
        """Test that model predictions have correct shape and value ranges."""
        X = sample_data.drop('Exited', axis=1)
        y = sample_data['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a simple model
        model, _ = churn_predictor.train_baseline_model(X_train, y_train, X_test, y_test)
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Check prediction shapes
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)  # Binary classification
        
        # Check prediction value ranges
        assert all(pred in [0, 1] for pred in predictions)  # Binary predictions
        assert all(0 <= prob <= 1 for prob_row in probabilities for prob in prob_row)  # Probabilities
        
        # Check that probabilities sum to 1
        prob_sums = probabilities.sum(axis=1)
        assert all(abs(prob_sum - 1.0) < 1e-6 for prob_sum in prob_sums)
    
    def test_model_feature_importance(self, churn_predictor, sample_data):
        """Test that model provides feature importance."""
        X = sample_data.drop('Exited', axis=1)
        y = sample_data['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model, _ = churn_predictor.train_baseline_model(X_train, y_train, X_test, y_test)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            assert len(importance) == X_train.shape[1]
            assert all(imp >= 0 for imp in importance)  # Non-negative importance
            assert sum(importance) > 0  # At least some features should be important
    
    def test_cross_validation_scores(self, churn_predictor, sample_data):
        """Test cross-validation functionality."""
        X = sample_data.drop('Exited', axis=1)
        y = sample_data['Exited']
        
        # Perform cross-validation
        cv_scores = churn_predictor.cross_validate_model(X, y, cv_folds=3)
        
        # Check that we get the right number of scores
        assert len(cv_scores) == 3
        
        # Check score ranges
        assert all(0 <= score <= 1 for score in cv_scores)
        
        # Check that scores are reasonable (not all zeros or ones)
        assert not all(score == 0 for score in cv_scores)
        assert not all(score == 1 for score in cv_scores)


class TestCustomerSegmentation:
    """Test cases for CustomerSegmentation model."""
    
    @pytest.fixture
    def segmentation_model(self):
        """Create a CustomerSegmentation instance."""
        return CustomerSegmentation()
    
    @pytest.fixture
    def sample_segmentation_data(self):
        """Create sample data for segmentation testing."""
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'CreditScore': np.random.normal(650, 100, n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(0, 11, n_samples),
            'Balance': np.random.exponential(50000, n_samples),
            'NumOfProducts': np.random.randint(1, 5, n_samples),
            'EstimatedSalary': np.random.normal(100000, 50000, n_samples),
            'Geography_Germany': np.random.choice([0, 1], n_samples),
            'Geography_Spain': np.random.choice([0, 1], n_samples),
            'Gender_Male': np.random.choice([0, 1], n_samples),
            'HasCrCard': np.random.choice([0, 1], n_samples),
            'IsActiveMember': np.random.choice([0, 1], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_clustering_initialization(self, segmentation_model):
        """Test CustomerSegmentation initialization."""
        assert segmentation_model.n_clusters is None
        assert segmentation_model.model is None
        assert segmentation_model.scaler is None
        assert segmentation_model.feature_cols is None
    
    def test_optimal_clusters_selection(self, segmentation_model, sample_segmentation_data):
        """Test optimal number of clusters selection."""
        optimal_k = segmentation_model.find_optimal_clusters(
            sample_segmentation_data, 
            max_clusters=5
        )
        
        # Check that optimal k is reasonable
        assert 2 <= optimal_k <= 5
        assert isinstance(optimal_k, int)
    
    def test_clustering_labels_validity(self, segmentation_model, sample_segmentation_data):
        """Test that clustering produces valid labels."""
        # Train the model
        segmentation_model.train(sample_segmentation_data, n_clusters=3)
        
        # Get cluster labels
        labels = segmentation_model.predict(sample_segmentation_data)
        
        # Check label properties
        assert len(labels) == len(sample_segmentation_data)
        assert all(isinstance(label, (int, np.integer)) for label in labels)
        assert min(labels) >= 0
        assert max(labels) < 3  # Should be 0, 1, 2 for 3 clusters
        
        # Check that all clusters are represented
        unique_labels = set(labels)
        assert len(unique_labels) <= 3
        assert len(unique_labels) >= 1  # At least one cluster should exist
    
    def test_cluster_centers_shape(self, segmentation_model, sample_segmentation_data):
        """Test that cluster centers have correct shape."""
        n_clusters = 4
        segmentation_model.train(sample_segmentation_data, n_clusters=n_clusters)
        
        # Check cluster centers
        centers = segmentation_model.model.cluster_centers_
        expected_features = len(segmentation_model.feature_cols)
        
        assert centers.shape == (n_clusters, expected_features)
        assert not np.isnan(centers).any()  # No NaN values
        assert not np.isinf(centers).any()  # No infinite values
    
    def test_clustering_reproducibility(self, sample_segmentation_data):
        """Test that clustering results are reproducible with same random state."""
        # Train two models with same random state
        model1 = CustomerSegmentation()
        model2 = CustomerSegmentation()
        
        model1.train(sample_segmentation_data, n_clusters=3, random_state=42)
        model2.train(sample_segmentation_data, n_clusters=3, random_state=42)
        
        labels1 = model1.predict(sample_segmentation_data)
        labels2 = model2.predict(sample_segmentation_data)
        
        # Labels should be identical
        assert np.array_equal(labels1, labels2)
    
    def test_cluster_analysis_output(self, segmentation_model, sample_segmentation_data):
        """Test cluster analysis output format and content."""
        segmentation_model.train(sample_segmentation_data, n_clusters=3)
        analysis = segmentation_model.analyze_segments(sample_segmentation_data)
        
        # Check analysis structure
        assert isinstance(analysis, dict)
        assert 'cluster_sizes' in analysis
        assert 'cluster_profiles' in analysis
        
        # Check cluster sizes
        cluster_sizes = analysis['cluster_sizes']
        assert len(cluster_sizes) == 3
        assert sum(cluster_sizes.values()) == len(sample_segmentation_data)
        assert all(size > 0 for size in cluster_sizes.values())
        
        # Check cluster profiles
        profiles = analysis['cluster_profiles']
        assert len(profiles) == 3
        for cluster_id, profile in profiles.items():
            assert isinstance(profile, dict)
            assert len(profile) > 0  # Should have some features
    
    def test_feature_scaling_consistency(self, segmentation_model, sample_segmentation_data):
        """Test that feature scaling is applied consistently."""
        segmentation_model.train(sample_segmentation_data, n_clusters=3)
        
        # Get scaled features
        scaled_features = segmentation_model.scaler.transform(
            sample_segmentation_data[segmentation_model.feature_cols]
        )
        
        # Check scaling properties
        feature_means = np.mean(scaled_features, axis=0)
        feature_stds = np.std(scaled_features, axis=0)
        
        # Should be approximately standardized (mean~0, std~1)
        assert all(abs(mean) < 1e-10 for mean in feature_means)  # Mean close to 0
        assert all(abs(std - 1.0) < 1e-10 for std in feature_stds)  # Std close to 1
    
    def test_empty_data_handling(self, segmentation_model):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, AttributeError)):
            segmentation_model.train(empty_data, n_clusters=3)
    
    def test_single_cluster_edge_case(self, segmentation_model, sample_segmentation_data):
        """Test edge case with single cluster."""
        segmentation_model.train(sample_segmentation_data, n_clusters=1)
        labels = segmentation_model.predict(sample_segmentation_data)
        
        # All labels should be 0
        assert all(label == 0 for label in labels)
        assert len(set(labels)) == 1
    
    def test_train_advanced_models(self, churn_predictor, sample_data):
        """Test advanced model training."""
        X = sample_data.drop('Exited', axis=1)
        y = sample_data['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train advanced models (with reduced parameter grid for speed)
        models, results = churn_predictor.train_advanced_models(
            X_train, y_train, X_test, y_test,
            param_grids={
                'RandomForest': {
                    'n_estimators': [10, 50],
                    'max_depth': [3, 5]
                }
            }
        )
        
        # Assertions
        assert 'RandomForest' in models
        assert 'RandomForest' in results
        
        model = models['RandomForest']
        metrics = results['RandomForest']
        
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
    
    def test_evaluate_model(self, churn_predictor, sample_data):
        """Test model evaluation."""
        X = sample_data.drop('Exited', axis=1)
        y = sample_data['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = churn_predictor.evaluate_model(model, X_test, y_test)
        
        # Check all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_cross_validate_model(self, churn_predictor, sample_data):
        """Test cross-validation."""
        X = sample_data.drop('Exited', axis=1)
        y = sample_data['Exited']
        
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Perform cross-validation
        cv_results = churn_predictor.cross_validate_model(model, X, y, cv=3)
        
        # Check results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert 'scores' in cv_results
        assert len(cv_results['scores']) == 3
        assert 0 <= cv_results['mean_score'] <= 1


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
        
        data = pd.DataFrame({
            'CreditScore': np.random.randint(300, 850, n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(0, 10, n_samples),
            'Balance': np.random.uniform(0, 200000, n_samples),
            'NumOfProducts': np.random.randint(1, 4, n_samples),
            'EstimatedSalary': np.random.uniform(10000, 150000, n_samples),
            'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        return data
    
    @pytest.fixture
    def segmentation(self, temp_dir):
        """Create CustomerSegmentation instance."""
        return CustomerSegmentation(project_root=temp_dir)
    
    def test_init(self, segmentation, temp_dir):
        """Test CustomerSegmentation initialization."""
        assert segmentation.project_root == temp_dir
        assert segmentation.processed_dir == temp_dir / 'data' / 'processed'
    
    def test_select_features(self, segmentation, sample_data):
        """Test feature selection for clustering."""
        features = segmentation.select_features(sample_data)
        
        # Check that features are selected
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        
        # Check that target variable is not included
        assert 'Exited' not in features.columns
        
        # Check that numeric features are included
        expected_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts']
        for feature in expected_features:
            if feature in sample_data.columns:
                assert feature in features.columns
    
    def test_find_optimal_clusters(self, segmentation, sample_data):
        """Test optimal cluster number finding."""
        features = segmentation.select_features(sample_data)
        scaled_features = segmentation.preprocess_features(features)
        
        # Find optimal clusters (with small range for speed)
        optimal_k, elbow_scores, silhouette_scores = segmentation.find_optimal_clusters(
            scaled_features, k_range=range(2, 5)
        )
        
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
        model, labels = segmentation.fit_final_model(scaled_features, n_clusters=3)
        
        # Assertions
        assert model is not None
        assert hasattr(model, 'cluster_centers_')
        assert len(labels) == len(sample_data)
        assert len(np.unique(labels)) <= 3  # Should have at most 3 clusters
        assert all(0 <= label <= 2 for label in labels)  # Labels should be 0, 1, 2
    
    def test_analyze_segments(self, segmentation, sample_data):
        """Test segment analysis."""
        features = segmentation.select_features(sample_data)
        scaled_features = segmentation.preprocess_features(features)
        model, labels = segmentation.fit_final_model(scaled_features, n_clusters=3)
        
        # Add cluster labels to original data
        data_with_clusters = sample_data.copy()
        data_with_clusters['Cluster'] = labels
        
        # Analyze segments
        segment_profiles = segmentation.analyze_segments(data_with_clusters, features, labels)
        
        # Assertions
        assert isinstance(segment_profiles, dict)
        assert len(segment_profiles) <= 3  # Should have at most 3 segments
        
        # Check segment profile structure
        for cluster_id, profile in segment_profiles.items():
            assert 'size' in profile
            assert 'percentage' in profile
            assert 'churn_rate' in profile
            assert 'characteristics' in profile


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
        X_sample = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [1, 0, 1, 0, 1]
        })
        y_sample = pd.Series([0, 1, 0, 1, 0])
        
        # Fit model
        model.fit(X_sample, y_sample)
        
        # Create model package
        model_package = {
            'model': model,
            'model_name': 'RandomForest',
            'version': '1.0.0',
            'feature_names': ['feature1', 'feature2', 'feature3'],
            'scaler': None,
            'label_encoders': {}
        }
        
        # Save model package
        models_dir = temp_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / 'churn_model.pkl'
        joblib.dump(model_package, model_path)
        
        return model_path
    
    @pytest.fixture
    def sample_explanation_data(self, temp_dir):
        """Create sample data for explanation."""
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.choice([0, 1], 100),
            'Exited': np.random.choice([0, 1], 100)
        })
        
        # Save data
        data_dir = temp_dir / 'data' / 'processed'
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / 'churn_features.parquet'
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
        assert explainer.model_name == 'RandomForest'
        assert explainer.feature_names == ['feature1', 'feature2', 'feature3']
    
    def test_load_data(self, explainer, sample_explanation_data):
        """Test data loading for explanation."""
        data = explainer.load_data(str(sample_explanation_data), sample_size=50)
        
        # Assertions
        assert isinstance(data, pd.DataFrame)
        assert len(data) <= 50  # Should be sampled
        assert 'Exited' in data.columns
    
    def test_prepare_explanation_data(self, explainer, sample_model_package, sample_explanation_data):
        """Test explanation data preparation."""
        # Load model and data
        explainer.load_model(str(sample_model_package))
        data = explainer.load_data(str(sample_explanation_data), sample_size=50)
        
        # Prepare data
        X_explain, y = explainer.prepare_explanation_data(data)
        
        # Assertions
        assert isinstance(X_explain, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X_explain) == len(data)
        assert list(X_explain.columns) == explainer.feature_names
        assert 'Exited' not in X_explain.columns


@pytest.mark.skipif(not APP_AVAILABLE, reason="API module not available")
class TestAPI:
    """
    Test cases for the FastAPI application.
    """
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
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
            "EstimatedSalary": 75000.0
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
    
    @patch('deployment.app.model_loaded', True)
    @patch('deployment.app.model')
    @patch('deployment.app.scaler')
    def test_predict_endpoint_success(self, mock_scaler, mock_model, client, sample_customer_data):
        """Test successful prediction endpoint."""
        # Mock model predictions
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        mock_model.predict.return_value = np.array([0])
        mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        
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
            "Age": 35
            # Missing required fields
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    @patch('deployment.app.model_loaded', True)
    @patch('deployment.app.model')
    @patch('deployment.app.scaler')
    def test_batch_predict_endpoint(self, mock_scaler, mock_model, client, sample_customer_data):
        """Test batch prediction endpoint."""
        # Mock model predictions
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6]])
        mock_model.predict.return_value = np.array([0, 1])
        mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        
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
        large_batch = {
            "customers": [sample_customer_data] * 1001  # Exceeds limit of 1000
        }
        
        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 422  # Validation error
    
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
            "EstimatedSalary": 75000.0
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
        
        data = pd.DataFrame({
            'CreditScore': np.random.randint(300, 850, n_samples),
            'Geography_France': np.random.choice([0, 1], n_samples),
            'Geography_Germany': np.random.choice([0, 1], n_samples),
            'Geography_Spain': np.random.choice([0, 1], n_samples),
            'Gender_Female': np.random.choice([0, 1], n_samples),
            'Gender_Male': np.random.choice([0, 1], n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(0, 10, n_samples),
            'Balance': np.random.uniform(0, 200000, n_samples),
            'NumOfProducts': np.random.randint(1, 4, n_samples),
            'HasCrCard': np.random.choice([0, 1], n_samples),
            'IsActiveMember': np.random.choice([0, 1], n_samples),
            'EstimatedSalary': np.random.uniform(10000, 150000, n_samples),
            'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        return data
    
    def test_model_training_performance(self, large_dataset):
        """Test model training performance on large dataset."""
        import time
        
        X = large_dataset.drop('Exited', axis=1)
        y = large_dataset['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test RandomForest training time
        start_time = time.time()
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Test prediction time
        start_time = time.time()
        predictions = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Performance assertions
        assert training_time < 60  # Should train in less than 60 seconds
        assert prediction_time < 5  # Should predict in less than 5 seconds
        
        # Accuracy should be reasonable
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.7  # Should achieve at least 70% accuracy
        
        print(f"Training time: {training_time:.2f}s")
        print(f"Prediction time: {prediction_time:.2f}s")
        print(f"Accuracy: {accuracy:.3f}")
    
    @pytest.mark.skipif(importlib.util.find_spec("psutil") is None, reason="psutil not installed")
    def test_memory_usage(self, large_dataset):
        """Test memory usage during model training."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        X = large_dataset.drop('Exited', axis=1)
        y = large_dataset['Exited']
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 1000  # Should use less than 1GB additional memory
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])