#!/usr/bin/env python3
"""
Churn Prediction Model Training

This module implements supervised learning for customer churn prediction.
It includes baseline and advanced models with comprehensive evaluation,
hyperparameter tuning, and model serialization.

Author: Bank Churn Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, 
    StratifiedKFold, learning_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import joblib
import warnings
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import time

warnings.filterwarnings('ignore')


class ChurnPredictor:
    """
    Customer churn prediction using machine learning.
    
    This class handles the complete churn prediction pipeline including:
    - Data preprocessing and feature engineering
    - Model training with multiple algorithms
    - Hyperparameter tuning and cross-validation
    - Model evaluation and calibration
    - Model serialization and deployment preparation
    """
    
    def __init__(self, random_state: int = 42, project_root: Optional[Path] = None):
        """
        Initialize the ChurnPredictor class.
        
        Args:
            random_state (int): Random state for reproducibility
            project_root (Path, optional): Project root directory
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.evaluation_results = {}
        
        # Set up paths
        self.project_root = project_root or Path.cwd().parent.parent
        self.data_dir = self.project_root / 'data'
        self.processed_dir = self.data_dir / 'processed'
        self.models_dir = self.project_root / 'models'
        self.reports_dir = self.project_root / 'reports'
        self.figures_dir = self.reports_dir / 'figures'
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load processed customer data.
        
        Args:
            file_path (str, optional): Path to data file. If None, uses default processed data.
            
        Returns:
            pd.DataFrame: Loaded customer data
        """
        if file_path is None:
            # Try feature-engineered data first, then processed, then interim
            feature_path = self.data_dir / 'processed' / 'churn_features.parquet'
            processed_path = self.data_dir / 'processed' / 'churn_cleaned.parquet'
            interim_path = self.data_dir / 'interim' / 'churn_raw.parquet'
            
            if feature_path.exists():
                file_path = feature_path
                print(f"✅ Loading feature-engineered data from {file_path}")
            elif processed_path.exists():
                file_path = processed_path
                print(f"✅ Loading processed data from {file_path}")
            elif interim_path.exists():
                file_path = interim_path
                print(f"⚠️  Loading interim data from {file_path}")
                print("Note: Consider running data cleaning and feature engineering pipelines.")
            else:
                raise FileNotFoundError("No data file found. Please run data loading pipeline first.")
        
        df = pd.read_parquet(file_path)
        print(f"Data loaded successfully: {df.shape}")
        
        # Verify target column exists
        if 'Exited' not in df.columns:
            raise ValueError("Target column 'Exited' not found in the dataset.")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for model training.
        
        Args:
            df (pd.DataFrame): Input customer data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
        """
        print("🔧 Preprocessing data for model training...")
        
        # Separate features and target
        target_col = 'Exited'
        feature_cols = [col for col in df.columns if col not in [target_col, 'CustomerId', 'Surname']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        print(f"Features: {len(feature_cols)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("⚠️  Found missing values. Filling with median/mode...")
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        print(f"Final feature matrix shape: {X.shape}")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets with stratification.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of data for testing
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        print(f"📊 Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training churn rate: {y_train.mean():.3f}")
        print(f"Test churn rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data from processed churn features.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
        """
        feature_path = self.data_dir / 'processed' / 'churn_features.parquet'
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        
        df = pd.read_parquet(feature_path)
        
        # Separate features and target
        target_col = 'Exited'
        feature_cols = [col for col in df.columns if col not in [target_col, 'CustomerId', 'Surname']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        return X, y
    
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[LogisticRegression, Dict[str, float]]:
        """
        Train baseline logistic regression model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Tuple[LogisticRegression, Dict[str, float]]: Trained model and metrics
        """
        print("🎯 Training baseline logistic regression model...")
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train logistic regression
        lr_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        lr_model.fit(X_train_scaled, y_train)
        
        self.models['logistic_regression'] = lr_model
        
        # Evaluate on test set
        y_pred = lr_model.predict(X_test_scaled)
        y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"Baseline model test ROC-AUC: {metrics['roc_auc']:.3f}")
        
        return lr_model, metrics
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> List[float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            List[float]: List of CV scores
        """
        # Use logistic regression for cross-validation
        lr_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            lr_model, X_scaled, y, 
            cv=cv_folds, scoring='roc_auc'
        )
        
        print(f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_scores.tolist()
    
    def train_advanced_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train advanced models with hyperparameter tuning.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        print("🚀 Training advanced models with hyperparameter tuning...")
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = self._train_random_forest(X_train, y_train)
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = self._train_xgboost(X_train, y_train)
        
        return {
            'random_forest': rf_model,
            'xgboost': xgb_model
        }
    
    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train Random Forest with hyperparameter tuning.
        """
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']
        }
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Random Forest training completed in {training_time:.2f} seconds")
        print(f"Best RF parameters: {grid_search.best_params_}")
        print(f"Best RF CV score: {grid_search.best_score_:.3f}")
        
        self.models['random_forest'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Train XGBoost with hyperparameter tuning.
        """
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'scale_pos_weight': [scale_pos_weight]
        }
        
        # Grid search with cross-validation
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            enable_categorical=False
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"XGBoost training completed in {training_time:.2f} seconds")
        print(f"Best XGBoost parameters: {grid_search.best_params_}")
        print(f"Best XGBoost CV score: {grid_search.best_score_:.3f}")
        
        self.models['xgboost'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def evaluate_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            y_train (pd.Series): Training target
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each model
        """
        print("📊 Evaluating models on test data...")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Prepare data based on model type
            if model_name == 'logistic_regression':
                X_test_processed = self.scaler.transform(X_test)
                X_train_processed = self.scaler.transform(X_train)
            else:
                X_test_processed = X_test
                X_train_processed = X_train
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train_processed, y_train, 
                cv=5, scoring='roc_auc'
            )
            metrics['cv_roc_auc_mean'] = cv_scores.mean()
            metrics['cv_roc_auc_std'] = cv_scores.std()
            
            evaluation_results[model_name] = metrics
            
            # Print results
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"  CV ROC-AUC: {metrics['cv_roc_auc_mean']:.3f} (+/- {metrics['cv_roc_auc_std'] * 2:.3f})")
        
        self.evaluation_results = evaluation_results
        
        # Select best model based on ROC-AUC
        best_model_name = max(evaluation_results.keys(), 
                             key=lambda x: evaluation_results[x]['roc_auc'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best model: {best_model_name} (ROC-AUC: {evaluation_results[best_model_name]['roc_auc']:.3f})")
        
        return evaluation_results
    
    def create_evaluation_plots(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Create comprehensive evaluation plots.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        """
        print("📈 Creating evaluation plots...")
        
        # 1. ROC curves
        self._plot_roc_curves(X_test, y_test)
        
        # 2. Precision-Recall curves
        self._plot_precision_recall_curves(X_test, y_test)
        
        # 3. Confusion matrices
        self._plot_confusion_matrices(X_test, y_test)
        
        # 4. Feature importance (for tree-based models)
        self._plot_feature_importance()
        
        # 5. Model comparison
        self._plot_model_comparison()
        
        # 6. Calibration plots
        self._plot_calibration_curves(X_test, y_test)
        
        print(f"✅ Evaluation plots saved to {self.figures_dir}")
    
    def _plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot ROC curves for all models.
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            # Prepare data
            if model_name == 'logistic_regression':
                X_test_processed = self.scaler.transform(X_test)
            else:
                X_test_processed = X_test
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Plot
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_precision_recall_curves(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot Precision-Recall curves for all models.
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            # Prepare data
            if model_name == 'logistic_regression':
                X_test_processed = self.scaler.transform(X_test)
            else:
                X_test_processed = X_test
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            # Plot
            plt.plot(recall, precision, linewidth=2,
                    label=f'{model_name.replace("_", " ").title()}')
        
        # Plot baseline
        baseline = y_test.mean()
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Baseline (Prevalence = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrices(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot confusion matrices for all models.
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, model) in enumerate(self.models.items()):
            # Prepare data
            if model_name == 'logistic_regression':
                X_test_processed = self.scaler.transform(X_test)
            else:
                X_test_processed = X_test
            
            # Get predictions
            y_pred = model.predict(X_test_processed)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Retained', 'Churned'],
                       yticklabels=['Retained', 'Churned'])
            axes[i].set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self) -> None:
        """
        Plot feature importance for tree-based models.
        """
        tree_models = ['random_forest', 'xgboost']
        available_tree_models = [name for name in tree_models if name in self.models]
        
        if not available_tree_models:
            print("⚠️  No tree-based models available for feature importance plot")
            return
        
        fig, axes = plt.subplots(1, len(available_tree_models), figsize=(8 * len(available_tree_models), 8))
        
        if len(available_tree_models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(available_tree_models):
            model = self.models[model_name]
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                continue
            
            # Create DataFrame for plotting
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            # Plot top 15 features
            top_features = feature_importance_df.tail(15)
            
            axes[i].barh(top_features['feature'], top_features['importance'])
            axes[i].set_title(f'Feature Importance - {model_name.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_model_comparison(self) -> None:
        """
        Plot model comparison metrics.
        """
        # Prepare data for plotting
        metrics_df = pd.DataFrame(self.evaluation_results).T
        
        # Select key metrics for comparison
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        available_metrics = [metric for metric in key_metrics if metric in metrics_df.columns]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot of metrics
        metrics_df[available_metrics].plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Performance Comparison', fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Model')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for model_name in metrics_df.index:
            values = metrics_df.loc[model_name, available_metrics].values
            values = np.concatenate((values, [values[0]]))
            
            axes[1].plot(angles, values, 'o-', linewidth=2, label=model_name.replace('_', ' ').title())
            axes[1].fill(angles, values, alpha=0.25)
        
        axes[1].set_xticks(angles[:-1])
        axes[1].set_xticklabels(available_metrics)
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Model Performance Radar Chart', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_calibration_curves(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot calibration curves for probability calibration analysis.
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            # Prepare data
            if model_name == 'logistic_regression':
                X_test_processed = self.scaler.transform(X_test)
            else:
                X_test_processed = X_test
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_pred_proba, n_bins=10
            )
            
            # Plot
            plt.plot(mean_predicted_value, fraction_of_positives, 'o-',
                    label=f'{model_name.replace("_", " ").title()}')
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves - Probability Calibration', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def calibrate_best_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Calibrate the best model for better probability estimates.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Any: Calibrated model
        """
        print(f"🎯 Calibrating best model ({self.best_model_name})...")
        
        # Prepare training data
        if self.best_model_name == 'logistic_regression':
            X_train_processed = self.scaler.transform(X_train)
        else:
            X_train_processed = X_train
        
        # Calibrate using Platt scaling
        calibrated_model = CalibratedClassifierCV(
            self.best_model, method='sigmoid', cv=3
        )
        calibrated_model.fit(X_train_processed, y_train)
        
        # Save calibrated model
        self.models[f'{self.best_model_name}_calibrated'] = calibrated_model
        
        print("✅ Model calibration completed")
        return calibrated_model
    
    def save_model(self, model, model_name: str, metrics: Dict[str, float]) -> None:
        """
        Save a single model with its metrics.
        
        Args:
            model: Trained model to save
            model_name: Name of the model
            metrics: Model evaluation metrics
        """
        print(f"💾 Saving {model_name} model...")
        
        # Create deployment package
        deployment_package = {
            'model': model,
            'model_name': model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'evaluation_metrics': metrics
        }
        
        model_path = self.models_dir / 'churn_model.pkl'
        joblib.dump(deployment_package, model_path)
        print(f"✅ Saved {model_name} model package to {model_path}")
    
    def save_models(self) -> None:
        """
        Save all trained models and preprocessing components.
        """
        print("💾 Saving models and preprocessing components...")
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = self.models_dir / f'{model_name}_model.pkl'
            joblib.dump(model, model_path)
            print(f"✅ Saved {model_name} to {model_path}")
        
        # Save preprocessing components
        preprocessing_components = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        preprocessing_path = self.models_dir / 'preprocessing_components.pkl'
        joblib.dump(preprocessing_components, preprocessing_path)
        print(f"✅ Saved preprocessing components to {preprocessing_path}")
        
        # Save best model separately for easy deployment
        if self.best_model is not None:
            best_model_path = self.models_dir / 'churn_model.pkl'
            
            # Create deployment package
            deployment_package = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'scaler': self.scaler if self.best_model_name == 'logistic_regression' else None,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'evaluation_metrics': self.evaluation_results[self.best_model_name]
            }
            
            joblib.dump(deployment_package, best_model_path)
            print(f"✅ Saved best model package to {best_model_path}")
        
        # Save evaluation results
        results_path = self.reports_dir / 'model_evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        print(f"✅ Saved evaluation results to {results_path}")


def run_full_pipeline(data_path: Optional[str] = None, test_size: float = 0.2) -> ChurnPredictor:
    """
    Run the complete churn prediction pipeline.
    
    Args:
        data_path (str, optional): Path to input data file
        test_size (float): Proportion of data for testing
        
    Returns:
        ChurnPredictor: Trained churn prediction model
    """
    print("🚀 Starting Churn Prediction Pipeline")
    print("=" * 50)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    try:
        # Load and preprocess data
        df = predictor.load_data(data_path)
        X, y = predictor.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size)
        
        # Train baseline model
        baseline_model = predictor.train_baseline_model(X_train, y_train)
        
        # Train advanced models
        advanced_models = predictor.train_advanced_models(X_train, y_train)
        
        # Evaluate all models
        evaluation_results = predictor.evaluate_models(X_train, X_test, y_train, y_test)
        
        # Create evaluation plots
        predictor.create_evaluation_plots(X_test, y_test)
        
        # Calibrate best model
        calibrated_model = predictor.calibrate_best_model(X_train, y_train)
        
        # Save models
        predictor.save_models()
        
        # Print final summary
        print("\n🎉 CHURN PREDICTION PIPELINE COMPLETED!")
        print("=" * 50)
        print(f"Best Model: {predictor.best_model_name}")
        print(f"Best ROC-AUC: {evaluation_results[predictor.best_model_name]['roc_auc']:.3f}")
        print(f"Models saved to: {predictor.models_dir}")
        print(f"Evaluation plots saved to: {predictor.figures_dir}")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Error in churn prediction pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the churn prediction pipeline
    churn_predictor = run_full_pipeline()