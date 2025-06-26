#!/usr/bin/env python3
"""
Model Explainability and Fairness Analysis

This module provides comprehensive model explainability using SHAP (SHapley Additive exPlanations)
and fairness analysis across different demographic groups. It generates interpretable insights
for stakeholders and ensures model fairness.

Author: Bank Churn Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
    make_subplots = None

warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Model explainability and fairness analysis using SHAP.
    
    This class provides comprehensive model interpretation including:
    - Global feature importance using SHAP
    - Local explanations for individual predictions
    - Fairness analysis across demographic groups
    - Interactive visualizations and reports
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelExplainer class.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.model_name = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.explainer = None
        self.shap_values = None
        self.X_explain = None
        self.fairness_results = {}
        
        # Set up paths
        self.project_root = Path.cwd().parent.parent
        self.data_dir = self.project_root / 'data'
        self.models_dir = self.project_root / 'models'
        self.reports_dir = self.project_root / 'reports'
        self.figures_dir = self.reports_dir / 'figures'
        
        # Create directories if they don't exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load trained model and preprocessing components.
        
        Args:
            model_path (str, optional): Path to model file. If None, uses default best model.
        """
        if model_path is None:
            model_path = self.models_dir / 'churn_model.pkl'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"📦 Loading model from {model_path}...")
        
        # Load model package
        model_package = joblib.load(model_path)
        
        self.model = model_package['model']
        self.model_name = model_package['model_name']
        self.scaler = model_package.get('scaler')
        self.label_encoders = model_package.get('label_encoders', {})
        self.feature_names = model_package['feature_names']
        
        # Populate feature_names from scaler if available
        if self.scaler is not None and hasattr(self.scaler, 'feature_names_in_'):
            self.feature_names = list(self.scaler.feature_names_in_)
        
        print(f"✅ Loaded {self.model_name} model with {len(self.feature_names)} features")
    
    def load_data(self, file_path: Optional[str] = None, sample_size: int = 1000) -> pd.DataFrame:
        """
        Load and prepare data for explanation.
        
        Args:
            file_path (str, optional): Path to data file
            sample_size (int): Number of samples to use for SHAP explanation
            
        Returns:
            pd.DataFrame: Prepared data for explanation
        """
        if file_path is None:
            # Try feature-engineered data first, then processed
            feature_path = self.data_dir / 'processed' / 'churn_features.parquet'
            processed_path = self.data_dir / 'processed' / 'churn_cleaned.parquet'
            
            if feature_path.exists():
                file_path = feature_path
            elif processed_path.exists():
                file_path = processed_path
            else:
                raise FileNotFoundError("No suitable data file found.")
        
        print(f"📊 Loading data from {file_path}...")
        df = pd.read_parquet(file_path)
        
        # Sample data for explanation (SHAP can be computationally expensive)
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=self.random_state)
            print(f"📉 Sampled {sample_size} rows for explanation (original: {len(df)})")
        else:
            df_sample = df.copy()
        
        return df_sample
    
    def prepare_explanation_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for SHAP explanation.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
        """
        print("🔧 Preparing data for explanation...")
        
        # Use self.feature_names for consistent feature selection
        X = df[self.feature_names]
        if self.scaler is not None:
            X = pd.DataFrame(self.scaler.transform(X), 
                            columns=self.feature_names, 
                            index=df.index)
        else:
            X = pd.DataFrame(X, columns=self.feature_names, index=df.index)
        y = pd.Series(df['Exited'], name='Exited', index=df.index)
        
        print(f"Explanation data shape: {X.shape}")
        return X, y
    
    def create_shap_explainer(self, background_size: int = 100) -> None:
        """
        Create SHAP explainer for the model.
        
        Args:
            background_size (int): Size of background dataset for SHAP
        """
        print(f"🔍 Creating SHAP explainer (background size: {background_size})...")
        
        # Create background dataset
        background = shap.sample(self.X_explain, background_size, random_state=self.random_state)
        
        # Create explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            # For probabilistic models
            self.explainer = shap.Explainer(self.model.predict_proba, background)
        else:
            # For other models
            self.explainer = shap.Explainer(self.model, background)
        
        print("✅ SHAP explainer created successfully")
    
    def calculate_shap_values(self, max_evals: int = 500) -> np.ndarray:
        """
        Calculate SHAP values for the explanation dataset.
        
        Args:
            max_evals (int): Maximum number of evaluations for SHAP
            
        Returns:
            np.ndarray: SHAP values
        """
        print(f"⚡ Calculating SHAP values (max_evals: {max_evals})...")
        
        # Calculate SHAP values
        self.shap_values = self.explainer(
            self.X_explain, 
            max_evals=max_evals,
            silent=True
        )
        
        # For binary classification, we typically want the positive class
        if len(self.shap_values.shape) == 3 and self.shap_values.shape[-1] == 2:
            self.shap_values = self.shap_values[:, :, 1]
        
        print(f"✅ SHAP values calculated: {self.shap_values.shape}")
        return self.shap_values
    
    def create_global_explanations(self) -> None:
        """
        Create global model explanations using SHAP.
        """
        print("🌍 Creating global explanations...")
        
        # 1. Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_explain, 
                         feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Churn Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature importance (mean absolute SHAP values)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_explain, 
                         feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance - Mean Impact on Model Output', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Dependence plots for top features
        self._create_dependence_plots()
        
        print(f"✅ Global explanations saved to {self.figures_dir}")
    
    def _create_dependence_plots(self, top_n: int = 4) -> None:
        """
        Create SHAP dependence plots for top features.
        
        Args:
            top_n (int): Number of top features to plot
        """
        # Calculate feature importance
        feature_importance = np.abs(self.shap_values).mean(0)
        top_features_idx = np.argsort(feature_importance)[-top_n:]
        top_features = [self.feature_names[i] for i in top_features_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, feature_idx in enumerate(top_features_idx):
            if i < len(axes):
                shap.dependence_plot(
                    feature_idx, self.shap_values, self.X_explain,
                    feature_names=self.feature_names,
                    ax=axes[i], show=False
                )
                axes[i].set_title(f'SHAP Dependence: {self.feature_names[feature_idx]}', 
                                fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'shap_dependence_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_local_explanations(self, sample_indices: Optional[List[int]] = None, 
                                 n_samples: int = 5) -> None:
        """
        Create local explanations for individual predictions.
        
        Args:
            sample_indices (List[int], optional): Specific indices to explain
            n_samples (int): Number of random samples to explain if indices not provided
        """
        print(f"🎯 Creating local explanations for {n_samples} samples...")
        
        if sample_indices is None:
            # Select random samples
            sample_indices = np.random.choice(
                len(self.X_explain), size=min(n_samples, len(self.X_explain)), 
                replace=False
            )
        
        # Create waterfall plots for individual predictions
        fig, axes = plt.subplots(len(sample_indices), 1, 
                                figsize=(12, 6 * len(sample_indices)))
        
        if len(sample_indices) == 1:
            axes = [axes]
        
        for i, idx in enumerate(sample_indices):
            # Get prediction
            if self.scaler is not None:
                prediction = self.model.predict_proba(self.X_explain.iloc[[idx]])[0, 1]
            else:
                prediction = self.model.predict_proba(self.X_explain.iloc[[idx]])[0, 1]
            
            # Create waterfall plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[idx],
                    base_values=self.explainer.expected_value,
                    data=self.X_explain.iloc[idx],
                    feature_names=self.feature_names
                ),
                max_display=10,
                show=False
            )
            
            plt.title(f'Local Explanation - Sample {idx} (Churn Probability: {prediction:.3f})', 
                     fontsize=12, fontweight='bold')
            
            if i < len(sample_indices) - 1:
                plt.xlabel('')  # Remove x-label for all but last plot
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'shap_local_explanations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_fairness(self, df: pd.DataFrame, y_true: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Analyze model fairness across different demographic groups.
        
        Args:
            df (pd.DataFrame): Original data with demographic information
            y_true (pd.Series): True labels
            
        Returns:
            Dict[str, Dict[str, float]]: Fairness metrics by group
        """
        print("⚖️  Analyzing model fairness...")
        
        # Get model predictions
        if self.scaler is not None:
            y_pred_proba = self.model.predict_proba(self.X_explain)[:, 1]
            y_pred = self.model.predict(self.X_explain)
        else:
            y_pred_proba = self.model.predict_proba(self.X_explain)[:, 1]
            y_pred = self.model.predict(self.X_explain)
        
        fairness_results = {}
        
        # Analyze by gender
        if 'Gender' in df.columns:
            fairness_results['Gender'] = self._analyze_group_fairness(
                df['Gender'], y_true, y_pred, y_pred_proba, 'Gender'
            )
        
        # Analyze by geography
        if 'Geography' in df.columns:
            fairness_results['Geography'] = self._analyze_group_fairness(
                df['Geography'], y_true, y_pred, y_pred_proba, 'Geography'
            )
        
        # Analyze by age groups
        if 'Age' in df.columns:
            age_groups = pd.cut(df['Age'], bins=[0, 35, 50, 65, 100], 
                              labels=['Young (≤35)', 'Middle (36-50)', 'Senior (51-65)', 'Elder (65+)'])
            fairness_results['Age_Group'] = self._analyze_group_fairness(
                age_groups, y_true, y_pred, y_pred_proba, 'Age Group'
            )
        
        self.fairness_results = fairness_results
        
        # Create fairness visualization
        self._plot_fairness_analysis()
        
        return fairness_results
    
    def _analyze_group_fairness(self, group_col: pd.Series, y_true: pd.Series, 
                               y_pred: np.ndarray, y_pred_proba: np.ndarray, 
                               group_name: str) -> Dict[str, float]:
        """
        Analyze fairness metrics for a specific demographic group.
        """
        group_metrics = {}
        
        for group_value in group_col.unique():
            if pd.isna(group_value):
                continue
                
            # Get indices for this group
            group_mask = group_col == group_value
            
            if group_mask.sum() == 0:
                continue
            
            # Calculate metrics for this group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            group_y_pred_proba = y_pred_proba[group_mask]
            
            metrics = {
                'size': group_mask.sum(),
                'base_rate': group_y_true.mean(),  # Actual churn rate
                'prediction_rate': group_y_pred.mean(),  # Predicted churn rate
                'accuracy': accuracy_score(group_y_true, group_y_pred),
                'precision': precision_score(group_y_true, group_y_pred, zero_division=0),
                'recall': recall_score(group_y_true, group_y_pred, zero_division=0),
                'roc_auc': roc_auc_score(group_y_true, group_y_pred_proba) if len(np.unique(group_y_true)) > 1 else 0
            }
            
            group_metrics[str(group_value)] = metrics
        
        return group_metrics
    
    def _plot_fairness_analysis(self) -> None:
        """
        Create visualizations for fairness analysis.
        """
        n_groups = len(self.fairness_results)
        if n_groups == 0:
            print("⚠️  No fairness results to plot")
            return
        
        fig, axes = plt.subplots(2, n_groups, figsize=(6 * n_groups, 12))
        
        if n_groups == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (group_name, group_data) in enumerate(self.fairness_results.items()):
            # Prepare data for plotting
            groups = list(group_data.keys())
            metrics_data = {
                'accuracy': [group_data[g]['accuracy'] for g in groups],
                'precision': [group_data[g]['precision'] for g in groups],
                'recall': [group_data[g]['recall'] for g in groups],
                'roc_auc': [group_data[g]['roc_auc'] for g in groups]
            }
            
            # Plot performance metrics
            x = np.arange(len(groups))
            width = 0.2
            
            for j, (metric, values) in enumerate(metrics_data.items()):
                axes[0, i].bar(x + j * width, values, width, label=metric.upper())
            
            axes[0, i].set_xlabel(group_name)
            axes[0, i].set_ylabel('Score')
            axes[0, i].set_title(f'Model Performance by {group_name}', fontweight='bold')
            axes[0, i].set_xticks(x + width * 1.5)
            axes[0, i].set_xticklabels(groups, rotation=45)
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot base rate vs prediction rate
            base_rates = [group_data[g]['base_rate'] for g in groups]
            pred_rates = [group_data[g]['prediction_rate'] for g in groups]
            
            axes[1, i].bar(x - width/2, base_rates, width, label='Actual Churn Rate', alpha=0.7)
            axes[1, i].bar(x + width/2, pred_rates, width, label='Predicted Churn Rate', alpha=0.7)
            
            axes[1, i].set_xlabel(group_name)
            axes[1, i].set_ylabel('Churn Rate')
            axes[1, i].set_title(f'Churn Rates by {group_name}', fontweight='bold')
            axes[1, i].set_xticks(x)
            axes[1, i].set_xticklabels(groups, rotation=45)
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fairness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_explanation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report.
        
        Returns:
            Dict[str, Any]: Explanation report with insights and recommendations
        """
        print("📋 Generating explanation report...")
        
        # Calculate feature importance from SHAP values
        feature_importance = np.abs(self.shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Top features
        top_features = feature_importance_df.head(10)
        
        # Generate insights
        insights = {
            'model_name': self.model_name,
            'explanation_summary': {
                'total_samples_explained': len(self.X_explain),
                'total_features': len(self.feature_names),
                'top_5_features': top_features.head(5)['feature'].tolist()
            },
            'feature_insights': self._generate_feature_insights(feature_importance_df),
            'fairness_summary': self._generate_fairness_summary(),
            'business_recommendations': self._generate_business_recommendations(feature_importance_df)
        }
        
        # Save report
        report_path = self.reports_dir / 'model_explanation_report.json'
        with open(report_path, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        print(f"✅ Explanation report saved to {report_path}")
        
        # Print summary
        self._print_explanation_summary(insights)
        
        return insights
    
    def _generate_feature_insights(self, feature_importance_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate insights about feature importance.
        """
        top_5 = feature_importance_df.head(5)
        
        insights = {
            'most_important_feature': top_5.iloc[0]['feature'],
            'most_important_feature_impact': float(top_5.iloc[0]['importance']),
            'top_5_features': [
                {
                    'feature': row['feature'],
                    'importance': float(row['importance']),
                    'rank': i + 1
                }
                for i, (_, row) in enumerate(top_5.iterrows())
            ],
            'feature_categories': self._categorize_features(feature_importance_df)
        }
        
        return insights
    
    def _categorize_features(self, feature_importance_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features by type and importance.
        """
        categories = {
            'demographic': [],
            'financial': [],
            'behavioral': [],
            'product': []
        }
        
        for _, row in feature_importance_df.iterrows():
            feature = row['feature'].lower()
            
            if any(term in feature for term in ['age', 'gender', 'geography']):
                categories['demographic'].append(row['feature'])
            elif any(term in feature for term in ['balance', 'salary', 'credit']):
                categories['financial'].append(row['feature'])
            elif any(term in feature for term in ['active', 'tenure', 'card']):
                categories['behavioral'].append(row['feature'])
            elif any(term in feature for term in ['product', 'num']):
                categories['product'].append(row['feature'])
        
        return categories
    
    def _generate_fairness_summary(self) -> Dict[str, Any]:
        """
        Generate summary of fairness analysis.
        """
        if not self.fairness_results:
            return {'status': 'No fairness analysis performed'}
        
        summary = {
            'groups_analyzed': list(self.fairness_results.keys()),
            'fairness_concerns': [],
            'recommendations': []
        }
        
        # Check for significant performance differences
        for group_name, group_data in self.fairness_results.items():
            accuracies = [metrics['accuracy'] for metrics in group_data.values()]
            if max(accuracies) - min(accuracies) > 0.1:  # 10% difference threshold
                summary['fairness_concerns'].append(
                    f"Significant accuracy difference across {group_name} groups"
                )
                summary['recommendations'].append(
                    f"Review model performance for {group_name} groups and consider bias mitigation"
                )
        
        return summary
    
    def _generate_business_recommendations(self, feature_importance_df: pd.DataFrame) -> List[str]:
        """
        Generate business recommendations based on feature importance.
        """
        recommendations = []
        top_features = feature_importance_df.head(5)['feature'].tolist()
        
        # Feature-specific recommendations
        for feature in top_features:
            feature_lower = feature.lower()
            
            if 'age' in feature_lower:
                recommendations.append(
                    "🎯 Develop age-specific retention strategies and products"
                )
            elif 'balance' in feature_lower:
                recommendations.append(
                    "💰 Focus on customers with specific balance ranges for targeted interventions"
                )
            elif 'product' in feature_lower:
                recommendations.append(
                    "📦 Optimize product portfolio and cross-selling strategies"
                )
            elif 'geography' in feature_lower:
                recommendations.append(
                    "🌍 Implement region-specific retention campaigns"
                )
            elif 'active' in feature_lower:
                recommendations.append(
                    "💼 Enhance customer engagement and activity programs"
                )
        
        # General recommendations
        recommendations.extend([
            "📊 Monitor top features regularly for early churn warning signs",
            "🔄 Implement real-time scoring based on key features",
            "🎯 Personalize retention offers based on individual feature profiles"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _print_explanation_summary(self, insights: Dict[str, Any]) -> None:
        """
        Print explanation summary to console.
        """
        print("\n🔍 MODEL EXPLANATION SUMMARY")
        print("=" * 50)
        
        print(f"Model: {insights['model_name']}")
        print(f"Samples Explained: {insights['explanation_summary']['total_samples_explained']:,}")
        print(f"Total Features: {insights['explanation_summary']['total_features']}")
        
        print("\n🏆 TOP 5 MOST IMPORTANT FEATURES:")
        print("-" * 40)
        for i, feature_info in enumerate(insights['feature_insights']['top_5_features'], 1):
            print(f"{i}. {feature_info['feature']} (Impact: {feature_info['importance']:.3f})")
        
        print("\n💡 BUSINESS RECOMMENDATIONS:")
        print("-" * 40)
        for rec in insights['business_recommendations']:
            print(f"  • {rec}")
        
        if insights['fairness_summary'].get('fairness_concerns'):
            print("\n⚠️  FAIRNESS CONCERNS:")
            print("-" * 30)
            for concern in insights['fairness_summary']['fairness_concerns']:
                print(f"  • {concern}")
    
    def save_explanation_artifacts(self) -> None:
        """
        Save all explanation artifacts for deployment.
        """
        print("💾 Saving explanation artifacts...")
        
        # Save SHAP explainer
        explainer_path = self.models_dir / 'shap_explainer.pkl'
        joblib.dump(self.explainer, explainer_path)
        
        # Save SHAP values
        shap_values_path = self.models_dir / 'shap_values.npy'
        np.save(shap_values_path, self.shap_values)
        
        # Save explanation data
        explanation_data_path = self.models_dir / 'explanation_data.parquet'
        self.X_explain.to_parquet(explanation_data_path)
        
        print(f"✅ Explanation artifacts saved to {self.models_dir}")


def run_full_explanation_pipeline(model_path: Optional[str] = None, 
                                 data_path: Optional[str] = None,
                                 sample_size: int = 1000) -> ModelExplainer:
    """
    Run the complete model explanation pipeline.
    
    Args:
        model_path (str, optional): Path to trained model
        data_path (str, optional): Path to data file
        sample_size (int): Number of samples for explanation
        
    Returns:
        ModelExplainer: Configured explainer with results
    """
    print("🚀 Starting Model Explanation Pipeline")
    print("=" * 50)
    
    # Initialize explainer
    explainer = ModelExplainer()
    
    try:
        # Load model
        explainer.load_model(model_path)
        
        # Load and prepare data
        df = explainer.load_data(data_path, sample_size)
        X_explain, y_true = explainer.prepare_explanation_data(df)
        
        # Create SHAP explainer
        explainer.create_shap_explainer()
        
        # Calculate SHAP values
        explainer.calculate_shap_values()
        
        # Create global explanations
        explainer.create_global_explanations()
        
        # Create local explanations
        explainer.create_local_explanations()
        
        # Analyze fairness
        if y_true is not None:
            explainer.analyze_fairness(df, y_true)
        
        # Generate explanation report
        explainer.generate_explanation_report()
        
        # Save artifacts
        explainer.save_explanation_artifacts()
        
        print("\n✅ Model explanation pipeline completed successfully!")
        print(f"📊 Visualizations saved to: {explainer.figures_dir}")
        print(f"📋 Report saved to: {explainer.reports_dir}")
        
        return explainer
        
    except Exception as e:
        print(f"❌ Error in explanation pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the explanation pipeline
    model_explainer = run_full_explanation_pipeline()