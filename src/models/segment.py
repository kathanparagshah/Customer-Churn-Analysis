#!/usr/bin/env python3
"""
Customer Segmentation Analysis

This module performs unsupervised learning to segment bank customers based on their
characteristics and behaviors. It uses K-Means clustering with optimal cluster
selection using elbow method and silhouette analysis.

Author: Bank Churn Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
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
import warnings
import joblib
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """
    Customer segmentation using K-Means clustering.
    
    This class handles the complete customer segmentation pipeline including:
    - Feature selection and preprocessing
    - Optimal cluster number selection
    - Model training and evaluation
    - Segment analysis and profiling
    - Visualization and reporting
    """
    
    def __init__(self, random_state: int = 42, project_root: Optional[Path] = None):
        """
        Initialize the CustomerSegmentation class.
        
        Args:
            random_state (int): Random state for reproducibility
            project_root (Path, optional): Project root directory
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self.optimal_k = None
        self.silhouette_scores = {}
        self.inertias = {}
        self.segment_profiles = {}
        
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
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for clustering by selecting numeric features and scaling them.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        # Select numeric features for clustering
        numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                          'NumOfProducts', 'EstimatedSalary']
        
        # Filter to available columns
        available_features = [col for col in numeric_features if col in df.columns]
        
        if not available_features:
            raise ValueError("No numeric features available for clustering")
        
        # Extract features and handle missing values
        X = df[available_features].copy()
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=available_features,
            index=X.index
        )
        
        return X_scaled
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load processed customer data.
        
        Args:
            file_path (str, optional): Path to data file. If None, uses default processed data.
            
        Returns:
            pd.DataFrame: Loaded customer data
        """
        if file_path is None:
            # Try processed data first, then interim
            processed_path = self.data_dir / 'processed' / 'churn_cleaned.parquet'
            interim_path = self.data_dir / 'interim' / 'churn_raw.parquet'
            
            if processed_path.exists():
                file_path = processed_path
                print(f"✅ Loading processed data from {file_path}")
            elif interim_path.exists():
                file_path = interim_path
                print(f"⚠️  Loading interim data from {file_path}")
                print("Note: Consider running data cleaning pipeline for better results.")
            else:
                raise FileNotFoundError("No data file found. Please run data loading pipeline first.")
        
        df = pd.read_parquet(file_path)
        print(f"Data loaded successfully: {df.shape}")
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare features for clustering.
        
        Args:
            df (pd.DataFrame): Input customer data
            
        Returns:
            pd.DataFrame: Selected features for clustering
        """
        # Primary clustering features
        clustering_features = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'EstimatedSalary'
        ]
        
        # Additional features if available
        optional_features = ['HasCrCard', 'IsActiveMember']
        
        # Select available features
        available_features = [col for col in clustering_features if col in df.columns]
        available_optional = [col for col in optional_features if col in df.columns]
        
        selected_features = available_features + available_optional
        
        print(f"Selected features for clustering: {selected_features}")
        
        # Create feature matrix
        X = df[selected_features].copy()
        
        # Handle missing values if any
        if X.isnull().sum().sum() > 0:
            print("⚠️  Found missing values. Filling with median/mode...")
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)
        
        return X
    
    def find_optimal_clusters(self, X: pd.DataFrame, max_k: int = 10, k_range: range = None) -> Tuple[int, List[float], List[float]]:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            X (pd.DataFrame): Feature matrix
            max_k (int): Maximum number of clusters to test (ignored if k_range provided)
            k_range (range): Range of k values to test
            
        Returns:
            Tuple[int, List[float], List[float]]: Optimal k, elbow scores (inertias), silhouette scores
        """
        print("🔍 Finding optimal number of clusters...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if k_range is None:
            k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            print(f"Testing k={k}...", end=" ")
            
            # Fit K-Means
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            
            inertias.append(inertia)
            silhouette_scores.append(silhouette_avg)
            
            print(f"Silhouette: {silhouette_avg:.3f}")
        
        # Store results
        self.inertias = dict(zip(k_range, inertias))
        self.silhouette_scores = dict(zip(k_range, silhouette_scores))
        
        # Find optimal k using silhouette score (highest is best)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        self.optimal_k = optimal_k
        
        print(f"\n✅ Optimal number of clusters: {optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k, inertias, silhouette_scores
    
    def plot_cluster_selection(self) -> None:
        """
        Plot elbow curve and silhouette scores for cluster selection.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow curve
        k_values = list(self.inertias.keys())
        inertia_values = list(self.inertias.values())
        
        axes[0].plot(k_values, inertia_values, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
        axes[0].set_title('Elbow Method for Optimal k', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight optimal k
        if self.optimal_k:
            optimal_inertia = self.inertias[self.optimal_k]
            axes[0].plot(self.optimal_k, optimal_inertia, 'ro', markersize=12, 
                        label=f'Optimal k={self.optimal_k}')
            axes[0].legend()
        
        # Silhouette scores
        silhouette_values = list(self.silhouette_scores.values())
        
        axes[1].plot(k_values, silhouette_values, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Average Silhouette Score')
        axes[1].set_title('Silhouette Analysis for Optimal k', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Highlight optimal k
        if self.optimal_k:
            optimal_silhouette = self.silhouette_scores[self.optimal_k]
            axes[1].plot(self.optimal_k, optimal_silhouette, 'ro', markersize=12,
                        label=f'Optimal k={self.optimal_k}')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cluster_selection.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def fit_final_model(self, X: pd.DataFrame, n_clusters: int = None, optimal_k: int = None) -> Tuple[KMeans, np.ndarray]:
        """
        Fit the final K-Means model with optimal number of clusters.
        
        Args:
            X (pd.DataFrame): Feature matrix
            n_clusters (int): Number of clusters (uses optimal_k if not provided)
            optimal_k (int): Alternative parameter name for n_clusters (for compatibility)
            
        Returns:
            Tuple[KMeans, np.ndarray]: Fitted model and cluster labels
        """
        # Handle both parameter names for compatibility
        if optimal_k is not None:
            n_clusters = optimal_k
        elif n_clusters is None:
            n_clusters = self.optimal_k
            
        if n_clusters is None:
            raise ValueError("No optimal_k found. Run find_optimal_clusters first or provide n_clusters/optimal_k.")
            
        print(f"🎯 Training final model with k={n_clusters}...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit final model
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Calculate final metrics
        final_silhouette = silhouette_score(X_scaled, cluster_labels)
        print(f"Final silhouette score: {final_silhouette:.3f}")
        
        return self.kmeans, cluster_labels
    
    def analyze_segments(self, df: pd.DataFrame, X: pd.DataFrame, 
                        cluster_labels: np.ndarray) -> Dict:
        """
        Analyze and profile customer segments.
        
        Args:
            df (pd.DataFrame): Original customer data
            X (pd.DataFrame): Feature matrix used for clustering
            cluster_labels (np.ndarray): Cluster assignments
            
        Returns:
            Dict: Segment profiles and analysis
        """
        print("📊 Analyzing customer segments...")
        
        # Add cluster labels to dataframe
        df_analysis = df.copy()
        df_analysis['Cluster'] = cluster_labels
        
        # Calculate segment profiles
        segment_profiles = {}
        
        # Use the actual number of unique clusters from labels if optimal_k is not set
        n_clusters = self.optimal_k if self.optimal_k is not None else len(np.unique(cluster_labels))
        
        for cluster_id in range(n_clusters):
            cluster_data = df_analysis[df_analysis['Cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_analysis) * 100,
                'churn_rate': cluster_data['Exited'].mean() * 100 if 'Exited' in cluster_data.columns else None,
                'avg_age': cluster_data['Age'].mean() if 'Age' in cluster_data.columns else None,
                'avg_credit_score': cluster_data['CreditScore'].mean() if 'CreditScore' in cluster_data.columns else None,
                'avg_balance': cluster_data['Balance'].mean() if 'Balance' in cluster_data.columns else None,
                'avg_tenure': cluster_data['Tenure'].mean() if 'Tenure' in cluster_data.columns else None,
                'avg_products': cluster_data['NumOfProducts'].mean() if 'NumOfProducts' in cluster_data.columns else None,
                'avg_salary': cluster_data['EstimatedSalary'].mean() if 'EstimatedSalary' in cluster_data.columns else None,
                'active_member_pct': cluster_data['IsActiveMember'].mean() * 100 if 'IsActiveMember' in cluster_data.columns else None,
                'has_card_pct': cluster_data['HasCrCard'].mean() * 100 if 'HasCrCard' in cluster_data.columns else None,
                'characteristics': {}  # Add characteristics key for test compatibility
            }
            
            # Geography distribution
            if 'Geography' in cluster_data.columns:
                geo_dist = cluster_data['Geography'].value_counts(normalize=True) * 100
                profile['geography_distribution'] = geo_dist.to_dict()
            
            # Gender distribution
            if 'Gender' in cluster_data.columns:
                gender_dist = cluster_data['Gender'].value_counts(normalize=True) * 100
                profile['gender_distribution'] = gender_dist.to_dict()
            
            segment_profiles[f'Cluster_{cluster_id}'] = profile
        
        self.segment_profiles = segment_profiles
        
        # Print segment summary
        print("\n📋 SEGMENT PROFILES SUMMARY:")
        print("=" * 50)
        
        for cluster_name, profile in segment_profiles.items():
            print(f"\n{cluster_name}:")
            print(f"  Size: {profile['size']:,} customers ({profile['percentage']:.1f}%)")
            if profile['churn_rate'] is not None:
                print(f"  Churn Rate: {profile['churn_rate']:.1f}%")
            if profile['avg_age'] is not None:
                print(f"  Average Age: {profile['avg_age']:.1f} years")
            if profile['avg_balance'] is not None:
                print(f"  Average Balance: ${profile['avg_balance']:,.0f}")
            if profile['avg_tenure'] is not None:
                print(f"  Average Tenure: {profile['avg_tenure']:.1f} years")
        
        return segment_profiles
    
    def create_segment_visualizations(self, df: pd.DataFrame, X: pd.DataFrame, 
                                    cluster_labels: np.ndarray) -> None:
        """
        Create comprehensive visualizations for customer segments.
        
        Args:
            df (pd.DataFrame): Original customer data
            X (pd.DataFrame): Feature matrix
            cluster_labels (np.ndarray): Cluster assignments
        """
        print("📈 Creating segment visualizations...")
        
        # Add cluster labels
        df_viz = df.copy()
        df_viz['Cluster'] = cluster_labels
        
        # 1. PCA visualization
        self._plot_pca_clusters(X, cluster_labels)
        
        # 2. Segment characteristics
        self._plot_segment_characteristics(df_viz)
        
        # 3. Churn analysis by segment
        if 'Exited' in df_viz.columns:
            self._plot_churn_by_segment(df_viz)
        
        # 4. Feature distributions by segment
        self._plot_feature_distributions(df_viz, X.columns)
        
        print(f"✅ Visualizations saved to {self.figures_dir}")
    
    def _plot_pca_clusters(self, X: pd.DataFrame, cluster_labels: np.ndarray) -> None:
        """
        Create PCA visualization of clusters.
        """
        # Apply PCA for visualization
        X_scaled = self.scaler.transform(X)
        self.pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Create PCA plot
        plt.figure(figsize=(12, 8))
        
        # Plot each cluster
        colors = plt.cm.Set3(np.linspace(0, 1, self.optimal_k))
        for i in range(self.optimal_k):
            mask = cluster_labels == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
        
        # Plot cluster centers in PCA space
        centers_scaled = self.scaler.transform(self.kmeans.cluster_centers_)
        centers_pca = self.pca.transform(centers_scaled)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.xlabel(f'First Principal Component (Explained Variance: {self.pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (Explained Variance: {self.pca.explained_variance_ratio_[1]:.2%})')
        plt.title('Customer Segments - PCA Visualization', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'segments_pca.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_segment_characteristics(self, df_viz: pd.DataFrame) -> None:
        """
        Plot segment characteristics comparison.
        """
        # Calculate segment means for key metrics
        numeric_cols = ['Age', 'CreditScore', 'Balance', 'Tenure', 'NumOfProducts', 'EstimatedSalary']
        available_cols = [col for col in numeric_cols if col in df_viz.columns]
        
        segment_means = df_viz.groupby('Cluster')[available_cols].mean()
        
        # Normalize for radar chart
        segment_means_norm = segment_means.div(segment_means.max())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Segment sizes
        segment_sizes = df_viz['Cluster'].value_counts().sort_index()
        axes[0,0].pie(segment_sizes.values, labels=[f'Cluster {i}' for i in segment_sizes.index], 
                     autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Segment Sizes', fontweight='bold')
        
        # 2. Average characteristics heatmap
        sns.heatmap(segment_means_norm.T, annot=True, fmt='.2f', cmap='viridis', 
                   ax=axes[0,1], cbar_kws={'label': 'Normalized Value'})
        axes[0,1].set_title('Segment Characteristics (Normalized)', fontweight='bold')
        axes[0,1].set_xlabel('Cluster')
        
        # 3. Age distribution by segment
        if 'Age' in df_viz.columns:
            for cluster in sorted(df_viz['Cluster'].unique()):
                cluster_data = df_viz[df_viz['Cluster'] == cluster]['Age']
                axes[1,0].hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}', bins=20)
            axes[1,0].set_xlabel('Age')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Age Distribution by Segment', fontweight='bold')
            axes[1,0].legend()
        
        # 4. Balance distribution by segment
        if 'Balance' in df_viz.columns:
            df_viz.boxplot(column='Balance', by='Cluster', ax=axes[1,1])
            axes[1,1].set_title('Balance Distribution by Segment', fontweight='bold')
            axes[1,1].set_xlabel('Cluster')
            axes[1,1].set_ylabel('Balance')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'segment_characteristics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_churn_by_segment(self, df_viz: pd.DataFrame) -> None:
        """
        Plot churn analysis by segment.
        """
        # Calculate churn rates by segment
        churn_by_segment = df_viz.groupby('Cluster')['Exited'].agg(['count', 'sum', 'mean'])
        churn_by_segment['churn_rate'] = churn_by_segment['mean'] * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Churn rate by segment
        bars = axes[0].bar(churn_by_segment.index, churn_by_segment['churn_rate'])
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Churn Rate (%)')
        axes[0].set_title('Churn Rate by Customer Segment', fontweight='bold')
        
        # Add value labels
        for bar, rate in zip(bars, churn_by_segment['churn_rate']):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Stacked bar chart of churned vs retained
        churn_counts = df_viz.groupby(['Cluster', 'Exited']).size().unstack(fill_value=0)
        churn_counts.plot(kind='bar', stacked=True, ax=axes[1], color=['lightblue', 'salmon'])
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Number of Customers')
        axes[1].set_title('Customer Count by Segment and Churn Status', fontweight='bold')
        axes[1].legend(['Retained', 'Churned'])
        axes[1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'churn_by_segment.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_distributions(self, df_viz: pd.DataFrame, feature_cols: List[str]) -> None:
        """
        Plot feature distributions by segment.
        """
        # Select key features for visualization
        key_features = ['Age', 'CreditScore', 'Balance', 'NumOfProducts']
        available_features = [col for col in key_features if col in feature_cols and col in df_viz.columns]
        
        if len(available_features) < 2:
            print("⚠️  Not enough features available for distribution plots")
            return
        
        n_features = len(available_features)
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(available_features):
            row, col = i // n_cols, i % n_cols
            
            # Box plot by cluster
            df_viz.boxplot(column=feature, by='Cluster', ax=axes[row, col])
            axes[row, col].set_title(f'{feature} Distribution by Segment', fontweight='bold')
            axes[row, col].set_xlabel('Cluster')
            axes[row, col].set_ylabel(feature)
        
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'feature_distributions_by_segment.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_and_results(self, X: pd.DataFrame) -> None:
        """
        Save the trained model and analysis results.
        
        Args:
            X (pd.DataFrame): Feature matrix used for training
        """
        print("💾 Saving model and results...")
        
        # Save the trained model components
        model_artifacts = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': list(X.columns),
            'optimal_k': self.optimal_k
        }
        
        joblib.dump(model_artifacts, self.models_dir / 'customer_segmentation_model.pkl')
        
        # Save analysis results
        results = {
            'optimal_k': self.optimal_k,
            'silhouette_scores': self.silhouette_scores,
            'inertias': self.inertias,
            'segment_profiles': self.segment_profiles,
            'feature_names': list(X.columns),
            'model_performance': {
                'final_silhouette_score': self.silhouette_scores.get(self.optimal_k, None),
                'final_inertia': self.inertias.get(self.optimal_k, None)
            }
        }
        
        with open(self.reports_dir / 'segmentation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Model saved to: {self.models_dir / 'customer_segmentation_model.pkl'}")
        print(f"✅ Results saved to: {self.reports_dir / 'segmentation_results.json'}")
    
    def generate_segment_recommendations(self) -> Dict[str, List[str]]:
        """
        Generate business recommendations for each segment.
        
        Returns:
            Dict[str, List[str]]: Recommendations for each segment
        """
        recommendations = {}
        
        for cluster_name, profile in self.segment_profiles.items():
            cluster_recommendations = []
            
            # High churn rate recommendations
            if profile.get('churn_rate', 0) > 25:
                cluster_recommendations.append("🚨 High churn risk - implement immediate retention campaigns")
                cluster_recommendations.append("📞 Proactive customer outreach and support")
            
            # Low activity recommendations
            if profile.get('active_member_pct', 100) < 50:
                cluster_recommendations.append("💼 Engagement campaigns to increase activity")
                cluster_recommendations.append("🎯 Targeted product recommendations")
            
            # High value recommendations
            if profile.get('avg_balance', 0) > 100000:
                cluster_recommendations.append("💎 VIP treatment and premium services")
                cluster_recommendations.append("🏆 Exclusive offers and rewards")
            
            # Low product usage
            if profile.get('avg_products', 0) < 1.5:
                cluster_recommendations.append("📦 Cross-selling opportunities")
                cluster_recommendations.append("🎁 Product bundling incentives")
            
            # Age-based recommendations
            if profile.get('avg_age', 0) > 55:
                cluster_recommendations.append("👴 Senior-friendly services and support")
                cluster_recommendations.append("📱 Digital literacy programs")
            elif profile.get('avg_age', 0) < 35:
                cluster_recommendations.append("📱 Mobile-first digital experiences")
                cluster_recommendations.append("🌟 Modern financial products")
            
            recommendations[cluster_name] = cluster_recommendations
        
        return recommendations


def run_full_pipeline(data_path: Optional[str] = None, max_clusters: int = 8) -> CustomerSegmentation:
    """
    Run the complete customer segmentation pipeline.
    
    Args:
        data_path (str, optional): Path to input data file
        max_clusters (int): Maximum number of clusters to test
        
    Returns:
        CustomerSegmentation: Trained segmentation model
    """
    print("🚀 Starting Customer Segmentation Pipeline")
    print("=" * 50)
    
    # Initialize segmentation class
    segmenter = CustomerSegmentation()
    
    try:
        # Load data
        df = segmenter.load_data(data_path)
        
        # Select features
        X = segmenter.select_features(df)
        print(f"Feature matrix shape: {X.shape}")
        
        # Find optimal clusters
        optimal_k = segmenter.find_optimal_clusters(X, max_clusters)
        
        # Plot cluster selection analysis
        segmenter.plot_cluster_selection()
        
        # Fit final model
        model, cluster_labels = segmenter.fit_final_model(X)
        
        # Analyze segments
        segment_profiles = segmenter.analyze_segments(df, X, cluster_labels)
        
        # Create visualizations
        segmenter.create_segment_visualizations(df, X, cluster_labels)
        
        # Generate recommendations
        recommendations = segmenter.generate_segment_recommendations()
        
        print("\n💡 SEGMENT RECOMMENDATIONS:")
        print("=" * 40)
        for cluster_name, recs in recommendations.items():
            print(f"\n{cluster_name}:")
            for rec in recs:
                print(f"  • {rec}")
        
        # Save model and results
        segmenter.save_model_and_results(X)
        
        print("\n✅ Customer segmentation pipeline completed successfully!")
        return segmenter
        
    except Exception as e:
        print(f"❌ Error in segmentation pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the segmentation pipeline
    segmentation_model = run_full_pipeline()