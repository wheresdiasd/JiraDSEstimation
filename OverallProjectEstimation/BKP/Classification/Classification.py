#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
import warnings
warnings.filterwarnings('ignore')


class ProjectClassificationSystem:
    """
    A comprehensive system for project classification based on feature importance
    and hyperparameter tuning.
    """
    
    def __init__(self, output_dir="project_analysis"):
        """
        Initialize the project classification system.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save outputs and visualizations
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Set up subdirectories
        self.feature_dir = os.path.join(output_dir, "feature_analysis")
        self.cluster_dir = os.path.join(output_dir, "cluster_analysis")
        self.results_dir = os.path.join(output_dir, "results")
        
        for directory in [self.feature_dir, self.cluster_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def load_data(self, file_paths):
        """
        Load and combine data from CSV files.
        
        Parameters:
        -----------
        file_paths : list or str
            Path(s) to CSV file(s)
            
        Returns:
        --------
        pandas.DataFrame
            Combined DataFrame
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        dfs = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if len(dfs) == 0:
            return None
        elif len(dfs) == 1:
            return dfs[0]
        else:
            # Combine datasets (outer join to keep all data)
            combined = pd.concat(dfs, ignore_index=True)
            print(f"Combined dataset: {combined.shape[0]} rows, {combined.shape[1]} columns")
            return combined
    
    def clean_data(self, df):
        """
        Perform basic data cleaning operations.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned DataFrame
        """
        print("Starting data cleaning...")
        df_clean = df.copy()
        
        # 1. Clean up column names
        # Replace "fields." prefix
        renamed_cols = {col: col.replace('fields.', '') for col in df_clean.columns if col.startswith('fields.')}
        df_clean = df_clean.rename(columns=renamed_cols)
        
        # Replace "<lambda>" with empty string
        renamed_cols = {col: col.replace('_<lambda>', '') for col in df_clean.columns if '_<lambda>' in col}
        df_clean = df_clean.rename(columns=renamed_cols)
        
        # 2. Handle missing values in critical columns
        critical_cols = ['project_id', 'project_name', 'issue_count', 'total_resolution_hours']
        critical_cols = [col for col in critical_cols if col in df_clean.columns]
        
        missing_before = df_clean.shape[0]
        df_clean = df_clean.dropna(subset=critical_cols)
        missing_after = df_clean.shape[0]
        
        if missing_before > missing_after:
            print(f"Removed {missing_before - missing_after} rows with missing critical values")
        
        # 3. Handle missing values in numeric columns
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            missing_count = df_clean[col].isna().sum()
            if missing_count > 0:
                print(f"Filling {missing_count} missing values in {col} with median")
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # 4. Handle negative values in columns where it doesn't make sense
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['count', 'hours', 'duration']):
                neg_count = (df_clean[col] < 0).sum()
                if neg_count > 0:
                    print(f"Replacing {neg_count} negative values in {col} with 0")
                    df_clean.loc[df_clean[col] < 0, col] = 0
        
        # 5. Drop columns with all zeros or constant values
        constant_cols = []
        for col in numeric_cols:
            if df_clean[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            df_clean = df_clean.drop(columns=constant_cols)
            print(f"Dropped {len(constant_cols)} columns with constant values")
        
        print(f"Cleaning complete. Resulting dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
    
    def analyze_feature_importance(self, df, target_column='total_resolution_hours', n_estimators=100):
        """
        Analyze feature importance using Random Forest regression.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        target_column : str
            Target variable for prediction
        n_estimators : int
            Number of trees in Random Forest
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importances
        """
        print(f"\nAnalyzing feature importance for predicting {target_column}...")
        
        # Ensure target column exists
        if target_column not in df.columns:
            print(f"Target column '{target_column}' not found in dataset")
            return None
        
        # Select numeric columns for feature importance
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Exclude ID and target columns from features
        exclude_patterns = ['project_id', 'id']
        features = [col for col in numeric_cols if col != target_column and 
                   not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        # Prepare data for modeling
        X = df[features].copy()
        y = df[target_column]
        
        # Create and train random forest model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )
        
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create a DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Calculate cumulative importance
        importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
        
        # Create a horizontal bar chart for top 15 features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=top_features,
            palette='viridis'
        )
        
        plt.title(f'Top 15 Most Important Features for Predicting {target_column}', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.feature_dir, 'top_features.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create cumulative importance plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(range(1, len(importance_df) + 1), importance_df['Cumulative_Importance'], 'o-', markersize=8)
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% Importance Threshold')
        
        # Find how many features are needed for 90% importance
        features_for_90 = sum(importance_df['Cumulative_Importance'] <= 0.9) + 1
        plt.axvline(x=features_for_90, color='g', linestyle='--', 
                   label=f'{features_for_90} Features = 90% Importance')
        
        plt.title('Cumulative Feature Importance', fontsize=16)
        plt.xlabel('Number of Features', fontsize=14)
        plt.ylabel('Cumulative Importance', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.feature_dir, 'cumulative_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Identified {features_for_90} features that account for 90% of importance")
        print(f"Top 5 most important features: {', '.join(importance_df['Feature'].head(5).tolist())}")
        
        # Save feature importance to CSV
        importance_df.to_csv(os.path.join(self.feature_dir, 'feature_importance.csv'), index=False)
        
        return importance_df, X, model
    
    def get_key_features(self, importance_df, importance_threshold=0.9):
        """
        Get key features based on importance threshold.
        
        Parameters:
        -----------
        importance_df : pandas.DataFrame
            DataFrame with feature importances
        importance_threshold : float
            Threshold for cumulative importance
            
        Returns:
        --------
        list
            List of key features
        """
        # Ensure cumulative importance is calculated
        if 'Cumulative_Importance' not in importance_df.columns:
            importance_df = importance_df.sort_values('Importance', ascending=False).copy()
            importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
        
        # Get features up to threshold
        key_features = importance_df[importance_df['Cumulative_Importance'] <= importance_threshold]['Feature'].tolist()
        
        # Add one more feature to cross the threshold if needed
        if len(key_features) < len(importance_df):
            next_feature = importance_df.iloc[len(key_features)]['Feature']
            key_features.append(next_feature)
        
        return key_features
    
    def analyze_feature_correlations(self, df, key_features):
        """
        Analyze correlations between key features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        key_features : list
            List of key features to analyze
        """
        # Ensure key features exist in the DataFrame
        available_features = [f for f in key_features if f in df.columns]
        
        if len(available_features) < 2:
            print("Not enough features for correlation analysis")
            return
        
        # Limit to top 10 features to avoid overcrowded plot
        if len(available_features) > 10:
            available_features = available_features[:10]
        
        # Calculate correlation matrix
        corr_matrix = df[available_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm',
            vmin=-1, 
            vmax=1, 
            center=0,
            fmt='.2f'
        )
        
        plt.title('Correlation Between Key Features', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.feature_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analyzed correlations between {len(available_features)} key features")
    
    def find_optimal_clusters_elbow(self, data, max_k=10):
        """
        Use the elbow method to find the optimal number of clusters.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data for clustering (already scaled)
        max_k : int
            Maximum number of clusters to try
            
        Returns:
        --------
        tuple
            (inertias, optimal_k)
        """
        print("\nUsing Elbow Method to determine optimal number of clusters...")
        
        inertias = {}
        
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias[k] = kmeans.inertia_
        
        # Plot the elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(list(inertias.keys()), list(inertias.values()), 'o-', markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=14)
        plt.ylabel('Inertia', fontsize=14)
        plt.title('Elbow Method for Optimal k', fontsize=16)
        plt.xticks(range(1, max_k + 1))
        plt.grid(True, alpha=0.3)
        
        # Calculate the rate of change to find the elbow point
        k_values = list(range(1, max_k + 1))
        inertia_values = [inertias[k] for k in k_values]
        
        # Calculate the first derivatives (slopes)
        slopes = []
        for i in range(1, len(k_values)):
            slope = (inertia_values[i] - inertia_values[i-1]) / (k_values[i] - k_values[i-1])
            slopes.append(slope)
        
        # Calculate the second derivatives (changes in slope)
        slope_changes = []
        for i in range(1, len(slopes)):
            change = slopes[i] - slopes[i-1]
            slope_changes.append(change)
        
        # The elbow point is where the second derivative is maximized
        # Add 2 because we're looking at the second derivative, which is offset by 2 from the original k values
        if len(slope_changes) > 0:
            elbow_k = k_values[np.argmax(np.abs(slope_changes)) + 2]
        else:
            # Fallback if we don't have enough data points
            elbow_k = 3
        
        # Highlight the elbow point
        plt.plot(elbow_k, inertias[elbow_k], 'ro', markersize=12)
        plt.annotate(f'Elbow Point: k={elbow_k}',
                    xy=(elbow_k, inertias[elbow_k]),
                    xytext=(elbow_k + 0.5, inertias[elbow_k] * 1.1),
                    arrowprops=dict(arrowstyle='->', lw=1.5),
                    fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.cluster_dir, 'elbow_method.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Elbow method suggests optimal number of clusters: {elbow_k}")
        
        return inertias, elbow_k
    
    def find_optimal_clusters_silhouette(self, data, max_k=10):
        """
        Use silhouette analysis to find the optimal number of clusters.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data for clustering (already scaled)
        max_k : int
            Maximum number of clusters to try
            
        Returns:
        --------
        tuple
            (silhouette_scores, optimal_k)
        """
        print("\nUsing Silhouette Analysis to determine optimal number of clusters...")
        
        silhouette_scores = {}
        
        # Must have at least 2 clusters for silhouette analysis
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores[k] = silhouette_avg
        
        # Plot the silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), 'o-', markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=14)
        plt.ylabel('Silhouette Score', fontsize=14)
        plt.title('Silhouette Analysis for Optimal k', fontsize=16)
        plt.xticks(range(2, max_k + 1))
        plt.grid(True, alpha=0.3)
        
        # Find the k with maximum silhouette score
        optimal_k = max(silhouette_scores.items(), key=lambda x: x[1])[0]
        
        # Highlight the optimal point
        plt.plot(optimal_k, silhouette_scores[optimal_k], 'ro', markersize=12)
        plt.annotate(f'Optimal k={optimal_k}',
                    xy=(optimal_k, silhouette_scores[optimal_k]),
                    xytext=(optimal_k + 0.5, silhouette_scores[optimal_k] * 0.95),
                    arrowprops=dict(arrowstyle='->', lw=1.5),
                    fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.cluster_dir, 'silhouette_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Silhouette analysis suggests optimal number of clusters: {optimal_k}")
        
        return silhouette_scores, optimal_k
    
    def find_optimal_clusters_davies_bouldin(self, data, max_k=10):
        """
        Use the Davies-Bouldin index to find the optimal number of clusters.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data for clustering (already scaled)
        max_k : int
            Maximum number of clusters to try
            
        Returns:
        --------
        tuple
            (db_scores, optimal_k)
        """
        print("\nUsing Davies-Bouldin Index to determine optimal number of clusters...")
        
        db_scores = {}
        
        # Need at least 2 clusters for Davies-Bouldin index
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            
            # Calculate Davies-Bouldin index
            db_index = davies_bouldin_score(data, labels)
            db_scores[k] = db_index
        
        # Plot the Davies-Bouldin indices
        plt.figure(figsize=(10, 6))
        plt.plot(list(db_scores.keys()), list(db_scores.values()), 'o-', markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=14)
        plt.ylabel('Davies-Bouldin Index', fontsize=14)
        plt.title('Davies-Bouldin Index for Optimal k', fontsize=16)
        plt.xticks(range(2, max_k + 1))
        plt.grid(True, alpha=0.3)
        
        # Find the k with minimum Davies-Bouldin index
        optimal_k = min(db_scores.items(), key=lambda x: x[1])[0]
        
        # Highlight the optimal point
        plt.plot(optimal_k, db_scores[optimal_k], 'ro', markersize=12)
        plt.annotate(f'Optimal k={optimal_k}',
                    xy=(optimal_k, db_scores[optimal_k]),
                    xytext=(optimal_k + 0.5, db_scores[optimal_k] * 0.9),
                    arrowprops=dict(arrowstyle='->', lw=1.5),
                    fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.cluster_dir, 'davies_bouldin.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Davies-Bouldin index suggests optimal number of clusters: {optimal_k}")
        
        return db_scores, optimal_k
    
    def determine_optimal_clusters(self, data, max_k=10):
        """
        Determine the optimal number of clusters using multiple methods.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data for clustering (already scaled)
        max_k : int
            Maximum number of clusters to try
            
        Returns:
        --------
        int
            Recommended optimal number of clusters
        """
        # Apply all methods
        _, elbow_k = self.find_optimal_clusters_elbow(data, max_k)
        _, silhouette_k = self.find_optimal_clusters_silhouette(data, max_k)
        _, db_k = self.find_optimal_clusters_davies_bouldin(data, max_k)
        
        methods = {
            'Elbow Method': elbow_k,
            'Silhouette Analysis': silhouette_k,
            'Davies-Bouldin Index': db_k
        }
        
        # Print the recommendations from each method
        print("\nCluster number recommendations:")
        for method, k in methods.items():
            print(f"  {method}: k = {k}")
        
        # Make a final recommendation
        from collections import Counter
        k_values = list(methods.values())
        most_common_k = Counter(k_values).most_common(1)[0][0]
        
        # If there's a tie and silhouette recommends a value, use that
        if most_common_k != silhouette_k and k_values.count(most_common_k) == k_values.count(silhouette_k):
            final_k = silhouette_k
        else:
            final_k = most_common_k
        
        print(f"\nRecommended optimal number of clusters: {final_k}")
        
        return final_k
    
    def tune_kmeans_hyperparameters(self, data, k):
        """
        Tune KMeans hyperparameters for a given number of clusters.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data for clustering (already scaled)
        k : int
            Number of clusters
            
        Returns:
        --------
        dict
            Best hyperparameters
        """
        print(f"\nTuning KMeans hyperparameters for k={k}...")
        
        # Define hyperparameter grid
        param_grid = {
            'init': ['k-means++', 'random'],
            'n_init': [10, 20],
            'max_iter': [300, 500]
        }
        
        # Generate all combinations of parameters
        import itertools
        all_params = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
        
        best_score = -1
        best_params = None
        
        print(f"Testing {len(all_params)} hyperparameter combinations...")
        
        # Test each combination
        for params in all_params:
            kmeans = KMeans(n_clusters=k, random_state=42, **params)
            labels = kmeans.fit_predict(data)
            
            # Evaluate using silhouette score
            score = silhouette_score(data, labels)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        print(f"Best hyperparameters: {best_params}")
        print(f"Silhouette score with best parameters: {best_score:.4f}")
        
        return best_params
    
    def classify_projects(self, df, key_features, optimal_k, best_params=None):
        """
        Classify projects into clusters using key features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        key_features : list
            List of key features to use for clustering
        optimal_k : int
            Optimal number of clusters
        best_params : dict, optional
            Best hyperparameters for KMeans
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cluster assignments
        """
        print(f"\nClassifying projects into {optimal_k} clusters...")
        
        # Make a copy of the dataframe
        df_classified = df.copy()
        
        # Select only key features that exist in the dataframe
        available_key_features = [f for f in key_features if f in df.columns]
        
        # Prepare data for clustering
        cluster_data = df[available_key_features].copy()
        
        # Handle missing values
        cluster_data = cluster_data.fillna(cluster_data.median())
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Apply KMeans with optimal parameters
        if best_params is None:
            best_params = {'init': 'k-means++', 'n_init': 10, 'max_iter': 300}
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, **best_params)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster assignments to the DataFrame
        df_classified['project_class'] = clusters
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Add PCA components for visualization
        df_classified['pca_x'] = pca_result[:, 0]
        df_classified['pca_y'] = pca_result[:, 1]
        
        # Visualize the clusters
        plt.figure(figsize=(12, 10))
        
        # Colors for clusters
        from matplotlib.colors import LinearSegmentedColormap
        colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
        
        # Plot each cluster
        for i in range(optimal_k):
            cluster_points = df_classified[df_classified['project_class'] == i]
            plt.scatter(
                cluster_points['pca_x'], 
                cluster_points['pca_y'],
                s=100, 
                alpha=0.7,
                c=[colors[i]],
                label=f'Class {i} ({len(cluster_points)} projects)'
            )
            
            # Add some project names as labels
            if len(cluster_points) > 0:
                sample_size = min(3, len(cluster_points))
                for _, row in cluster_points.sample(sample_size).iterrows():
                    plt.annotate(
                        row['project_name'],
                        (row['pca_x'], row['pca_y']),
                        fontsize=9,
                        alpha=0.8
                    )
        
        plt.title('Project Classification Using Key Features', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(self.cluster_dir, 'project_classes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze each cluster
        cluster_profiles = self.analyze_clusters(df_classified, available_key_features)
        
        # Save the classified data
        df_classified.to_csv(os.path.join(self.results_dir, 'classified_projects.csv'), index=False)
        
        # Save a summary of cluster memberships
        cluster_counts = df_classified['project_class'].value_counts().sort_index()
        cluster_pcts = (cluster_counts / len(df_classified) * 100).round(1)
        
        summary_df = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Projects': cluster_counts.values,
            'Percentage': cluster_pcts.values
        })
        
        summary_df.to_csv(os.path.join(self.results_dir, 'cluster_summary.csv'), index=False)
        
        print(f"Project classification complete. Results saved to {self.results_dir}")
        
        return df_classified, cluster_profiles
    def create_team_metrics(self, df):
        """
        Create advanced team-related metrics from basic team data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added team metrics
        """
        df_enhanced = df.copy()
        
        # Find team-related columns
        creator_cols = [col for col in df.columns if 'creator' in col.lower()]
        reporter_cols = [col for col in df.columns if 'reporter' in col.lower()]
        assignee_cols = [col for col in df.columns if 'assignee' in col.lower()]
        
        # Basic team size metrics
        if creator_cols:
            df_enhanced['creator_count'] = df[creator_cols].max(axis=1)
        
        if reporter_cols:
            df_enhanced['reporter_count'] = df[reporter_cols].max(axis=1)
        
        if assignee_cols:
            df_enhanced['assignee_count'] = df[assignee_cols].max(axis=1)
        
        # Combined team size estimate (taking the maximum of available metrics)
        team_cols = ['creator_count', 'reporter_count', 'assignee_count']
        available_team_cols = [col for col in team_cols if col in df_enhanced.columns]
        
        if available_team_cols:
            df_enhanced['team_size_estimate'] = df_enhanced[available_team_cols].max(axis=1)
            
            # Team productivity metrics
            if 'issue_count' in df.columns:
                df_enhanced['issues_per_team_member'] = df['issue_count'] / df_enhanced['team_size_estimate'].replace(0, 1)
            
            if 'total_resolution_hours' in df.columns:
                df_enhanced['resolution_hours_per_team_member'] = df['total_resolution_hours'] / df_enhanced['team_size_estimate'].replace(0, 1)
        
        # Team diversity approximation
        if len(available_team_cols) > 1:
            # Calculate the ratio between different types of team members
            # This could indicate how evenly distributed the team roles are
            df_enhanced['team_role_diversity'] = df_enhanced[available_team_cols].std(axis=1) / df_enhanced[available_team_cols].mean(axis=1).replace(0, 1)
        
        print(f"Added team metrics: {[col for col in df_enhanced.columns if col not in df.columns]}")
        
        return df_enhanced
    
    def analyze_clusters(self, df_classified, key_features):
        """
        Analyze the characteristics of each cluster.
        
        Parameters:
        -----------
        df_classified : pandas.DataFrame
            DataFrame with cluster assignments
        key_features : list
            List of key features used for clustering
            
        Returns:
        --------
        dict
            Dictionary with cluster profiles
        """
        if 'project_class' not in df_classified.columns:
            print("No cluster assignments found in DataFrame")
            return {}
        
        # Create profiles for each cluster
        cluster_profiles = {}
        
        # Get all numeric columns
        numeric_cols = df_classified.select_dtypes(include=np.number).columns
        
        # Remove cluster-related columns
        analysis_cols = [col for col in numeric_cols if col not in ['project_class', 'pca_x', 'pca_y']]
        
        # Calculate overall metrics for comparison
        overall_means = df_classified[analysis_cols].mean()
        
        # Analyze each cluster
        for cluster in sorted(df_classified['project_class'].unique()):
            cluster_data = df_classified[df_classified['project_class'] == cluster]
            
            # Basic statistics
            profile = {
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(df_classified) * 100, 1),
                'sample_projects': cluster_data['project_name'].sample(min(5, len(cluster_data))).tolist(),
                'features': {}
            }
            
            # Calculate statistics for key features
            for feature in key_features:
                if feature in cluster_data.columns:
                    cluster_mean = cluster_data[feature].mean()
                    overall_mean = overall_means[feature]
                    
                    # Calculate the percentage difference from overall mean
                    if overall_mean != 0:
                        pct_diff = ((cluster_mean - overall_mean) / overall_mean) * 100
                    else:
                        pct_diff = 0 if cluster_mean == 0 else float('inf')
                    
                    profile['features'][feature] = {
                        'mean': cluster_mean,
                        'median': cluster_data[feature].median(),
                        'min': cluster_data[feature].min(),
                        'max': cluster_data[feature].max(),
                        'pct_diff': pct_diff
                    }
            
            # Sort features by absolute percentage difference to identify distinguishing features
            sorted_features = sorted(
                [(f, profile['features'][f]['pct_diff']) for f in profile['features']],
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Keep top 5 distinguishing features
            top_features = sorted_features[:5]
            
            # Generate a description based on distinguishing features
            description_parts = []
            for feature, pct_diff in top_features:
                direction = "higher" if pct_diff > 0 else "lower"
                description_parts.append(f"{feature} is {abs(pct_diff):.1f}% {direction} than average")
            
            profile['distinguishing_features'] = top_features
            profile['description'] = "Projects where " + ", ".join(description_parts)
            
            # Determine cluster characteristics based on key metrics
            if 'issue_count' in cluster_data.columns:
                issue_count_mean = cluster_data['issue_count'].mean()
                
                if issue_count_mean > 10000:
                    size_label = "Large"
                elif issue_count_mean > 1000:
                    size_label = "Medium"
                else:
                    size_label = "Small"
                
                profile['size_label'] = size_label
            
            if 'total_resolution_hours' in cluster_data.columns:
                hours_mean = cluster_data['total_resolution_hours'].mean()
                
                if hours_mean > 1000000:
                    effort_label = "High-effort"
                elif hours_mean > 100000:
                    effort_label = "Medium-effort"
                else:
                    effort_label = "Low-effort"
                
                profile['effort_label'] = effort_label
            
            # Add cluster to profiles
            cluster_name = f"Cluster {cluster}"
            if 'size_label' in profile and 'effort_label' in profile:
                cluster_name = f"{profile['size_label']} {profile['effort_label']} Projects (Cluster {cluster})"
            
            cluster_profiles[cluster_name] = profile
            
            # Print cluster information
            print(f"\n{cluster_name} ({profile['size']} projects, {profile['percentage']}%):")
            print(f"  Sample projects: {', '.join(profile['sample_projects'][:3])}")
            print(f"  Distinguishing characteristics:")
            for feature, pct_diff in top_features:
                direction = "higher" if pct_diff > 0 else "lower"
                print(f"    - {feature}: {abs(pct_diff):.1f}% {direction} than average")
        
        # Save cluster profiles to JSON
        import json
        
        # Convert to serializable format
        serializable_profiles = {}
        for cluster, profile in cluster_profiles.items():
            serializable_profiles[cluster] = {
                'size': int(profile['size']),
                'percentage': float(profile['percentage']),
                'sample_projects': profile['sample_projects'],
                'description': profile['description']
            }
            
            if 'features' in profile:
                serializable_profiles[cluster]['features'] = {}
                for feature, stats in profile['features'].items():
                    serializable_profiles[cluster]['features'][feature] = {
                        stat: float(value) if isinstance(value, (int, float, np.number)) else value
                        for stat, value in stats.items()
                    }
        
        with open(os.path.join(self.results_dir, 'cluster_profiles.json'), 'w') as f:
            json.dump(serializable_profiles, f, indent=2)
        
        return cluster_profiles
    
    def execute_classification_pipeline(self, file_paths, target_column='total_resolution_hours', max_k=10):
        """
        Execute the complete project classification pipeline.
        
        Parameters:
        -----------
        file_paths : list or str
            Path(s) to CSV file(s)
        target_column : str
            Target variable for feature importance analysis
        max_k : int
            Maximum number of clusters to consider
            
        Returns:
        --------
        tuple
            (classified_df, optimal_k, cluster_profiles)
        """
        print("\n========== PROJECT CLASSIFICATION PIPELINE ==========\n")
        
        # Step 1: Load data
        df = self.load_data(file_paths)
        if df is None:
            print("Failed to load data. Exiting pipeline.")
            return None, None, None
        
        # Step 2: Clean data
        df_clean = self.clean_data(df)

        df_clean = self.create_team_metrics(df_clean)

        # Step 3: Analyze feature importance
        importance_df, X, model = self.analyze_feature_importance(df_clean, target_column)
    
        
        # Step 4: Get key features for classification
        key_features = self.get_key_features(importance_df)
        
        # Step 5: Analyze feature correlations
        self.analyze_feature_correlations(df_clean, key_features)
        
        # Step 6: Prepare data for clustering
        # Select only key features that exist in the dataframe
        available_key_features = [f for f in key_features if f in df_clean.columns]
        
        # Prepare data for clustering
        cluster_data = df_clean[available_key_features].copy()
        
        # Handle missing values
        cluster_data = cluster_data.fillna(cluster_data.median())
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Step 7: Determine optimal number of clusters
        optimal_k = self.determine_optimal_clusters(scaled_data, max_k)
        
        # Step 8: Tune KMeans hyperparameters
        best_params = self.tune_kmeans_hyperparameters(scaled_data, optimal_k)
        
        # Step 9: Classify projects
        df_classified, cluster_profiles = self.classify_projects(df_clean, key_features, optimal_k, best_params)
        
        print("\n============= PIPELINE COMPLETE =============\n")
        
        return df_classified, optimal_k, cluster_profiles

# Example usage
if __name__ == "__main__":
    classifier = ProjectClassificationSystem(output_dir="project_classification_results")
    
    df_classified, optimal_k, cluster_profiles = classifier.execute_classification_pipeline(
        ["../DataSets/data_export_1741772203780.csv", "../DataSets/data_export_1741699774916.csv"],
        target_column='total_resolution_hours',
        max_k=10
    )

