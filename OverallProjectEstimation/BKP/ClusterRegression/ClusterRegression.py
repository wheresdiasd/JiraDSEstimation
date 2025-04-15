#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import os
import warnings
warnings.filterwarnings('ignore')

# Set Seaborn style for better visualizations
sns.set(style="whitegrid")

class ClusterRegressionModule:
    def __init__(self, output_dir="regression_results"):
        """Initialize the regression module with output directory"""
        self.output_dir = output_dir
        
        # Create directories if they don't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.plots_dir = os.path.join(output_dir, "plots")
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        
        # Storage for results
        self.model_results = {}
        
    def run(self, classified_projects_path, target_col='total_resolution_hours'):
        """Run regression models on classified projects"""
        print("\n=== CLUSTER REGRESSION MODULE ===")
        
        # Load classified data
        df = self.load_and_preprocess(classified_projects_path)
        
        # Get unique clusters and make sure they're strings
        clusters = df['project_class'].unique()
        
        # Results container
        all_results = {'global': {}, 'clusters': {}}
        
        # Train global model (baseline)
        print("\nTraining global model (all projects)...")
        global_results = self.run_regression_for_subset(df, "Global", target_col)
        all_results['global'] = global_results
        
        # Train model for each cluster
        for cluster in clusters:
            # Convert cluster to string to avoid numeric conversion issues
            cluster_str = str(cluster)
            cluster_df = df[df['project_class'] == cluster]
            
            # Skip if too few samples
            if len(cluster_df) < 10:
                print(f"Skipping Cluster {cluster_str} - insufficient data ({len(cluster_df)} projects)")
                continue
                
            print(f"\nTraining model for Cluster {cluster_str} ({len(cluster_df)} projects)...")
            cluster_results = self.run_regression_for_subset(cluster_df, f"Cluster_{cluster_str}", target_col)
            all_results['clusters'][cluster_str] = cluster_results
        
        # Compare model performance
        self.compare_models(all_results)
        
        return all_results
        
    def load_and_preprocess(self, file_path):
        """Load the classified projects and preprocess data"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Number of clusters: {df['project_class'].nunique()}")
        
        # Handle infinity values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with column median
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            
        return df
    
    def prepare_data(self, df, target_col):
        """Prepare data for modeling"""
        if target_col not in df.columns:
            available_cols = [col for col in df.columns if 'hour' in col.lower()]
            if available_cols:
                target_col = available_cols[0]
                print(f"Target column not found. Using {target_col} instead.")
            else:
                raise ValueError(f"Target column {target_col} not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove non-numeric and inappropriate columns
        exclude_patterns = ['project_id', 'id', 'project_class', 'pca_x', 'pca_y']
        X = X.select_dtypes(include=[np.number])
        X = X[[col for col in X.columns if not any(pattern in col for pattern in exclude_patterns)]]
        
        # Clean column names - replace problematic characters
        X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X.columns]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def run_regression_for_subset(self, df, label, target_col):
        """Run regression models on a specific subset of data"""
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(df, target_col)
        
        # Train models
        models = self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(models, X_test, y_test)
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['R²'])[0]
        print(f"Best model for {label}: {best_model} with R² = {results[best_model]['R²']:.4f}")
        
        # Visualize predictions for best model
        self.visualize_predictions(best_model, results[best_model], y_test, label)
        
        # Analyze feature importance
        importance_df = self.analyze_feature_importance(
            models[best_model], X_test, y_test, feature_names, label
        )
        
        # Store in instance variable for later reference
        self.model_results[label] = {
            'model_name': best_model,
            'metrics': results[best_model],
            'feature_importance': importance_df,
            'model': models[best_model]
        }
        
        return {
            'model_name': best_model,
            'metrics': results[best_model],
            'feature_importance': importance_df,
            'model': models[best_model]
        }
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        print("Training models...")
        for name, model in models.items():
            model.fit(X_train, y_train)
        
        return models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate models using multiple metrics"""
        results = {}
        print("\nModel Evaluation:")
        
        for name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MMRE
            with np.errstate(divide='ignore', invalid='ignore'):
                mre = np.abs(y_test - y_pred) / np.maximum(np.abs(y_test), 1e-10)
                valid_mres = mre[~np.isinf(mre) & ~np.isnan(mre)]
                mmre = np.mean(valid_mres) if len(valid_mres) > 0 else float('inf')
            
            # Store results
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MMRE': mmre,
                'predictions': y_pred
            }
            
            print(f"  {name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, MMRE={mmre:.4f}")
        
        return results
    
    def analyze_feature_importance(self, model, X_test, y_test, feature_names, model_label, top_n=15):
        """Analyze and visualize feature importance"""
        # Clean feature names to avoid plot errors
        feature_names = [str(name).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                         for name in feature_names]
        
        plt.figure(figsize=(12, 8))
        
        # Different approaches based on model type
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:]
            
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            
        else:
            # Use permutation importance for other models
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            importances = perm_importance.importances_mean
            indices = np.argsort(importances)[-top_n:]
            
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
        
        plt.title(f'Top {top_n} Feature Importance - {model_label}', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.plots_dir, f'importance_{model_label}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create DataFrame with importance info
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def visualize_predictions(self, model_name, results, y_test, model_label):
        """Visualize predictions vs actual values"""
        plt.figure(figsize=(12, 6))
        
        # Get predictions
        y_pred = results['predictions']
        
        # Scatter plot of actual vs predicted
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Perfect prediction line
        max_val = max(max(y_test), max(y_pred))
        min_val = min(min(y_test), min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f'{model_label} - {model_name}: Actual vs Predicted Values', fontsize=14)
        plt.xlabel('Actual', fontsize=12)
        plt.ylabel('Predicted', fontsize=12)
        
        # Add metrics text
        r2 = results['R²']
        rmse = results['RMSE']
        mae = results['MAE']
        mmre = results['MMRE']
        
        plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nMMRE = {mmre:.4f}',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                     fontsize=12, ha='left', va='top')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.plots_dir, f'predictions_{model_label}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(self, all_results):
        """Compare global model with cluster-specific models"""
        if not all_results['clusters']:
            print("No cluster models to compare with global model.")
            return
            
        # Create comparison dataframe
        comparison_data = []
        
        # Add global model
        global_metrics = all_results['global']['metrics']
        comparison_data.append({
            'Model': 'Global',
            'RMSE': global_metrics['RMSE'],
            'MAE': global_metrics['MAE'],
            'R²': global_metrics['R²'],
            'MMRE': global_metrics['MMRE']
        })
        
        # Add cluster models
        for cluster, results in all_results['clusters'].items():
            metrics = results['metrics']
            # Cluster values are already strings now
            comparison_data.append({
                'Model': f'Cluster {cluster}',
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R²': metrics['R²'],
                'MMRE': metrics['MMRE']
            })
        
        # Convert to dataframe
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate average metrics for cluster models - explicitly exclude global
        cluster_metrics = comparison_df[comparison_df['Model'] != 'Global']
        avg_metrics = {
            'RMSE': cluster_metrics['RMSE'].mean(),
            'MAE': cluster_metrics['MAE'].mean(),
            'R²': cluster_metrics['R²'].mean(),
            'MMRE': cluster_metrics['MMRE'].mean()
        }
        
        # Get global metrics
        global_row = comparison_df[comparison_df['Model'] == 'Global'].iloc[0]
        
        # Calculate improvement
        rmse_improvement = ((global_row['RMSE'] - avg_metrics['RMSE']) / global_row['RMSE']) * 100
        r2_improvement = ((avg_metrics['R²'] - global_row['R²']) / max(0.0001, abs(global_row['R²']))) * 100
        mmre_improvement = ((global_row['MMRE'] - avg_metrics['MMRE']) / max(0.0001, global_row['MMRE'])) * 100
        
        print("\nPerformance Comparison:")
        print(f"Global model: RMSE={global_row['RMSE']:.4f}, R²={global_row['R²']:.4f}, MMRE={global_row['MMRE']:.4f}")
        print(f"Cluster avg.: RMSE={avg_metrics['RMSE']:.4f}, R²={avg_metrics['R²']:.4f}, MMRE={avg_metrics['MMRE']:.4f}")
        print(f"Improvement:  RMSE={rmse_improvement:.1f}%, R²={r2_improvement:.1f}%, MMRE={mmre_improvement:.1f}%")
        
        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Make sure Model is treated as categorical, not numeric
        comparison_df_plot = comparison_df.copy()
        comparison_df_plot['Model'] = pd.Categorical(comparison_df_plot['Model'])
        
        # RMSE plot (lower is better)
        sns.barplot(data=comparison_df_plot, x='Model', y='RMSE', ax=axes[0])
        axes[0].set_title('RMSE Comparison (lower is better)')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R² plot (higher is better)
        sns.barplot(data=comparison_df_plot, x='Model', y='R²', ax=axes[1])
        axes[1].set_title('R² Comparison (higher is better)')
        axes[1].set_ylabel('R²')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MMRE plot (lower is better)
        sns.barplot(data=comparison_df_plot, x='Model', y='MMRE', ax=axes[2])
        axes[2].set_title('MMRE Comparison (lower is better)')
        axes[2].set_ylabel('MMRE')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison to CSV
        comparison_df.to_csv(os.path.join(self.plots_dir, 'model_comparison.csv'), index=False)
        
        # Add summary row
        comparison_df.loc[len(comparison_df)] = [
            'Cluster Average', 
            avg_metrics['RMSE'], 
            avg_metrics['MAE'],
            avg_metrics['R²'],
            avg_metrics['MMRE']
        ]
        
        return comparison_df


if __name__ == "__main__":
    # Example standalone usage
    regression = ClusterRegressionModule(output_dir="regression_results")
    results = regression.run("classified_projects.csv", target_col="total_resolution_hours")