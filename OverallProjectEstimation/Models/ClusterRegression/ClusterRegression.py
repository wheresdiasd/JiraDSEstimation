#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fix for the compare_models function in the regression module
# Replace the compare_models function with this version

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
    for cluster_id, results in all_results['clusters'].items():
        metrics = results['metrics']
        # Convert cluster id to string to avoid numeric conversion issues
        cluster_name = f'Cluster_{cluster_id}'
        comparison_data.append({
            'Model': cluster_name,  
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R²': metrics['R²'],
            'MMRE': metrics['MMRE']
        })
    
    # Convert to dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate average metrics for cluster models
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
    
    # RMSE plot (lower is better)
    sns.barplot(data=comparison_df, x='Model', y='RMSE', ax=axes[0])
    axes[0].set_title('RMSE Comparison (lower is better)')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # R² plot (higher is better)
    sns.barplot(data=comparison_df, x='Model', y='R²', ax=axes[1])
    axes[1].set_title('R² Comparison (higher is better)')
    axes[1].set_ylabel('R²')
    axes[1].tick_params(axis='x', rotation=45)
    
    # MMRE plot (lower is better)
    sns.barplot(data=comparison_df, x='Model', y='MMRE', ax=axes[2])
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

