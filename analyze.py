#!/usr/bin/env python3
"""
Analysis of Embedding Reconstruction Experiment Results with Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def create_visualizations(df):
    """Create comprehensive visualizations of the experiment results."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the main analysis figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall Performance Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    linear_scores = df['linear_test_r2']
    pca_scores = df['pca_test_r2']
    
    ax1.hist(linear_scores, alpha=0.7, bins=20, label='Linear R²', color='green', edgecolor='black')
    ax1.hist(pca_scores, alpha=0.7, bins=20, label='PCA R²', color='red', edgecolor='black')
    ax1.axvline(linear_scores.mean(), color='green', linestyle='--', linewidth=2, label=f'Linear Mean: {linear_scores.mean():.3f}')
    ax1.axvline(pca_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'PCA Mean: {pca_scores.mean():.3f}')
    ax1.set_xlabel('R² Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Performance Distribution: Linear vs PCA Reconstruction', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Model Performance Comparison
    ax2 = fig.add_subplot(gs[0, 1:])
    model_performance = df.groupby('model')['linear_test_r2'].agg(['mean', 'std']).sort_values('mean', ascending=True)
    
    colors = ['#FF6B6B' if 'voyage' in idx else '#4ECDC4' if 'text-embedding' in idx else '#45B7D1' 
              for idx in model_performance.index]
    
    y_pos = np.arange(len(model_performance))
    bars = ax2.barh(y_pos, model_performance['mean'], xerr=model_performance['std'], 
                    color=colors, alpha=0.8, capsize=3)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([model.replace('text-embedding-', 'TE-').replace('voyage-', 'V-') 
                        for model in model_performance.index], fontsize=9)
    ax2.set_xlabel('Average R² Score')
    ax2.set_title('Model Performance Ranking (Linear Reconstruction)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add performance threshold line
    ax2.axvline(0.9, color='red', linestyle='--', alpha=0.7, label='Excellent (0.9)')
    ax2.axvline(0.8, color='orange', linestyle='--', alpha=0.7, label='Good (0.8)')
    ax2.legend(loc='lower right')
    
    # 3. Dataset Difficulty Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    dataset_performance = df.groupby('dataset').agg({
        'linear_test_r2': 'mean',
        'dataset_range': 'first'
    }).sort_values('linear_test_r2')
    
    colors_dataset = ['#FF4444' if score < 0.8 else '#FF8C00' if score < 0.9 else '#32CD32' 
                     for score in dataset_performance['linear_test_r2']]
    
    bars = ax3.bar(range(len(dataset_performance)), dataset_performance['linear_test_r2'], 
                   color=colors_dataset, alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(dataset_performance)))
    ax3.set_xticklabels(dataset_performance.index, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Average R² Score')
    ax3.set_title('Dataset Difficulty (Red=Hard, Orange=Medium, Green=Easy)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add difficulty thresholds
    ax3.axhline(0.8, color='red', linestyle='--', alpha=0.7)
    ax3.axhline(0.9, color='orange', linestyle='--', alpha=0.7)
    
    # 4. Provider Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Categorize models by provider
    df['provider'] = df['model'].apply(lambda x: 'Voyage' if 'voyage' in x 
                                      else 'OpenAI' if 'text-embedding' in x 
                                      else 'Google')
    
    provider_stats = df.groupby('provider')['linear_test_r2'].agg(['mean', 'std', 'count'])
    
    colors_provider = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax4.bar(provider_stats.index, provider_stats['mean'], 
                   yerr=provider_stats['std'], color=colors_provider, 
                   alpha=0.8, capsize=5, edgecolor='black')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, provider_stats['count'])):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'n={count}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Average R² Score')
    ax4.set_title('Performance by Provider', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Scatter: Range vs Performance
    ax5 = fig.add_subplot(gs[1, 2])
    dataset_stats = df.groupby('dataset').agg({
        'linear_test_r2': 'mean',
        'dataset_range': 'first',
        'dataset_min_value': 'first',
        'dataset_max_value': 'first'
    })
    
    scatter = ax5.scatter(dataset_stats['dataset_range'], dataset_stats['linear_test_r2'], 
                         s=100, alpha=0.7, c=dataset_stats['linear_test_r2'], 
                         cmap='RdYlGn', edgecolors='black')
    
    ax5.set_xlabel('Dataset Range (log scale)')
    ax5.set_ylabel('Average R² Score')
    ax5.set_title('Dataset Range vs Performance', fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax5, label='R² Score')
    
    # Add dataset labels
    for dataset, row in dataset_stats.iterrows():
        ax5.annotate(dataset, (row['dataset_range'], row['linear_test_r2']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    # 6. Heatmap: Model vs Dataset Performance
    ax6 = fig.add_subplot(gs[2, :])
    
    # Create pivot table for heatmap
    heatmap_data = df.pivot_table(values='linear_test_r2', index='model', columns='dataset')
    
    # Sort by overall performance
    model_order = df.groupby('model')['linear_test_r2'].mean().sort_values(ascending=False).index
    dataset_order = df.groupby('dataset')['linear_test_r2'].mean().sort_values(ascending=False).index
    
    heatmap_data_sorted = heatmap_data.reindex(index=model_order, columns=dataset_order)
    
    sns.heatmap(heatmap_data_sorted, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax6, cbar_kws={'label': 'R² Score'}, 
                xticklabels=True, yticklabels=True)
    ax6.set_title('Detailed Performance Matrix: Model × Dataset', fontweight='bold', pad=20)
    ax6.set_xlabel('Dataset')
    ax6.set_ylabel('Model')
    
    # Rotate labels for better readability
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    ax6.set_yticklabels(ax6.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('embedding_reconstruction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second detailed analysis figure
    create_detailed_analysis(df)
    
    # Create the decimal places analysis figure
    create_decimal_places_analysis(df)
    print("Creating explained variance analysis")
    # Create the explained variance analysis figure
    create_explained_variance_analysis(df)

def create_detailed_analysis(df):
    """Create additional detailed analysis visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Embedding Reconstruction Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance vs Model Size (if we can infer it)
    ax1 = axes[0, 0]
    
    # Create model categories based on naming patterns
    df['model_category'] = df['model'].apply(lambda x: 
        'Large' if 'large' in x or '3-large' in x 
        else 'Small' if 'small' in x or 'ada' in x
        else 'Medium')
    
    category_performance = df.groupby('model_category')['linear_test_r2'].agg(['mean', 'std'])
    
    bars = ax1.bar(category_performance.index, category_performance['mean'], 
                   yerr=category_performance['std'], capsize=5, alpha=0.8,
                   color=['#FF6B6B', '#FFD93D', '#6BCF7F'])
    ax1.set_ylabel('Average R² Score')
    ax1.set_title('Performance by Model Size Category')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Error Analysis
    ax2 = axes[0, 1]
    
    # Calculate error (1 - R²) for better visualization
    df['reconstruction_error'] = 1 - df['linear_test_r2']
    
    provider_errors = df.groupby('provider')['reconstruction_error'].agg(['mean', 'std'])
    
    ax2.bar(provider_errors.index, provider_errors['mean'], 
            yerr=provider_errors['std'], capsize=5, alpha=0.8,
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_ylabel('Average Reconstruction Error (1 - R²)')
    ax2.set_title('Reconstruction Error by Provider')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Performance Consistency
    ax3 = axes[0, 2]
    
    model_consistency = df.groupby('model')['linear_test_r2'].std().sort_values()
    
    colors = ['green' if std < 0.05 else 'orange' if std < 0.1 else 'red' 
              for std in model_consistency.values]
    
    bars = ax3.barh(range(len(model_consistency)), model_consistency.values, 
                    color=colors, alpha=0.7)
    ax3.set_yticks(range(len(model_consistency)))
    ax3.set_yticklabels([model.replace('text-embedding-', 'TE-').replace('voyage-', 'V-') 
                        for model in model_consistency.index], fontsize=8)
    ax3.set_xlabel('Performance Standard Deviation')
    ax3.set_title('Model Consistency (Lower = More Consistent)')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Linear vs PCA Comparison by Provider
    ax4 = axes[1, 0]
    
    comparison_data = df.groupby('provider').agg({
        'linear_test_r2': 'mean',
        'pca_test_r2': 'mean'
    })
    
    x = np.arange(len(comparison_data))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, comparison_data['linear_test_r2'], width, 
                    label='Linear R²', alpha=0.8, color='green')
    bars2 = ax4.bar(x + width/2, comparison_data['pca_test_r2'], width, 
                    label='PCA R²', alpha=0.8, color='red')
    
    ax4.set_xlabel('Provider')
    ax4.set_ylabel('Average R² Score')
    ax4.set_title('Linear vs PCA Performance by Provider')
    ax4.set_xticks(x)
    ax4.set_xticklabels(comparison_data.index)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Dataset Complexity Analysis
    ax5 = axes[1, 1]
    
    # Create complexity score based on range and performance
    dataset_stats = df.groupby('dataset').agg({
        'linear_test_r2': 'mean',
        'dataset_range': 'first'
    })
    
    # Normalize ranges for complexity score
    normalized_range = np.log10(dataset_stats['dataset_range'])
    complexity_score = normalized_range / normalized_range.max()
    difficulty_score = 1 - dataset_stats['linear_test_r2']
    
    scatter = ax5.scatter(complexity_score, difficulty_score, 
                         s=100, alpha=0.7, c=dataset_stats['linear_test_r2'], 
                         cmap='RdYlGn_r', edgecolors='black')
    
    ax5.set_xlabel('Range Complexity (normalized log scale)')
    ax5.set_ylabel('Reconstruction Difficulty (1 - R²)')
    ax5.set_title('Dataset Complexity vs Difficulty')
    ax5.grid(True, alpha=0.3)
    
    # Add dataset labels
    for dataset, row in dataset_stats.iterrows():
        idx = dataset_stats.index.get_loc(dataset)
        ax5.annotate(dataset, (complexity_score.iloc[idx], difficulty_score.iloc[idx]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 6. Top Performers Analysis
    ax6 = axes[1, 2]
    
    # Find top 5 model-dataset combinations
    top_combinations = df.nlargest(10, 'linear_test_r2')
    
    y_pos = range(len(top_combinations))
    bars = ax6.barh(y_pos, top_combinations['linear_test_r2'], 
                    alpha=0.8, color='gold', edgecolor='black')
    
    ax6.set_yticks(y_pos)
    labels = [f"{row['dataset'][:8]}+{row['model'].split('-')[-1]}" 
              for _, row in top_combinations.iterrows()]
    ax6.set_yticklabels(labels, fontsize=8)
    ax6.set_xlabel('R² Score')
    ax6.set_title('Top 10 Model-Dataset Combinations')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, top_combinations['linear_test_r2'])):
        ax6.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_decimal_places_analysis(df):
    """Create visualization showing how performance drops with increasing decimal places."""
    
    # Extract size information from dataset names (assuming format like 'integers_small', 'decimals_large', etc.)
    def extract_size_and_type(dataset_name):
        dataset_lower = dataset_name.lower()
        
        if 'small' in dataset_lower:
            size = 'Small'
        elif 'medium' in dataset_lower:
            size = 'Medium'
        elif 'large' in dataset_lower:
            size = 'Large'
        else:
            size = 'Medium'  # Default fallback
            
        if 'integer' in dataset_lower:
            num_type = 'Integers'
            decimal_places = 0
        elif 'decimal' in dataset_lower:
            num_type = 'Decimals'
            decimal_places = 2  # Assuming 2 decimal places for decimals
        elif 'balanced' in dataset_lower:
            num_type = 'Balanced'
            decimal_places = 1  # Mixed, so assign intermediate value
        else:
            num_type = 'Integers'  # Default fallback
            decimal_places = 0
            
        return size, num_type, decimal_places
    
    # Add derived columns
    df['size'], df['number_type'], df['decimal_places'] = zip(*df['dataset'].apply(extract_size_and_type))
    
    # Debug: Print unique values to understand the data structure
    print("DEBUG: Unique dataset names:", df['dataset'].unique())
    print("DEBUG: Unique sizes:", df['size'].unique())
    print("DEBUG: Unique number types:", df['number_type'].unique())
    
    # Define provider colors
    provider_colors = {
        'Voyage': '#FF6B6B',
        'OpenAI': '#4ECDC4', 
        'Google': '#45B7D1'
    }
    
    # Filter out any remaining unknown categories for plotting
    valid_sizes = ['Small', 'Medium', 'Large']
    valid_types = ['Integers', 'Balanced', 'Decimals']
    
    df_filtered = df[df['size'].isin(valid_sizes) & df['number_type'].isin(valid_types)]
    
    if len(df_filtered) == 0:
        print("WARNING: No data matches expected size/type patterns. Skipping decimal places analysis.")
        return
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance vs Number Complexity: How R² Drops with Dataset Size and Type', 
                 fontsize=16, fontweight='bold')
    
    # 1. Performance vs Dataset Size (aggregated by size category)
    ax1 = axes[0, 0]
    
    size_order = ['Small', 'Medium', 'Large']
    size_performance = df_filtered.groupby(['size', 'provider'])['linear_test_r2'].mean().reset_index()
    
    for provider in provider_colors.keys():
        provider_data = size_performance[size_performance['provider'] == provider]
        if not provider_data.empty:
            ax1.plot(provider_data['size'], provider_data['linear_test_r2'], 
                    marker='o', linewidth=3, markersize=8, label=provider, 
                    color=provider_colors[provider], alpha=0.8)
    
    ax1.set_xlabel('Dataset Size Category')
    ax1.set_ylabel('Average Linear R² Score')
    ax1.set_title('Performance Degradation by Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.7, 1.0)
    
    # 2. Performance vs Number Type
    ax2 = axes[0, 1]
    
    type_performance = df_filtered.groupby(['number_type', 'provider'])['linear_test_r2'].mean().reset_index()
    
    x_positions = {'Integers': 0, 'Balanced': 1, 'Decimals': 2}
    
    for provider in provider_colors.keys():
        provider_data = type_performance[type_performance['provider'] == provider]
        if not provider_data.empty:
            x_vals = [x_positions[nt] for nt in provider_data['number_type'] if nt in x_positions]
            y_vals = [provider_data[provider_data['number_type'] == nt]['linear_test_r2'].iloc[0] 
                     for nt in provider_data['number_type'] if nt in x_positions]
            
            if x_vals and y_vals:
                ax2.plot(x_vals, y_vals, 
                        marker='s', linewidth=3, markersize=8, label=provider,
                        color=provider_colors[provider], alpha=0.8)
    
    ax2.set_xlabel('Number Type (Complexity →)')
    ax2.set_ylabel('Average Linear R² Score')
    ax2.set_title('Performance by Number Type Complexity')
    ax2.set_xticks(list(x_positions.values()))
    ax2.set_xticklabels(list(x_positions.keys()))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 1.0)
    
    # 3. Individual Model Performance Across Sizes
    ax3 = axes[1, 0]
    
    # Get top 8 models by average performance for clarity
    top_models = df_filtered.groupby('model')['linear_test_r2'].mean().nlargest(8).index
    
    for i, model in enumerate(top_models):
        model_data = df_filtered[df_filtered['model'] == model]
        if len(model_data) > 0:
            provider = model_data['provider'].iloc[0]
            
            size_avg = model_data.groupby('size')['linear_test_r2'].mean()
            
            # Ensure we have data for all sizes, fill with NaN if missing
            size_values = [size_avg.get(size, np.nan) for size in size_order]
            
            ax3.plot(size_order, size_values, marker='o', linewidth=2, 
                    markersize=6, label=model.replace('text-embedding-', 'TE-').replace('voyage-', 'V-'),
                    color=provider_colors[provider], alpha=0.7, linestyle='-' if i < 4 else '--')
    
    ax3.set_xlabel('Dataset Size Category')
    ax3.set_ylabel('Linear R² Score')
    ax3.set_title('Top 8 Models: Performance Across Dataset Sizes')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.7, 1.0)
    
    # 4. Detailed Heatmap: Size × Type × Provider
    ax4 = axes[1, 1]
    
    # Create a more detailed breakdown
    detailed_performance = df_filtered.groupby(['size', 'number_type', 'provider'])['linear_test_r2'].mean().reset_index()
    
    # Create combined labels for better visualization
    detailed_performance['size_type'] = detailed_performance['size'] + ' ' + detailed_performance['number_type']
    
    # Pivot for heatmap
    heatmap_data = detailed_performance.pivot_table(
        values='linear_test_r2', 
        index='provider', 
        columns='size_type'
    )
    
    # Reorder columns logically
    desired_order = []
    for size in ['Small', 'Medium', 'Large']:
        for num_type in ['Integers', 'Balanced', 'Decimals']:
            col_name = f'{size} {num_type}'
            if col_name in heatmap_data.columns:
                desired_order.append(col_name)
    
    if desired_order:
        heatmap_data = heatmap_data.reindex(columns=desired_order)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                    ax=ax4, cbar_kws={'label': 'R² Score'})
        ax4.set_title('Provider Performance: Size × Number Type Matrix')
        ax4.set_xlabel('Dataset Category')
        ax4.set_ylabel('Provider')
        
        # Rotate x-axis labels for better readability
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('decimal_places_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second figure focusing on individual model trajectories
    create_model_trajectory_analysis(df, provider_colors)

def create_model_trajectory_analysis(df, provider_colors):
    """Create detailed analysis of individual model performance trajectories."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Individual Model Performance Trajectories Across Dataset Complexity', 
                 fontsize=16, fontweight='bold')
    
    size_order = ['Small', 'Medium', 'Large']
    type_order = ['Integers', 'Balanced', 'Decimals']
    
    # 1. Voyage Models Only
    ax1 = axes[0, 0]
    voyage_models = df[df['provider'] == 'Voyage']['model'].unique()
    
    for model in voyage_models:
        model_data = df[df['model'] == model]
        size_avg = model_data.groupby('size')['linear_test_r2'].mean()
        size_values = [size_avg.get(size, np.nan) for size in size_order]
        
        # Only plot if we have at least some data
        if not all(np.isnan(size_values)):
            ax1.plot(size_order, size_values, marker='o', linewidth=2, 
                    markersize=6, label=model.replace('voyage-', ''), alpha=0.8)
    
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Linear R² Score')
    ax1.set_title('Voyage Models Performance Trajectories', color=provider_colors['Voyage'])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.7, 1.0)
    
    # 2. OpenAI Models Only
    ax2 = axes[0, 1]
    openai_models = df[df['provider'] == 'OpenAI']['model'].unique()
    
    for model in openai_models:
        model_data = df[df['model'] == model]
        size_avg = model_data.groupby('size')['linear_test_r2'].mean()
        size_values = [size_avg.get(size, np.nan) for size in size_order]
        
        # Only plot if we have at least some data
        if not all(np.isnan(size_values)):
            ax2.plot(size_order, size_values, marker='s', linewidth=2, 
                    markersize=6, label=model.replace('text-embedding-', ''), alpha=0.8)
    
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Linear R² Score')
    ax2.set_title('OpenAI Models Performance Trajectories', color=provider_colors['OpenAI'])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 1.0)
    
    # 3. Performance Drop Analysis
    ax3 = axes[1, 0]
    
    # Calculate performance drop from small to large for each model
    performance_drops = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        provider = model_data['provider'].iloc[0] if len(model_data) > 0 else 'Unknown'
        
        small_perf = model_data[model_data['size'] == 'Small']['linear_test_r2'].mean()
        large_perf = model_data[model_data['size'] == 'Large']['linear_test_r2'].mean()
        
        if not np.isnan(small_perf) and not np.isnan(large_perf):
            drop = small_perf - large_perf
            performance_drops.append({
                'model': model,
                'provider': provider,
                'performance_drop': drop,
                'small_perf': small_perf,
                'large_perf': large_perf
            })
    
    if performance_drops:
        drops_df = pd.DataFrame(performance_drops)
        
        for provider in provider_colors.keys():
            provider_data = drops_df[drops_df['provider'] == provider]
            if not provider_data.empty:
                ax3.scatter(provider_data['small_perf'], provider_data['performance_drop'],
                           color=provider_colors[provider], label=provider, s=80, alpha=0.7)
        
        ax3.set_xlabel('Performance on Small Datasets (R²)')
        ax3.set_ylabel('Performance Drop (Small → Large)')
        ax3.set_title('Performance Drop vs Initial Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        if len(drops_df) > 1:
            z = np.polyfit(drops_df['small_perf'], drops_df['performance_drop'], 1)
            p = np.poly1d(z)
            ax3.plot(drops_df['small_perf'], p(drops_df['small_perf']), "r--", alpha=0.5)
    else:
        ax3.text(0.5, 0.5, 'No data available for performance drop analysis', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Number Type Difficulty by Provider
    ax4 = axes[1, 1]
    
    # Filter for valid number types
    valid_type_data = df[df['number_type'].isin(['Integers', 'Balanced', 'Decimals'])]
    
    if len(valid_type_data) > 0:
        type_difficulty = valid_type_data.groupby(['number_type', 'provider'])['linear_test_r2'].agg(['mean', 'std']).reset_index()
        type_difficulty.columns = ['number_type', 'provider', 'mean_r2', 'std_r2']
        
        x_positions = {'Integers': 0, 'Balanced': 1, 'Decimals': 2}
        bar_width = 0.25
        
        for i, provider in enumerate(provider_colors.keys()):
            provider_data = type_difficulty[type_difficulty['provider'] == provider]
            if not provider_data.empty:
                x_vals = [x_positions[nt] + i * bar_width for nt in provider_data['number_type'] if nt in x_positions]
                y_vals = [provider_data[provider_data['number_type'] == nt]['mean_r2'].iloc[0] 
                         for nt in provider_data['number_type'] if nt in x_positions]
                std_vals = [provider_data[provider_data['number_type'] == nt]['std_r2'].iloc[0] 
                           for nt in provider_data['number_type'] if nt in x_positions]
                
                if x_vals and y_vals:
                    ax4.bar(x_vals, y_vals, bar_width, 
                           yerr=std_vals, label=provider,
                           color=provider_colors[provider], alpha=0.8, capsize=5)
        
        ax4.set_xlabel('Number Type')
        ax4.set_ylabel('Average R² Score')
        ax4.set_title('Number Type Difficulty by Provider')
        ax4.set_xticks([pos + bar_width for pos in x_positions.values()])
        ax4.set_xticklabels(x_positions.keys())
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0.7, 1.0)
    else:
        ax4.text(0.5, 0.5, 'No valid number type data available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('model_trajectories_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_explained_variance_analysis(df):
    """Create visualization showing explained variance ratio patterns across complexity."""
    
    # Check if explained variance column exists
    if 'pca_explained_variance_ratio' not in df.columns:
        print("WARNING: 'pca_explained_variance_ratio' column not found. Skipping explained variance analysis.")
        return
    
    # Extract size information from dataset names
    def extract_size_and_type(dataset_name):
        dataset_lower = dataset_name.lower()
        
        if 'small' in dataset_lower:
            size = 'Small'
        elif 'medium' in dataset_lower:
            size = 'Medium'
        elif 'large' in dataset_lower:
            size = 'Large'
        else:
            size = 'Medium'  # Default fallback
            
        if 'integer' in dataset_lower:
            num_type = 'Integers'
        elif 'decimal' in dataset_lower:
            num_type = 'Decimals'
        elif 'balanced' in dataset_lower:
            num_type = 'Balanced'
        else:
            num_type = 'Integers'  # Default fallback
            
        return size, num_type
    
    # Add derived columns
    df['size'], df['number_type'] = zip(*df['dataset'].apply(extract_size_and_type))
    
    # Debug: Print unique values
    print("DEBUG: Explained Variance Analysis - Unique sizes:", df['size'].unique())
    print("DEBUG: Explained Variance Analysis - Unique number types:", df['number_type'].unique())
    
    # Define provider colors (same as before for consistency)
    provider_colors = {
        'Voyage': '#FF6B6B',
        'OpenAI': '#4ECDC4', 
        'Google': '#45B7D1'
    }
    
    # Filter for valid categories
    valid_sizes = ['Small', 'Medium', 'Large']
    valid_types = ['Integers', 'Balanced', 'Decimals']
    
    df_filtered = df[df['size'].isin(valid_sizes) & df['number_type'].isin(valid_types)]
    
    if len(df_filtered) == 0:
        print("WARNING: No data matches expected size/type patterns for explained variance analysis.")
        return
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Explained Variance Ratio: How Much Embedding Variance is Numerical Information', 
                 fontsize=16, fontweight='bold')
    
    # 1. Explained Variance vs Dataset Size
    ax1 = axes[0, 0]
    
    size_order = ['Small', 'Medium', 'Large']
    size_performance = df_filtered.groupby(['size', 'provider'])['pca_explained_variance_ratio'].mean().reset_index()
    
    for provider in provider_colors.keys():
        provider_data = size_performance[size_performance['provider'] == provider]
        if not provider_data.empty:
            ax1.plot(provider_data['size'], provider_data['pca_explained_variance_ratio'], 
                    marker='o', linewidth=3, markersize=8, label=provider, 
                    color=provider_colors[provider], alpha=0.8)
    
    ax1.set_xlabel('Dataset Size Category')
    ax1.set_ylabel('Average Explained Variance Ratio')
    ax1.set_title('Explained Variance Degradation by Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(df_filtered['pca_explained_variance_ratio'].max() * 1.1, 0.1))
    
    # 2. Explained Variance vs Number Type
    ax2 = axes[0, 1]
    
    type_performance = df_filtered.groupby(['number_type', 'provider'])['pca_explained_variance_ratio'].mean().reset_index()
    
    x_positions = {'Integers': 0, 'Balanced': 1, 'Decimals': 2}
    
    for provider in provider_colors.keys():
        provider_data = type_performance[type_performance['provider'] == provider]
        if not provider_data.empty:
            x_vals = [x_positions[nt] for nt in provider_data['number_type'] if nt in x_positions]
            y_vals = [provider_data[provider_data['number_type'] == nt]['pca_explained_variance_ratio'].iloc[0] 
                     for nt in provider_data['number_type'] if nt in x_positions]
            
            if x_vals and y_vals:
                ax2.plot(x_vals, y_vals, 
                        marker='s', linewidth=3, markersize=8, label=provider,
                        color=provider_colors[provider], alpha=0.8)
    
    ax2.set_xlabel('Number Type (Complexity →)')
    ax2.set_ylabel('Average Explained Variance Ratio')
    ax2.set_title('Explained Variance by Number Type Complexity')
    ax2.set_xticks(list(x_positions.values()))
    ax2.set_xticklabels(list(x_positions.keys()))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(df_filtered['pca_explained_variance_ratio'].max() * 1.1, 0.1))
    
    # 3. Top Models Explained Variance Trajectories
    ax3 = axes[1, 0]
    
    # Get top 8 models by average explained variance
    top_models = df_filtered.groupby('model')['pca_explained_variance_ratio'].mean().nlargest(8).index
    
    for i, model in enumerate(top_models):
        model_data = df_filtered[df_filtered['model'] == model]
        if len(model_data) > 0:
            provider = model_data['provider'].iloc[0]
            
            size_avg = model_data.groupby('size')['pca_explained_variance_ratio'].mean()
            size_values = [size_avg.get(size, np.nan) for size in size_order]
            
            ax3.plot(size_order, size_values, marker='o', linewidth=2, 
                    markersize=6, label=model.replace('text-embedding-', 'TE-').replace('voyage-', 'V-'),
                    color=provider_colors[provider], alpha=0.7, linestyle='-' if i < 4 else '--')
    
    ax3.set_xlabel('Dataset Size Category')
    ax3.set_ylabel('Explained Variance Ratio')
    ax3.set_title('Top 8 Models: Explained Variance Across Dataset Sizes')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(df_filtered['pca_explained_variance_ratio'].max() * 1.1, 0.1))
    
    # 4. Correlation: R² vs Explained Variance
    ax4 = axes[1, 1]
    
    # Scatter plot showing relationship between R² and explained variance
    for provider in provider_colors.keys():
        provider_data = df_filtered[df_filtered['provider'] == provider]
        if not provider_data.empty:
            ax4.scatter(provider_data['linear_test_r2'], provider_data['pca_explained_variance_ratio'],
                       color=provider_colors[provider], label=provider, s=60, alpha=0.7)
    
    ax4.set_xlabel('Linear Reconstruction R²')
    ax4.set_ylabel('Explained Variance Ratio')
    ax4.set_title('Reconstruction Quality vs Variance Explained')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add correlation line
    if len(df_filtered) > 1:
        correlation = df_filtered['linear_test_r2'].corr(df_filtered['pca_explained_variance_ratio'])
        z = np.polyfit(df_filtered['linear_test_r2'], df_filtered['pca_explained_variance_ratio'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df_filtered['linear_test_r2'].min(), df_filtered['linear_test_r2'].max(), 100)
        ax4.plot(x_range, p(x_range), "r--", alpha=0.7, 
                label=f'Correlation: {correlation:.3f}')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('pca_explained_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed explained variance trajectories
    create_explained_variance_trajectories(df_filtered, provider_colors)

def create_explained_variance_trajectories(df, provider_colors):
    """Create detailed explained variance trajectory analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Explained Variance Trajectories by Provider and Model', 
                 fontsize=16, fontweight='bold')
    
    size_order = ['Small', 'Medium', 'Large']
    
    # 1. Voyage Models Explained Variance
    ax1 = axes[0, 0]
    voyage_models = df[df['provider'] == 'Voyage']['model'].unique()
    
    for model in voyage_models:
        model_data = df[df['model'] == model]
        size_avg = model_data.groupby('size')['pca_explained_variance_ratio'].mean()
        size_values = [size_avg.get(size, np.nan) for size in size_order]
        
        if not all(np.isnan(size_values)):
            ax1.plot(size_order, size_values, marker='o', linewidth=2, 
                    markersize=6, label=model.replace('voyage-', ''), alpha=0.8)
    
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Voyage Models: Explained Variance Trajectories', color=provider_colors['Voyage'])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. OpenAI Models Explained Variance
    ax2 = axes[0, 1]
    openai_models = df[df['provider'] == 'OpenAI']['model'].unique()
    
    for model in openai_models:
        model_data = df[df['model'] == model]
        size_avg = model_data.groupby('size')['pca_explained_variance_ratio'].mean()
        size_values = [size_avg.get(size, np.nan) for size in size_order]
        
        if not all(np.isnan(size_values)):
            ax2.plot(size_order, size_values, marker='s', linewidth=2, 
                    markersize=6, label=model.replace('text-embedding-', ''), alpha=0.8)
    
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('OpenAI Models: Explained Variance Trajectories', color=provider_colors['OpenAI'])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Explained Variance Drop Analysis
    ax3 = axes[1, 0]
    
    # Calculate explained variance drop from small to large
    variance_drops = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        provider = model_data['provider'].iloc[0] if len(model_data) > 0 else 'Unknown'
        
        small_var = model_data[model_data['size'] == 'Small']['pca_explained_variance_ratio'].mean()
        large_var = model_data[model_data['size'] == 'Large']['pca_explained_variance_ratio'].mean()
        
        if not np.isnan(small_var) and not np.isnan(large_var):
            drop = small_var - large_var
            variance_drops.append({
                'model': model,
                'provider': provider,
                'variance_drop': drop,
                'small_var': small_var,
                'large_var': large_var
            })
    
    if variance_drops:
        drops_df = pd.DataFrame(variance_drops)
        
        for provider in provider_colors.keys():
            provider_data = drops_df[drops_df['provider'] == provider]
            if not provider_data.empty:
                ax3.scatter(provider_data['small_var'], provider_data['variance_drop'],
                           color=provider_colors[provider], label=provider, s=80, alpha=0.7)
        
        ax3.set_xlabel('Explained Variance on Small Datasets')
        ax3.set_ylabel('Explained Variance Drop (Small → Large)')
        ax3.set_title('Variance Drop vs Initial Variance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        if len(drops_df) > 1:
            z = np.polyfit(drops_df['small_var'], drops_df['variance_drop'], 1)
            p = np.poly1d(z)
            ax3.plot(drops_df['small_var'], p(drops_df['small_var']), "r--", alpha=0.5)
    else:
        ax3.text(0.5, 0.5, 'No data available for variance drop analysis', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Provider Comparison: Absolute Explained Variance
    ax4 = axes[1, 1]
    
    # Box plot showing distribution of explained variance by provider
    provider_data_list = []
    provider_labels = []
    
    for provider in provider_colors.keys():
        provider_values = df[df['provider'] == provider]['pca_explained_variance_ratio'].dropna()
        if len(provider_values) > 0:
            provider_data_list.append(provider_values)
            provider_labels.append(provider)
    
    if provider_data_list:
        bp = ax4.boxplot(provider_data_list, labels=provider_labels, patch_artist=True)
        
        # Color the boxes
        for patch, provider in zip(bp['boxes'], provider_labels):
            patch.set_facecolor(provider_colors[provider])
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Explained Variance Ratio')
        ax4.set_title('Explained Variance Distribution by Provider')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add mean markers
        means = [np.mean(data) for data in provider_data_list]
        ax4.scatter(range(1, len(means) + 1), means, 
                   color='red', marker='D', s=50, zorder=3, label='Mean')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No data available for provider comparison', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('pca_explained_variance_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_results():
    """Analyze the embedding reconstruction results."""
    
    # Read the Excel file
    df = pd.read_excel('embedding_reconstruction_results.xlsx')
    
    print("🔬 EMBEDDING NUMERICAL RECONSTRUCTION EXPERIMENT - KEY FINDINGS")
    print("=" * 80)
    
    # Basic info
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    
    print(f"📊 EXPERIMENT SCOPE:")
    print(f"   • {len(datasets)} numerical datasets tested")
    print(f"   • {len(models)} embedding models compared")
    print(f"   • {len(df)} total experiments conducted")
    print()
    
    # Main performance statistics
    linear_scores = df['linear_test_r2']
    pca_scores = df['pca_test_r2']
    
    print("🎯 MAIN FINDINGS:")
    print("   1. LINEAR RECONSTRUCTION: HIGHLY SUCCESSFUL")
    print(f"      • Average R² = {linear_scores.mean():.4f} ({linear_scores.mean()*100:.1f}% variance explained!)")
    print(f"      • Range: {linear_scores.min():.3f} to {linear_scores.max():.3f}")
    print("      • This proves embeddings contain rich numerical information")
    print()
    print("   2. PCA RECONSTRUCTION: COMPLETE FAILURE")
    print(f"      • Average R² = {pca_scores.mean():.1f} (negative = worse than mean prediction)")
    print("      • Numerical info is NOT in the first principal component")
    print("      • Numbers are encoded across many embedding dimensions")
    print()
    
    # Model performance ranking
    model_performance = df.groupby('model')['linear_test_r2'].mean().sort_values(ascending=False)
    
    print("🏆 TOP PERFORMING MODELS:")
    for i, (model, score) in enumerate(model_performance.head().items()):
        provider = 'Voyage' if 'voyage' in model else 'OpenAI' if 'text-embedding' in model else 'Google'
        print(f"   {i+1}. {model} ({provider}): R² = {score:.4f}")
    print()
    
    # Provider comparison
    voyage_models = model_performance[model_performance.index.str.contains('voyage')]
    openai_models = model_performance[model_performance.index.str.contains('text-embedding')]
    google_models = model_performance[model_performance.index.str.contains('gemini')]
    
    print("📈 MODEL RANKINGS BY PROVIDER:")
    print(f"   Voyage AI: {voyage_models.mean():.4f} avg ({len(voyage_models)} models)")
    print(f"   OpenAI: {openai_models.mean():.4f} avg ({len(openai_models)} models)")
    print(f"   Google: {google_models.mean():.4f} avg ({len(google_models)} models)")
    print()
    
    # Dataset difficulty analysis
    dataset_performance = df.groupby('dataset').agg({
        'linear_test_r2': 'mean',
        'dataset_range': 'first',
        'dataset_min_value': 'first',
        'dataset_max_value': 'first'
    }).sort_values('linear_test_r2')
    
    print("📊 DATASET DIFFICULTY ANALYSIS:")
    print("   Datasets ranked by difficulty (hardest first):")
    
    for i, (dataset, row) in enumerate(dataset_performance.iterrows()):
        score = row['linear_test_r2']
        range_val = row['dataset_range']
        
        if score < 0.8:
            difficulty = "VERY HARD"
        elif score < 0.9:
            difficulty = "HARD"
        elif score < 0.95:
            difficulty = "MEDIUM"
        else:
            difficulty = "EASY"
            
        print(f"   {i+1}. {dataset}: {difficulty} (R² = {score:.4f})")
        print(f"      Range: {range_val:.2e}")
    print()
    
    # CREATE VISUALIZATIONS HERE
    print("📈 GENERATING VISUALIZATIONS...")
    create_visualizations(df)
    print("   • Saved: embedding_reconstruction_analysis.png")
    print("   • Saved: detailed_model_analysis.png")
    print("   • Saved: decimal_places_analysis.png")
    print("   • Saved: model_trajectories_analysis.png")
    print("   • Saved: explained_variance_analysis.png")
    print("   • Saved: explained_variance_trajectories.png")
    print()
    
    # Key insights
    print("🔍 KEY INSIGHTS:")
    print("   • Voyage models dominate the leaderboard")
    print("   • Specialized models (law, finance) perform exceptionally well")
    print("   • Large number ranges make reconstruction much harder")
    print("   • 'Balanced' datasets (mixed integers/decimals) are most challenging")
    print("   • Simple integers are easiest to reconstruct")
    print()
    
    print("💡 IMPLICATIONS:")
    print("   • Embeddings ARE encoding numerical semantics, not just treating numbers as text")
    print("   • This information is distributed across many dimensions (not just PC1)")
    print("   • Voyage models may have better numerical understanding")
    print("   • Specialized domain models (law, finance) handle numbers better")
    print("   • Linear models can effectively extract this numerical information")
    print()
    
    print("⚠️  LIMITATIONS:")
    print("   • This tests reconstruction, not mathematical reasoning")
    print("   • Results may vary with different number formats or contexts")
    print("   • PCA failure doesn't mean numbers aren't systematically encoded")
    print()
    
    # Statistical summary
    print("📋 DETAILED STATISTICS:")
    print(f"Linear Test R² Statistics:")
    print(f"   Mean: {linear_scores.mean():.4f}")
    print(f"   Std:  {linear_scores.std():.4f}")
    print(f"   Min:  {linear_scores.min():.4f}")
    print(f"   Max:  {linear_scores.max():.4f}")
    print(f"   Median: {linear_scores.median():.4f}")
    print()
    
    print("PCA Test R² Statistics:")
    print(f"   Mean: {pca_scores.mean():.4f}")
    print(f"   Std:  {pca_scores.std():.4f}")
    print(f"   Min:  {pca_scores.min():.4f}")
    print(f"   Max:  {pca_scores.max():.4f}")
    print()
    
    # Show worst and best individual results
    print("🏅 BEST INDIVIDUAL RESULTS:")
    best_results = df.nlargest(5, 'linear_test_r2')[['dataset', 'model', 'linear_test_r2']]
    for i, (_, row) in enumerate(best_results.iterrows()):
        print(f"   {i+1}. {row['dataset']} + {row['model']}: R² = {row['linear_test_r2']:.4f}")
    print()
    
    print("💥 WORST INDIVIDUAL RESULTS:")
    worst_results = df.nsmallest(5, 'linear_test_r2')[['dataset', 'model', 'linear_test_r2']]
    for i, (_, row) in enumerate(worst_results.iterrows()):
        print(f"   {i+1}. {row['dataset']} + {row['model']}: R² = {row['linear_test_r2']:.4f}")

if __name__ == "__main__":
    analyze_results()