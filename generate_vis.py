import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load the statsmodels k-fold data
df = pd.read_csv('results/sweep/statsmodels_sweep_results.csv')

# Set up the plot style
plt.style.use('default')

# Create directory structure
os.makedirs('final_plots/statsmodels/general_models', exist_ok=True)
os.makedirs('final_plots/statsmodels/specialist_models', exist_ok=True)
os.makedirs('final_plots/statsmodels/by_provider/openai', exist_ok=True)
os.makedirs('final_plots/statsmodels/by_provider/google', exist_ok=True)
os.makedirs('final_plots/statsmodels/by_provider/voyage', exist_ok=True)
os.makedirs('final_plots/statsmodels/significance', exist_ok=True)

# Create a mapping for provider and model type
def get_provider_and_type(model_name):
    model_lower = model_name.lower()
    
    # Exclude multimodal models
    if 'multimodal' in model_lower:
        return None, None

    if model_lower in ['text-embedding-3-large', 'gemini-embedding-001', 'voyage-3-large']:
        is_specialist = False
    else:
        is_specialist = True
    
    if any(openai_model in model_lower for openai_model in ['text-embedding', 'ada']):
        provider = 'OpenAI'
        # is_specialist = False
    elif 'gemini' in model_lower:
        provider = 'Google'
        # is_specialist = False
    elif 'voyage' in model_lower:
        provider = 'Voyage'
        # if any(specialist in model_lower for specialist in ['finance', 'law', 'code']):
        #     is_specialist = True
        # else:
        #     is_specialist = False
    else:
        provider = 'Other'
        # is_specialist = False
    
    return provider, is_specialist

# Add provider and specialist columns
df_temp = []
for index, row in df.iterrows():
    provider, is_specialist = get_provider_and_type(row['model'])
    if provider is not None:  # Skip multimodal models
        new_row = row.copy()
        new_row['provider'] = provider
        new_row['is_specialist'] = is_specialist
        df_temp.append(new_row)

# Create new dataframe without multimodal models
df = pd.DataFrame(df_temp).reset_index(drop=True)

# Define colors for providers
provider_colors = {
    'OpenAI': '#1f77b4',
    'Google': '#ff7f0e', 
    'Voyage': '#2ca02c',
    'Other': '#d62728'
}

# Define line styles for different models within providers
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5, 1, 5))]
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# Get unique experiments
experiments = df['experiment'].unique()

print(f"Found {len(experiments)} experiments: {experiments}")
print(f"Providers found: {df['provider'].unique()}")
print(f"K-folds: {df['k_folds'].iloc[0] if len(df) > 0 else 'N/A'}")
print(f"Model type: {df['model_type'].iloc[0] if len(df) > 0 else 'N/A'}")
print(f"Significance level: {df['alpha'].iloc[0] if len(df) > 0 else 'N/A'}")

# Separate general and specialist models
general_models = df[df['is_specialist'] == False].copy()
specialist_models = df[df['is_specialist'] == True].copy()

print(f"\nGeneral models: {general_models['model'].unique()}")
print(f"Specialist models: {specialist_models['model'].unique()}")

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method for axis limits"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def create_plots(data_subset, plot_type, save_dir, title_suffix=""):
    """Create the original four plot types plus new significance plots for a given data subset"""
    
    for experiment in data_subset['experiment'].unique():
        exp_data = data_subset[data_subset['experiment'] == experiment].copy()
        exp_data = exp_data.sort_values('size')
        
        if len(exp_data) == 0:
            continue
        
        # Calculate outlier bounds for R2 values using means
        linear_r2_lower, linear_r2_upper = detect_outliers_iqr(exp_data, 'linear_r2_mean')
        pca_r2_lower, pca_r2_upper = detect_outliers_iqr(exp_data, 'pca_r2_mean')
        
        # Figure 1: Linear R² vs Number Precision with error bars
        plt.figure(figsize=(12, 6))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.errorbar(model_data['size'], model_data['linear_r2_mean'],
                        yerr=model_data['linear_r2_std'], 
                        color=provider_colors[provider],
                        linestyle=line_styles[model_idx % len(line_styles)],
                        marker=markers[model_idx % len(markers)],
                        linewidth=2, markersize=6, capsize=5,
                        label=f'{model}')
            model_idx += 1
        
        plt.xlabel('Number Precision (size)')
        plt.ylabel('Linear R² (Mean ± Std)')
        plt.title(f'Linear R² vs Number Precision - {experiment.replace("_", " ").title()}')
        plt.ylim(max(0, linear_r2_lower - 0.05), linear_r2_upper + 0.05)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(exp_data['model'].unique()), 4))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f'{save_dir}/linear_r2_vs_precision_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: Explained Variance Ratio vs Number Precision with error bars
        plt.figure(figsize=(12, 6))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.errorbar(model_data['size'], model_data['pca_explained_var_mean'],
                        yerr=model_data['pca_explained_var_std'], 
                        color=provider_colors[provider],
                        linestyle=line_styles[model_idx % len(line_styles)],
                        marker=markers[model_idx % len(markers)],
                        linewidth=2, markersize=6, capsize=5,
                        label=f'{model}')
            model_idx += 1
        
        plt.xlabel('Number Precision (size)')
        plt.ylabel('PCA Component 1 Explained Variance (Mean ± Std)')
        plt.title(f'PCA Explained Variance vs Number Precision - {experiment.replace("_", " ").title()}')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(exp_data['model'].unique()), 4))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f'{save_dir}/explained_variance_vs_precision_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 3: PCA R² vs Number Precision with error bars
        plt.figure(figsize=(12, 6))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.errorbar(model_data['size'], model_data['pca_r2_mean'],
                        yerr=model_data['pca_r2_std'], 
                        color=provider_colors[provider],
                        linestyle=line_styles[model_idx % len(line_styles)],
                        marker=markers[model_idx % len(markers)],
                        linewidth=2, markersize=6, capsize=5,
                        label=f'{model}')
            model_idx += 1
        
        plt.xlabel('Number Precision (size)')
        plt.ylabel('PCA R² (Mean ± Std)')
        plt.title(f'PCA R² vs Number Precision - {experiment.replace("_", " ").title()}')
        plt.ylim(max(0, pca_r2_lower - 0.05), pca_r2_upper + 0.05)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(exp_data['model'].unique()), 4))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f'{save_dir}/pca_r2_vs_precision_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 4: PCA R² vs Linear R² scatter plot with error bars
        plt.figure(figsize=(10, 8))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.errorbar(model_data['linear_r2_mean'], model_data['pca_r2_mean'],
                        xerr=model_data['linear_r2_std'], yerr=model_data['pca_r2_std'],
                        fmt=markers[model_idx % len(markers)], 
                        color=provider_colors[provider],
                        markersize=8, capsize=3, alpha=0.7,
                        label=f'{model}')
            model_idx += 1
        
        plt.xlabel('Linear Model R² (Mean ± Std)')
        plt.ylabel('PCA R² (Mean ± Std)')
        plt.title(f'PCA vs Linear R² - {experiment.replace("_", " ").title()}')
        plt.xlim(max(0, linear_r2_lower - 0.05), linear_r2_upper + 0.05)
        plt.ylim(max(0, pca_r2_lower - 0.05), pca_r2_upper + 0.05)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(exp_data['model'].unique()), 4))
        plt.grid(True, alpha=0.3)
        
        # Add diagonal reference line
        min_val = min(plt.xlim()[0], plt.ylim()[0])
        max_val = max(plt.xlim()[1], plt.ylim()[1])
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='y=x')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f'{save_dir}/pca_vs_linear_r2_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 5: NEW - Proportion of Significant Linear Features vs Number Precision
        plt.figure(figsize=(12, 6))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.errorbar(model_data['size'], model_data['linear_proportion_significant_mean'],
                        yerr=model_data['linear_proportion_significant_std'], 
                        color=provider_colors[provider],
                        linestyle=line_styles[model_idx % len(line_styles)],
                        marker=markers[model_idx % len(markers)],
                        linewidth=2, markersize=6, capsize=5,
                        label=f'{model}')
            model_idx += 1
        
        plt.xlabel('Number Precision (size)')
        plt.ylabel('Proportion Significant Linear Features (Mean ± Std)')
        plt.title(f'Statistical Significance: Linear Features - {experiment.replace("_", " ").title()}')
        plt.ylim(0, 1.05)
        alpha_val = exp_data['alpha'].iloc[0] if len(exp_data) > 0 else 0.05
        plt.axhline(y=alpha_val, color='red', linestyle=':', alpha=0.7, label=f'α={alpha_val}')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(exp_data['model'].unique()), 4))
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f'{save_dir}/linear_significance_proportion_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 6: NEW - F-test P-values vs Number Precision (Log scale)
        plt.figure(figsize=(12, 6))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.errorbar(model_data['size'], model_data['linear_f_pvalue_mean'],
                        yerr=model_data['linear_f_pvalue_std'], 
                        color=provider_colors[provider],
                        linestyle=line_styles[model_idx % len(line_styles)],
                        marker=markers[model_idx % len(markers)],
                        linewidth=2, markersize=6, capsize=5,
                        label=f'{model} Linear')
            
            plt.errorbar(model_data['size'], model_data['pca_f_pvalue_mean'],
                        yerr=model_data['pca_f_pvalue_std'], 
                        color=provider_colors[provider],
                        linestyle=line_styles[model_idx % len(line_styles)],
                        marker=markers[model_idx % len(markers)],
                        linewidth=2, markersize=4, capsize=3, alpha=0.7,
                        label=f'{model} PCA')
            model_idx += 1
        
        plt.xlabel('Number Precision (size)')
        plt.ylabel('F-test P-value (Mean ± Std)')
        plt.title(f'Model Statistical Significance (F-test) - {experiment.replace("_", " ").title()}')
        plt.yscale('log')
        alpha_val = exp_data['alpha'].iloc[0] if len(exp_data) > 0 else 0.05
        plt.axhline(y=alpha_val, color='red', linestyle=':', alpha=0.7, label=f'α={alpha_val}')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(exp_data['model'].unique()), 3))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f'{save_dir}/f_test_significance_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 7: NEW - Model Selection Criteria (AIC/BIC)
        plt.figure(figsize=(12, 6))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.plot(model_data['size'], model_data['linear_aic_mean'],
                    color=provider_colors[provider],
                    linestyle=line_styles[model_idx % len(line_styles)],
                    marker=markers[model_idx % len(markers)],
                    linewidth=2, markersize=6,
                    label=f'{model} Linear AIC')
            
            plt.plot(model_data['size'], model_data['pca_aic_mean'],
                    color=provider_colors[provider],
                    linestyle=line_styles[model_idx % len(line_styles)],
                    marker=markers[model_idx % len(markers)],
                    linewidth=2, markersize=4, alpha=0.7,
                    label=f'{model} PCA AIC')
            model_idx += 1
        
        plt.xlabel('Number Precision (size)')
        plt.ylabel('AIC (Mean)')
        plt.title(f'Model Selection Criterion (AIC) - {experiment.replace("_", " ").title()}')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(exp_data['model'].unique()), 3))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f'{save_dir}/model_selection_aic_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_multi_axis_plots(data_subset, plot_type, save_dir, title_suffix=""):
    """Create multi-axis plots showing all three datasets together including significance plots"""
    
    if len(data_subset) == 0:
        return
    
    # Get global bounds for consistent y-axis scaling
    linear_r2_lower, linear_r2_upper = detect_outliers_iqr(data_subset, 'linear_r2_mean')
    pca_r2_lower, pca_r2_upper = detect_outliers_iqr(data_subset, 'pca_r2_mean')
    pca_var_lower, pca_var_upper = detect_outliers_iqr(data_subset, 'pca_explained_var_mean')
    
    experiment_names = sorted(data_subset['experiment'].unique())
    
    # Plot 1: Linear R² vs Number Precision (all experiments)
    fig, axes = plt.subplots(1, len(experiment_names), figsize=(3.5 * len(experiment_names), 4))
    if len(experiment_names) == 1:
        axes = [axes]
    
    experiment_written_names = ['Positive Decimals', 'Mixed Sign Decimals', 'Mixed Sign Integers']
    xlabels = ['Decimal Precision $b$', 'Decimal Precision $b$', 'Integer Places $a$']

    for idx, experiment in enumerate(experiment_names[::-1]):
        ax = axes[idx]
        exp_data = data_subset[data_subset['experiment'] == experiment].copy()
        exp_data = exp_data.sort_values('size')
        
        if len(exp_data) == 0:
            continue
        
        model_idx = 0
        for model in sorted(exp_data['model'].unique()):
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            ax.errorbar(model_data['size'], model_data['linear_r2_mean'],
                       yerr=model_data['linear_r2_std'],
                       color=provider_colors[provider],
                       linestyle=line_styles[model_idx % len(line_styles)],
      
                       linewidth=2, capsize=5,
                       label=f'{model}')
            model_idx += 1
        
        ax.set_xlabel(xlabels[idx])
        ax.set_title(experiment_written_names[idx],  fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Linear $R^2$')
        
        ax.set_ylim(max(0, linear_r2_lower - 0.05), linear_r2_upper + 0.05)
        ax.grid(True, alpha=0.3)
    
    # Create a single legend at the bottom
    if len(experiment_names) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(labels), 8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f'{save_dir}/multi_linear_r2_vs_precision.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Significance Analysis - Multi-axis
    fig, axes = plt.subplots(1, len(experiment_names), figsize=(3.5 * len(experiment_names), 4))
    if len(experiment_names) == 1:
        axes = [axes]
    
    for idx, experiment in enumerate(experiment_names[::-1]):
        ax = axes[idx]
        exp_data = data_subset[data_subset['experiment'] == experiment].copy()
        exp_data = exp_data.sort_values('size')
        
        if len(exp_data) == 0:
            continue
        
        model_idx = 0
        for model in sorted(exp_data['model'].unique()):
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            ax.errorbar(model_data['size'], model_data['linear_proportion_significant_mean'],
                       yerr=model_data['linear_proportion_significant_std'],
                       color=provider_colors[provider],
                       linestyle=line_styles[model_idx % len(line_styles)],
      
                       linewidth=2, capsize=5,
                       label=f'{model}')
            model_idx += 1
        
        ax.set_xlabel('Number Precision (size)')
        if idx == 0:
            ax.set_ylabel('Proportion Significant Features (Mean ± Std)')
        ax.set_title(experiment.replace('_', ' ').title(),  fontweight='bold')
        ax.set_ylim(0, 1.05)
        alpha_val = exp_data['alpha'].iloc[0] if len(exp_data) > 0 else 0.05
        ax.axhline(y=alpha_val, color='red', linestyle=':', alpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Create a single legend at the bottom
    if len(experiment_names) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(labels), 8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig(f'{save_dir}/multi_significance_proportion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: PCA R² vs Number Precision (all experiments) - RESTORED
    fig, axes = plt.subplots(1, len(experiment_names), figsize=(3.5 * len(experiment_names), 4))
    if len(experiment_names) == 1:
        axes = [axes]
    
    for idx, experiment in enumerate(experiment_names[::-1]):
        ax = axes[idx]
        exp_data = data_subset[data_subset['experiment'] == experiment].copy()
        exp_data = exp_data.sort_values('size')
        
        if len(exp_data) == 0:
            continue
        
        model_idx = 0
        for model in sorted(exp_data['model'].unique()):
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            ax.errorbar(model_data['size'], model_data['pca_r2_mean'],
                       yerr=model_data['pca_r2_std'],
                       color=provider_colors[provider],
                       linestyle=line_styles[model_idx % len(line_styles)],
      
                       linewidth=2, capsize=5,
                       label=f'{model}')
            model_idx += 1
        
        ax.set_xlabel(xlabels[idx])
        ax.set_title(experiment_written_names[idx],  fontweight='bold')
        if idx == 0:
            ax.set_ylabel('PCA$_0$ $R^2$')
        # ax.set_title(experiment.replace('_', ' ').title(),  fontweight='bold')
        ax.set_ylim(max(0, pca_r2_lower - 0.05), pca_r2_upper + 0.05)
        ax.grid(True, alpha=0.3)
    
    if len(experiment_names) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(labels), 8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f'{save_dir}/multi_pca_r2_vs_precision.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: PCA R² vs Linear R² scatter (all experiments) - RESTORED
    fig, axes = plt.subplots(1, len(experiment_names), figsize=(3.5 * len(experiment_names), 4))
    if len(experiment_names) == 1:
        axes = [axes]
    
    for idx, experiment in enumerate(experiment_names[::-1]):
        ax = axes[idx]
        exp_data = data_subset[data_subset['experiment'] == experiment].copy()
        
        if len(exp_data) == 0:
            continue
        
        model_idx = 0
        for model in sorted(exp_data['model'].unique()):
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            ax.errorbar(model_data['linear_r2_mean'], model_data['pca_r2_mean'],
                       xerr=model_data['linear_r2_std'], yerr=model_data['pca_r2_std'],
                       fmt=markers[model_idx % len(markers)], 
                       color=provider_colors[provider],
                       markersize=8, capsize=3, alpha=0.7,
                       label=f'{model}')
            model_idx += 1
        
        ax.set_xlabel(xlabels[idx])
        ax.set_title(experiment_written_names[idx],  fontweight='bold')
        if idx == 0:
            ax.set_ylabel('PCA$_0$ $R^2$')
        # ax.set_title(experiment.replace('_', ' ').title(),  fontweight='bold')
        ax.set_xlim(max(0, linear_r2_lower - 0.05), linear_r2_upper + 0.05)
        ax.set_ylim(max(0, pca_r2_lower - 0.05), pca_r2_upper + 0.05)
        ax.grid(True, alpha=0.3)
        
        # Add diagonal reference line
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
        
    if len(experiment_names) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(labels), 8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f'{save_dir}/multi_pca_vs_linear_r2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Explained Variance Multi-axis
    fig, axes = plt.subplots(1, len(experiment_names), figsize=(3.5 * len(experiment_names), 4))
    if len(experiment_names) == 1:
        axes = [axes]
    
    for idx, experiment in enumerate(experiment_names):
        ax = axes[idx]
        exp_data = data_subset[data_subset['experiment'] == experiment].copy()
        exp_data = exp_data.sort_values('size')
        
        if len(exp_data) == 0:
            continue
        
        model_idx = 0
        for model in sorted(exp_data['model'].unique()):
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            ax.errorbar(model_data['size'], model_data['pca_explained_var_mean'],
                       yerr=model_data['pca_explained_var_std'],
                       color=provider_colors[provider],
                       linestyle=line_styles[model_idx % len(line_styles)],
      
                       linewidth=2, capsize=5,
                       label=f'{model}')
            model_idx += 1
        ax.set_xlabel(xlabels[idx])
        ax.set_title(experiment_written_names[idx],  fontweight='bold')
        # ax.set_xlabel('Number Precision (size)')
        if idx == 0:
            ax.set_ylabel('PCA$_0$ Variance Ratio')
        # ax.set_title(experiment.replace('_', ' ').title(),  fontweight='bold')
        ax.set_ylim(max(0, pca_var_lower - 0.05), pca_var_upper + 0.05)
        ax.grid(True, alpha=0.3)
    
    if len(experiment_names) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(labels), 8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f'{save_dir}/multi_explained_variance_vs_precision.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_significance_summary_plots(data_subset, save_dir):
    """Create summary plots focused on statistical significance across all experiments"""
    
    if len(data_subset) == 0:
        return
    
    # Plot 1: Significance Heatmap by Model and Size
    pivot_data = data_subset.pivot_table(
        values='linear_proportion_significant_mean',
        index='model',
        columns='size',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, annot=True, cmap='viridis', cbar_kws={'label': 'Proportion Significant Features'})
    plt.title('Statistical Significance Heatmap: Linear Models')
    plt.xlabel('Number Precision (size)')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/significance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: F-test Significance Summary
    plt.figure(figsize=(12, 8))
    
    for experiment in data_subset['experiment'].unique():
        exp_data = data_subset[data_subset['experiment'] == experiment]
        
        # Count models with significant F-tests
        significant_counts = []
        sizes = sorted(exp_data['size'].unique())
        
        for size in sizes:
            size_data = exp_data[exp_data['size'] == size]
            alpha_val = size_data['alpha'].iloc[0] if len(size_data) > 0 else 0.05
            significant = (size_data['linear_f_pvalue_mean'] < alpha_val).sum()
            total = len(size_data)
            significant_counts.append(significant / total if total > 0 else 0)
        
        plt.plot(sizes, significant_counts, marker='o', linewidth=2, markersize=6, 
                label=experiment.replace('_', ' ').title())
    
    plt.xlabel('Number Precision (size)')
    plt.ylabel('Proportion of Models with Significant F-test')
    plt.title('Statistical Significance Summary: F-test Results')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/f_test_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create plots for general models
print("\n" + "="*60)
print("CREATING PLOTS FOR GENERAL MODELS (STATSMODELS)")
print("="*60)
create_plots(general_models, 'general', 'final_plots/statsmodels/general_models')
create_multi_axis_plots(general_models, 'general', 'final_plots/statsmodels/general_models')

# Create plots for specialist models
if len(specialist_models) > 0:
    print("\n" + "="*60)
    print("CREATING PLOTS FOR SPECIALIST MODELS (STATSMODELS)")
    print("="*60)
    create_plots(specialist_models, 'specialist', 'final_plots/statsmodels/specialist_models')
    create_multi_axis_plots(specialist_models, 'specialist', 'final_plots/statsmodels/specialist_models')

# Create plots by provider
print("\n" + "="*60)
print("CREATING PLOTS BY PROVIDER (STATSMODELS)")
print("="*60)

for provider in general_models['provider'].unique():
    print(f"\nProcessing {provider}...")
    provider_data = general_models[general_models['provider'] == provider]
    provider_dir = f"final_plots/statsmodels/by_provider/{provider.lower()}"
    create_plots(provider_data, 'provider', provider_dir)
    create_multi_axis_plots(provider_data, 'provider', provider_dir)

# Create significance summary plots
print("\n" + "="*60)
print("CREATING SIGNIFICANCE SUMMARY PLOTS")
print("="*60)
create_significance_summary_plots(df, 'final_plots/statsmodels/significance')

def create_latex_tables():
    """Generate LaTeX tables for statsmodels results"""
    
    # Create tables directory
    os.makedirs('final_plots/statsmodels/tables', exist_ok=True)
    
    print("\nGenerating LaTeX tables for statsmodels results...")
    
    # Table 1: Model Overview with Statistical Information
    size_ranges = df.groupby('model').agg({
        'size': ['min', 'max'],
        'provider': 'first',
        'is_specialist': 'first',
        'k_folds': 'first',
        'alpha': 'first'
    }).reset_index()
    size_ranges.columns = ['Model', 'Min Size', 'Max Size', 'Provider', 'Is Specialist', 'K Folds', 'Alpha']
    size_ranges['Type'] = size_ranges['Is Specialist'].map({True: 'Specialist', False: 'General'})
    size_ranges['Size Range'] = size_ranges['Min Size'].astype(int).astype(str) + '-' + size_ranges['Max Size'].astype(int).astype(str)
    
    model_overview = size_ranges[['Provider', 'Model', 'Type', 'Size Range', 'K Folds', 'Alpha']].sort_values(['Provider', 'Type', 'Model'])
    
    latex_table = model_overview.to_latex(
        index=False,
        caption='Statsmodels Analysis: Model Overview',
        label='tab:statsmodels_overview',
        position='htbp',
        column_format='llcccc',
        escape=False
    )
    
    with open('final_plots/statsmodels/tables/model_overview.txt', 'w') as f:
        f.write(latex_table)
    print("✓ Generated model_overview.txt")
    
    # Table 2: Performance and Significance Summary by Experiment
    for experiment in experiments:
        exp_data = df[df['experiment'] == experiment]
        
        # Calculate statistics by provider
        stats_list = []
        for provider in sorted(exp_data['provider'].unique()):
            provider_data = exp_data[exp_data['provider'] == provider]
            
            # Calculate significance statistics
            avg_sig_prop = provider_data['linear_proportion_significant_mean'].mean()
            avg_f_pvalue = provider_data['linear_f_pvalue_mean'].mean()
            alpha_val = provider_data['alpha'].iloc[0] if len(provider_data) > 0 else 0.05
            
            stats_list.append({
                'Provider': provider,
                'Models': len(provider_data['model'].unique()),
                'Linear R² (μ±σ)': f"{provider_data['linear_r2_mean'].mean():.3f}±{provider_data['linear_r2_mean'].std():.3f}",
                'PCA R² (μ±σ)': f"{provider_data['pca_r2_mean'].mean():.3f}±{provider_data['pca_r2_mean'].std():.3f}",
                'Sig. Features (%)': f"{avg_sig_prop:.1%}",
                'Avg F-p-value': f"{avg_f_pvalue:.2e}",
                'Significant Models': f"{(provider_data['linear_f_pvalue_mean'] < alpha_val).sum()}/{len(provider_data)}"
            })
        
        stats_df = pd.DataFrame(stats_list)
        
        latex_table = stats_df.to_latex(
            index=False,
            caption=f'Statsmodels Performance & Significance - {experiment.replace("_", " ").title()}',
            label=f'tab:statsmodels_{experiment}',
            position='htbp',
            column_format='lcccccc',
            escape=False
        )
        
        with open(f'final_plots/statsmodels/tables/performance_significance_{experiment}.txt', 'w') as f:
            f.write(latex_table)
        print(f"✓ Generated performance_significance_{experiment}.txt")
    
    # Table 3: Statistical Significance Analysis by Size
    for experiment in experiments:
        exp_data = df[df['experiment'] == experiment]
        
        if len(exp_data) == 0:
            continue
        
        # Analyze significance by size
        sig_analysis = []
        for size in sorted(exp_data['size'].unique()):
            size_data = exp_data[exp_data['size'] == size]
            
            if len(size_data) == 0:
                continue
            
            alpha_val = size_data['alpha'].iloc[0] if len(size_data) > 0 else 0.05
            significant_f_tests = (size_data['linear_f_pvalue_mean'] < alpha_val).sum()
            total_models = len(size_data)
            
            # Best and worst significance
            best_sig = size_data.loc[size_data['linear_proportion_significant_mean'].idxmax()]
            worst_sig = size_data.loc[size_data['linear_proportion_significant_mean'].idxmin()]
            
            sig_analysis.append({
                'Size': int(size),
                'Models Tested': total_models,
                'Significant F-tests': f"{significant_f_tests}/{total_models}",
                'Best Model (Sig %)': f"{best_sig['model'][:20]}... ({best_sig['linear_proportion_significant_mean']:.1%})",
                'Worst Model (Sig %)': f"{worst_sig['model'][:20]}... ({worst_sig['linear_proportion_significant_mean']:.1%})",
                'Avg Sig Features': f"{size_data['linear_proportion_significant_mean'].mean():.1%}±{size_data['linear_proportion_significant_mean'].std():.1%}"
            })
        
        if sig_analysis:
            sig_df = pd.DataFrame(sig_analysis)
            
            latex_table = sig_df.to_latex(
                index=False,
                caption=f'Statistical Significance by Size - {experiment.replace("_", " ").title()}',
                label=f'tab:significance_{experiment}',
                position='htbp',
                column_format='cccccc',
                escape=False
            )
            
            with open(f'final_plots/statsmodels/tables/significance_analysis_{experiment}.txt', 'w') as f:
                f.write(latex_table)
            print(f"✓ Generated significance_analysis_{experiment}.txt")
    
    # Table 4: Model Comparison - Linear vs PCA Performance
    comparison_data = []
    for experiment in experiments:
        exp_data = df[df['experiment'] == experiment]
        
        for model in sorted(exp_data['model'].unique()):
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            # Calculate averages and comparisons
            avg_linear_r2 = model_data['linear_r2_mean'].mean()
            avg_pca_r2 = model_data['pca_r2_mean'].mean()
            avg_linear_sig = model_data['linear_proportion_significant_mean'].mean()
            avg_pca_sig = model_data['pca_proportion_significant_mean'].mean()
            
            # Count where linear > PCA
            linear_better = (model_data['linear_r2_mean'] > model_data['pca_r2_mean']).sum()
            total_comparisons = len(model_data)
            
            comparison_data.append({
                'Experiment': experiment.replace('_', ' ').title(),
                'Provider': provider,
                'Model': model[:25] + '...' if len(model) > 25 else model,
                'Linear R²': f"{avg_linear_r2:.4f}",
                'PCA R²': f"{avg_pca_r2:.4f}",
                'Linear > PCA': f"{linear_better}/{total_comparisons}",
                'Linear Sig %': f"{avg_linear_sig:.1%}",
                'PCA Sig %': f"{avg_pca_sig:.1%}"
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        latex_table = comp_df.to_latex(
            index=False,
            caption='Linear vs PCA Model Comparison',
            label='tab:linear_pca_comparison',
            position='htbp',
            column_format='llccccccc',
            escape=False
        )
        
        with open('final_plots/statsmodels/tables/linear_pca_comparison.txt', 'w') as f:
            f.write(latex_table)
        print("✓ Generated linear_pca_comparison.txt")
    
    # Table 5: Top Performers by Statistical Significance
    top_performers = []
    
    for experiment in experiments:
        exp_data = df[df['experiment'] == experiment]
        
        # Sort by proportion of significant features (descending)
        exp_data_sorted = exp_data.sort_values('linear_proportion_significant_mean', ascending=False)
        
        # Get top 5 per experiment
        for idx, (_, row) in enumerate(exp_data_sorted.head(10).iterrows()):
            top_performers.append({
                'Rank': idx + 1,
                'Experiment': experiment.replace('_', ' ').title(),
                'Model': row['model'][:30] + '...' if len(row['model']) > 30 else row['model'],
                'Provider': row['provider'],
                'Size': int(row['size']),
                'Sig Features': f"{row['linear_proportion_significant_mean']:.1%}",
                'Linear R²': f"{row['linear_r2_mean']:.4f}",
                'F-p-value': f"{row['linear_f_pvalue_mean']:.2e}"
            })
    
    if top_performers:
        top_df = pd.DataFrame(top_performers)
        
        latex_table = top_df.to_latex(
            index=False,
            caption='Top Models by Statistical Significance',
            label='tab:top_significance',
            position='htbp',
            column_format='cllccccc',
            escape=False
        )
        
        with open('final_plots/statsmodels/tables/top_performers_significance.txt', 'w') as f:
            f.write(latex_table)
        print("✓ Generated top_performers_significance.txt")
    
    print(f"\nGenerated LaTeX tables for statsmodels analysis")

# Generate LaTeX tables
create_latex_tables()

# Summary statistics for statsmodels
print("\n" + "="*60)
print("STATSMODELS ANALYSIS SUMMARY")
print("="*60)

print("\nGENERAL MODELS (STATSMODELS):")
for experiment in general_models['experiment'].unique():
    exp_data = general_models[general_models['experiment'] == experiment]
    print(f"\n{experiment.upper()}:")
    
    for provider in exp_data['provider'].unique():
        provider_data = exp_data[exp_data['provider'] == provider]
        alpha_val = provider_data['alpha'].iloc[0] if len(provider_data) > 0 else 0.05
        significant_models = (provider_data['linear_f_pvalue_mean'] < alpha_val).sum()
        
        print(f"\n  {provider}:")
        print(f"    Models: {', '.join(provider_data['model'].unique())}")
        print(f"    Size range: {provider_data['size'].min()} - {provider_data['size'].max()}")
        print(f"    K-folds: {provider_data['k_folds'].iloc[0]}")
        print(f"    Significance level (α): {alpha_val}")
        print(f"    Linear R² range: {provider_data['linear_r2_mean'].min():.4f} - {provider_data['linear_r2_mean'].max():.4f}")
        print(f"    Significant features: {provider_data['linear_proportion_significant_mean'].min():.1%} - {provider_data['linear_proportion_significant_mean'].max():.1%}")
        print(f"    Statistically significant models: {significant_models}/{len(provider_data)}")
        print(f"    Average cross-validation std: Linear R²={provider_data['linear_r2_std'].mean():.4f}, PCA R²={provider_data['pca_r2_std'].mean():.4f}")

if len(specialist_models) > 0:
    print("\nSPECIALIST MODELS (STATSMODELS):")
    for experiment in specialist_models['experiment'].unique():
        exp_data = specialist_models[specialist_models['experiment'] == experiment]
        print(f"\n{experiment.upper()}:")
        
        for provider in exp_data['provider'].unique():
            provider_data = exp_data[exp_data['provider'] == provider]
            alpha_val = provider_data['alpha'].iloc[0] if len(provider_data) > 0 else 0.05
            significant_models = (provider_data['linear_f_pvalue_mean'] < alpha_val).sum()
            
            print(f"\n  {provider}:")
            print(f"    Models: {', '.join(provider_data['model'].unique())}")
            print(f"    Size range: {provider_data['size'].min()} - {provider_data['size'].max()}")
            print(f"    K-folds: {provider_data['k_folds'].iloc[0]}")
            print(f"    Significance level (α): {alpha_val}")
            print(f"    Linear R² range: {provider_data['linear_r2_mean'].min():.4f} - {provider_data['linear_r2_mean'].max():.4f}")
            print(f"    Significant features: {provider_data['linear_proportion_significant_mean'].min():.1%} - {provider_data['linear_proportion_significant_mean'].max():.1%}")
            print(f"    Statistically significant models: {significant_models}/{len(provider_data)}")
            print(f"    Average cross-validation std: Linear R²={provider_data['linear_r2_std'].mean():.4f}, PCA R²={provider_data['pca_r2_std'].mean():.4f}")

# Statistical significance insights
print("\n" + "="*60)
print("STATISTICAL SIGNIFICANCE INSIGHTS")
print("="*60)

overall_alpha = df['alpha'].iloc[0] if len(df) > 0 else 0.05
total_models_tested = len(df)
overall_significant = (df['linear_f_pvalue_mean'] < overall_alpha).sum()

print(f"\nOVERALL STATISTICS:")
print(f"  Total model-size combinations tested: {total_models_tested}")
print(f"  Significance level (α): {overall_alpha}")
print(f"  Models with significant F-tests: {overall_significant}/{total_models_tested} ({overall_significant/total_models_tested:.1%})")
print(f"  Average proportion of significant features: {df['linear_proportion_significant_mean'].mean():.1%}")
print(f"  Models with >10% significant features: {(df['linear_proportion_significant_mean'] > 0.1).sum()}/{total_models_tested}")
print(f"  Models with >50% significant features: {(df['linear_proportion_significant_mean'] > 0.5).sum()}/{total_models_tested}")

print(f"\n" + "="*60)
print("STATSMODELS FILES GENERATED:")
print("final_plots/statsmodels/")
print("├── general_models/ (performance + significance plots)")
print("├── specialist_models/ (performance + significance plots)")
print("├── by_provider/ (performance + significance plots)")
print("├── significance/ (summary significance analysis)")
print("└── tables/ (LaTeX tables with statistical metrics)")
print("="*60)

print(f"\n" + "="*60)
print("NEW STATSMODELS VISUALIZATIONS:")
print("- Linear Feature Significance Proportion vs Size")
print("- F-test P-values (Linear & PCA) vs Size")
print("- Model Selection Criteria (AIC/BIC)")
print("- Statistical Significance Heatmaps")
print("- Cross-validation error bars on all metrics")
print("- Comprehensive significance analysis tables")
print("="*60)