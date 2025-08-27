import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load the data
df = pd.read_csv('results/sweep/number_size_sweep_results.csv')

# Set up the plot style
plt.style.use('default')

# Create directory structure
os.makedirs('final_plots/general_models', exist_ok=True)
os.makedirs('final_plots/specialist_models', exist_ok=True)
os.makedirs('final_plots/by_provider/openai', exist_ok=True)
os.makedirs('final_plots/by_provider/google', exist_ok=True)
os.makedirs('final_plots/by_provider/voyage', exist_ok=True)

# Create a mapping for provider and model type
def get_provider_and_type(model_name):
    model_lower = model_name.lower()
    
    # Exclude multimodal models
    if 'multimodal' in model_lower:
        return None, None
    
    if any(openai_model in model_lower for openai_model in ['text-embedding', 'ada']):
        provider = 'OpenAI'
        # OpenAI general models
        is_specialist = False
    elif 'gemini' in model_lower:
        provider = 'Google'
        # Google general models
        is_specialist = False
    elif 'voyage' in model_lower:
        provider = 'Voyage'
        # Voyage specialist models
        if any(specialist in model_lower for specialist in ['finance', 'law', 'code']):
            is_specialist = True
        else:
            is_specialist = False
    else:
        provider = 'Other'
        is_specialist = False
    
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
    """Create the four plot types for a given data subset"""
    
    for experiment in data_subset['experiment'].unique():
        exp_data = data_subset[data_subset['experiment'] == experiment].copy()
        exp_data = exp_data.sort_values('size')
        
        if len(exp_data) == 0:
            continue
        
        # Calculate outlier bounds for R2 values
        linear_r2_lower, linear_r2_upper = detect_outliers_iqr(exp_data, 'linear_r2')
        pca_r2_lower, pca_r2_upper = detect_outliers_iqr(exp_data, 'pca_r2')
        
        # Figure 1: Linear R² vs Number Precision
        plt.figure(figsize=(12, 5))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.plot(model_data['size'], model_data['linear_r2'], 
                    color=provider_colors[provider],
                    linestyle=line_styles[model_idx % len(line_styles)],
                    marker=markers[model_idx % len(markers)],
                    linewidth=2, markersize=6, 
                    label=f'{model}')
            model_idx += 1
        
        plt.xlabel('Number Precision (size)')
        plt.ylabel('Linear R²')
        plt.ylim(linear_r2_lower - 0.05, linear_r2_upper + 0.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/linear_r2_vs_precision_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: Explained Variance Ratio vs Number Precision
        plt.figure(figsize=(12, 5))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.plot(model_data['size'], model_data['pca_var_comp_1'], 
                    color=provider_colors[provider],
                    linestyle=line_styles[model_idx % len(line_styles)],
                    marker=markers[model_idx % len(markers)],
                    linewidth=2, markersize=6, 
                    label=f'{model}')
            model_idx += 1
        
        plt.xlabel('Number Precision (size)')
        plt.ylabel('PCA Component 1 Explained Variance Ratio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/explained_variance_vs_precision_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 3: PCA R² vs Number Precision
        plt.figure(figsize=(12, 5))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.plot(model_data['size'], model_data['pca_r2'], 
                    color=provider_colors[provider],
                    linestyle=line_styles[model_idx % len(line_styles)],
                    marker=markers[model_idx % len(markers)],
                    linewidth=2, markersize=6, 
                    label=f'{model}')
            model_idx += 1
        
        plt.xlabel('Number Precision (size)')
        plt.ylabel('PCA R²')
        plt.ylim(max(0,pca_r2_lower - 0.05), pca_r2_upper + 0.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/pca_r2_vs_precision_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 4: PCA R² vs Linear R²
        plt.figure(figsize=(10, 8))
        
        model_idx = 0
        for model in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            plt.scatter(model_data['linear_r2'], model_data['pca_r2'],
                       c=provider_colors[provider],
                       marker=markers[model_idx % len(markers)],
                       s=100, alpha=0.7,
                       label=f'{model}')
            model_idx += 1
        
        plt.xlabel('Linear Model R²')
        plt.ylabel('PCA R²')
        plt.xlim(linear_r2_lower - 0.05, linear_r2_upper + 0.05)
        plt.ylim(pca_r2_lower - 0.05, pca_r2_upper + 0.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add diagonal reference line
        min_val = min(plt.xlim()[0], plt.ylim()[0])
        max_val = max(plt.xlim()[1], plt.ylim()[1])
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='y=x')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/pca_vs_linear_r2_{experiment}.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_multi_axis_plots(data_subset, plot_type, save_dir, title_suffix=""):
    """Create multi-axis plots showing all three datasets together"""
    
    if len(data_subset) == 0:
        return
    
    # Get global bounds for consistent y-axis scaling
    linear_r2_lower, linear_r2_upper = detect_outliers_iqr(data_subset, 'linear_r2')
    pca_r2_lower, pca_r2_upper = detect_outliers_iqr(data_subset, 'pca_r2')
    pca_var_lower, pca_var_upper = detect_outliers_iqr(data_subset, 'pca_var_comp_1')
    
    # Create figure with subplots for each experiment
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(title_suffix.strip(), fontsize=16, fontweight='bold') if title_suffix else None
    
    experiment_names = sorted(data_subset['experiment'].unique())
    
    # Plot 1: Linear R² vs Number Precision (all experiments)
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
            
            ax.plot(model_data['size'], model_data['linear_r2'], 
                   color=provider_colors[provider],
                   linestyle=line_styles[model_idx % len(line_styles)],
                   marker=markers[model_idx % len(markers)],
                   linewidth=2, markersize=6, 
                   label=f'{model}')
            model_idx += 1
        
        ax.set_xlabel('Number Precision (size)')
        if idx == 0:
            ax.set_ylabel('Linear R²')
        ax.set_title(experiment.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylim(linear_r2_lower - 0.05, linear_r2_upper + 0.05)
        ax.grid(True, alpha=0.3)
        
    
    # Create a single legend at the bottom for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=min(len(labels), 4))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for bottom legend
    plt.savefig(f'{save_dir}/multi_linear_r2_vs_precision.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Explained Variance Ratio vs Number Precision (all experiments)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
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
            
            ax.plot(model_data['size'], model_data['pca_var_comp_1'], 
                   color=provider_colors[provider],
                   linestyle=line_styles[model_idx % len(line_styles)],
                   marker=markers[model_idx % len(markers)],
                   linewidth=2, markersize=6, 
                   label=f'{model}')
            model_idx += 1
        
        ax.set_xlabel('Number Precision (size)')
        if idx == 0:
            ax.set_ylabel('PCA Component 1 Explained Variance Ratio')
        ax.set_title(experiment.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylim(max(0, pca_var_lower - 0.05), pca_var_upper + 0.05)
        ax.grid(True, alpha=0.3)
        
    
    # Create a single legend at the bottom for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=min(len(labels), 4))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for bottom legend
    plt.savefig(f'{save_dir}/multi_explained_variance_vs_precision.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: PCA R² vs Number Precision (all experiments)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
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
            
            ax.plot(model_data['size'], model_data['pca_r2'], 
                   color=provider_colors[provider],
                   linestyle=line_styles[model_idx % len(line_styles)],
                   marker=markers[model_idx % len(markers)],
                   linewidth=2, markersize=6, 
                   label=f'{model}')
            model_idx += 1
        
        ax.set_xlabel('Number Precision (size)')
        if idx == 0:
            ax.set_ylabel('PCA R²')
        ax.set_title(experiment.replace('_', ' ').title(), fontweight='bold')
        plt.ylim(pca_r2_lower - 0.05, pca_r2_upper + 0.05)
        ax.grid(True, alpha=0.3)
        
    
    # Create a single legend at the bottom for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=min(len(labels), 4))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for bottom legend
    plt.savefig(f'{save_dir}/multi_pca_r2_vs_precision.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: PCA R² vs Linear R² (all experiments)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    for idx, experiment in enumerate(experiment_names):
        ax = axes[idx]
        exp_data = data_subset[data_subset['experiment'] == experiment].copy()
        
        if len(exp_data) == 0:
            continue
        
        model_idx = 0
        for model in sorted(exp_data['model'].unique()):
            model_data = exp_data[exp_data['model'] == model]
            provider = model_data['provider'].iloc[0]
            
            ax.scatter(model_data['linear_r2'], model_data['pca_r2'],
                      c=provider_colors[provider],
                      marker=markers[model_idx % len(markers)],
                      s=100, alpha=0.7,
                      label=f'{model}')
            model_idx += 1
        
        ax.set_xlabel('Linear Model R²')
        if idx == 0:
            ax.set_ylabel('PCA R²')
        ax.set_title(experiment.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlim(linear_r2_lower - 0.05, linear_r2_upper + 0.05)
        ax.set_ylim(pca_r2_lower - 0.05, pca_r2_upper + 0.05)
        ax.grid(True, alpha=0.3)
        
        # Add diagonal reference line
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
        
    
    # Create a single legend at the bottom for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=min(len(labels), 4))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for bottom legend
    plt.savefig(f'{save_dir}/multi_pca_vs_linear_r2.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create multi-axis plots for general models
print("\n" + "="*60)
print("CREATING MULTI-AXIS PLOTS FOR GENERAL MODELS")
print("="*60)
create_multi_axis_plots(general_models, 'general', 'final_plots/general_models', 'General Models')

# Create multi-axis plots for specialist models
if len(specialist_models) > 0:
    print("\n" + "="*60)
    print("CREATING MULTI-AXIS PLOTS FOR SPECIALIST MODELS")
    print("="*60)
    create_multi_axis_plots(specialist_models, 'specialist', 'final_plots/specialist_models', 'Specialist Models')

# Create multi-axis plots by provider (general models only)
print("\n" + "="*60)
print("CREATING MULTI-AXIS PLOTS BY PROVIDER (GENERAL MODELS ONLY)")
print("="*60)

for provider in general_models['provider'].unique():
    print(f"\nProcessing multi-axis plots for {provider}...")
    provider_data = general_models[general_models['provider'] == provider]
    provider_dir = f"final_plots/by_provider/{provider.lower()}"
    create_multi_axis_plots(provider_data, 'provider', provider_dir, f'{provider} Models')

# Create multi-axis plots by provider for specialist models (if any)
if len(specialist_models) > 0:
    for provider in specialist_models['provider'].unique():
        print(f"\nProcessing multi-axis plots for {provider} specialist models...")
        provider_data = specialist_models[specialist_models['provider'] == provider]
        provider_dir = f"final_plots/by_provider/{provider.lower()}_specialist"
        create_multi_axis_plots(provider_data, 'specialist_provider', provider_dir, f'{provider} Specialist Models')

# Create plots for general models
print("\n" + "="*60)
print("CREATING INDIVIDUAL PLOTS FOR GENERAL MODELS")
print("="*60)
create_plots(general_models, 'general', 'final_plots/general_models')

# Create plots for specialist models
print("\n" + "="*60)
print("CREATING INDIVIDUAL PLOTS FOR SPECIALIST MODELS")
print("="*60)
create_plots(specialist_models, 'specialist', 'final_plots/specialist_models')

# Create plots by provider (general models only)
print("\n" + "="*60)
print("CREATING INDIVIDUAL PLOTS BY PROVIDER (GENERAL MODELS ONLY)")
print("="*60)

for provider in general_models['provider'].unique():
    print(f"\nProcessing {provider}...")
    provider_data = general_models[general_models['provider'] == provider]
    provider_dir = f"final_plots/by_provider/{provider.lower()}"
    create_plots(provider_data, 'provider', provider_dir)

# Create plots by provider for specialist models (if any)
if len(specialist_models) > 0:
    print("\n" + "="*60)
    print("CREATING INDIVIDUAL PLOTS BY PROVIDER (SPECIALIST MODELS)")
    print("="*60)
    
    for provider in specialist_models['provider'].unique():
        print(f"\nProcessing {provider} specialist models...")
        provider_data = specialist_models[specialist_models['provider'] == provider]
        provider_dir = f"final_plots/by_provider/{provider.lower()}_specialist"
        os.makedirs(provider_dir, exist_ok=True)
        create_plots(provider_data, 'specialist_provider', provider_dir)

def create_latex_tables():
    """Generate LaTeX tables using pandas to_latex method"""
    
    # Create tables directory
    os.makedirs('final_plots/tables', exist_ok=True)
    
    print("\nGenerating LaTeX tables using pandas...")
    
    # Table 1: Model Overview by Provider
    size_ranges = df.groupby('model').agg({
        'size': ['min', 'max'],
        'provider': 'first',
        'is_specialist': 'first'
    }).reset_index()
    size_ranges.columns = ['Model', 'Min Size', 'Max Size', 'Provider', 'Is Specialist']
    size_ranges['Type'] = size_ranges['Is Specialist'].map({True: 'Specialist', False: 'General'})
    size_ranges['Size Range'] = size_ranges['Min Size'].astype(int).astype(str) + '-' + size_ranges['Max Size'].astype(int).astype(str)
    
    model_overview = size_ranges[['Provider', 'Model', 'Type', 'Size Range']].sort_values(['Provider', 'Type', 'Model'])
    
    latex_table = model_overview.to_latex(
        index=False,
        caption='Model Overview by Provider',
        label='tab:model_overview',
        position='htbp',
        column_format='llcc',
        escape=False
    )
    
    with open('final_plots/tables/model_overview.txt', 'w') as f:
        f.write(latex_table)
    
    # Table 2: Performance Summary by Experiment and Provider
    for experiment in experiments:
        exp_data = df[df['experiment'] == experiment]
        
        # Calculate statistics by provider
        stats_list = []
        for provider in sorted(exp_data['provider'].unique()):
            provider_data = exp_data[exp_data['provider'] == provider]
            
            stats_list.append({
                'Provider': provider,
                'Models': len(provider_data['model'].unique()),
                'Linear R² (μ±σ)': f"{provider_data['linear_r2'].mean():.3f}±{provider_data['linear_r2'].std():.3f}",
                'PCA R² (μ±σ)': f"{provider_data['pca_r2'].mean():.3f}±{provider_data['pca_r2'].std():.3f}",
                'PCA Var Comp 1 (μ±σ)': f"{provider_data['pca_var_comp_1'].mean():.3f}±{provider_data['pca_var_comp_1'].std():.3f}"
            })
        
        stats_df = pd.DataFrame(stats_list)
        
        latex_table = stats_df.to_latex(
            index=False,
            caption=f'Performance Summary - {experiment.replace("_", " ").title()}',
            label=f'tab:perf_{experiment}',
            position='htbp',
            column_format='lccccc',
            escape=False
        )
        
        with open(f'final_plots/tables/performance_summary_{experiment}.txt', 'w') as f:
            f.write(latex_table)
    
    # Table 3: Best Performance by Size for each Experiment
    for experiment in experiments:
        exp_data = df[df['experiment'] == experiment]
        
        # Find best performing model at each size
        best_models = []
        for size in sorted(exp_data['size'].unique()):
            size_data = exp_data[exp_data['size'] == size]
            
            # Best linear R²
            best_linear = size_data.loc[size_data['linear_r2'].idxmax()]
            # Best PCA R²
            best_pca = size_data.loc[size_data['pca_r2'].idxmax()]
            
            best_models.append({
                'Size': int(size),
                'Best Linear Model': best_linear['model'],
                'Linear R²': f"{best_linear['linear_r2']:.4f}",
                'Best PCA Model': best_pca['model'],
                'PCA R²': f"{best_pca['pca_r2']:.4f}"
            })
        
        best_df = pd.DataFrame(best_models)
        
        latex_table = best_df.to_latex(
            index=False,
            caption=f'Best Performing Models by Size - {experiment.replace("_", " ").title()}',
            label=f'tab:best_{experiment}',
            position='htbp',
            column_format='ccccc',
            escape=False
        )
        
        with open(f'final_plots/tables/best_performance_{experiment}.txt', 'w') as f:
            f.write(latex_table)
    
    # Table 4: Model-Experiment Correlation Analysis
    correlation_list = []
    for experiment in experiments:
        exp_data = df[df['experiment'] == experiment]
        
        for model in sorted(exp_data['model'].unique()):
            model_data = exp_data[exp_data['model'] == model]
            
            if len(model_data) > 2:  # Need at least 3 points for meaningful correlation
                corr_size_linear = model_data['size'].corr(model_data['linear_r2'])
                corr_size_pca = model_data['size'].corr(model_data['pca_r2'])
                corr_size_var = model_data['size'].corr(model_data['pca_var_comp_1'])
                
                correlation_list.append({
                    'Experiment': experiment.replace('_', ' ').title(),
                    'Model': model,
                    'Provider': model_data['provider'].iloc[0],
                    'Size vs Linear R²': f"{corr_size_linear:.3f}" if not pd.isna(corr_size_linear) else 'N/A',
                    'Size vs PCA R²': f"{corr_size_pca:.3f}" if not pd.isna(corr_size_pca) else 'N/A',
                    'Size vs PCA Var': f"{corr_size_var:.3f}" if not pd.isna(corr_size_var) else 'N/A'
                })
    
    if correlation_list:
        corr_df = pd.DataFrame(correlation_list)
        
        latex_table = corr_df.to_latex(
            index=False,
            caption='Model-Level Correlation with Precision (Size)',
            label='tab:model_correlations',
            position='htbp',
            column_format='llcccc',
            escape=False
        )
        
        with open('final_plots/tables/model_correlations.txt', 'w') as f:
            f.write(latex_table)
    
    # Table 5: Overall Correlation Analysis by Experiment
    overall_correlation_data = []
    for experiment in experiments:
        exp_data = df[df['experiment'] == experiment]
        
        # Overall correlations
        corr_linear_pca = exp_data['linear_r2'].corr(exp_data['pca_r2'])
        corr_size_linear = exp_data['size'].corr(exp_data['linear_r2'])
        corr_size_pca = exp_data['size'].corr(exp_data['pca_r2'])
        corr_size_var = exp_data['size'].corr(exp_data['pca_var_comp_1'])
        
        overall_correlation_data.append({
            'Experiment': experiment.replace('_', ' ').title(),
            'Linear vs PCA R²': f"{corr_linear_pca:.3f}",
            'Size vs Linear R²': f"{corr_size_linear:.3f}",
            'Size vs PCA R²': f"{corr_size_pca:.3f}",
            'Size vs PCA Var': f"{corr_size_var:.3f}"
        })
    
    overall_corr_df = pd.DataFrame(overall_correlation_data)
    
    latex_table = overall_corr_df.to_latex(
        index=False,
        caption='Overall Correlation Analysis Across Experiments',
        label='tab:overall_correlations',
        position='htbp',
        column_format='lcccc',
        escape=False
    )
    
    with open('final_plots/tables/overall_correlations.txt', 'w') as f:
        f.write(latex_table)
    
    # Table 6: General vs Specialist Model Comparison (if applicable)
    if len(specialist_models) > 0:
        comparison_data = []
        for experiment in experiments:
            exp_general = general_models[general_models['experiment'] == experiment]
            exp_specialist = specialist_models[specialist_models['experiment'] == experiment]
            
            if len(exp_general) > 0 and len(exp_specialist) > 0:
                comparison_data.append({
                    'Experiment': experiment.replace('_', ' ').title(),
                    'General Models': len(exp_general['model'].unique()),
                    'General Linear R²': f"{exp_general['linear_r2'].mean():.3f}±{exp_general['linear_r2'].std():.3f}",
                    'General PCA R²': f"{exp_general['pca_r2'].mean():.3f}±{exp_general['pca_r2'].std():.3f}",
                    'Specialist Models': len(exp_specialist['model'].unique()),
                    'Specialist Linear R²': f"{exp_specialist['linear_r2'].mean():.3f}±{exp_specialist['linear_r2'].std():.3f}",
                    'Specialist PCA R²': f"{exp_specialist['pca_r2'].mean():.3f}±{exp_specialist['pca_r2'].std():.3f}"
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            latex_table = comp_df.to_latex(
                index=False,
                caption='General vs Specialist Model Performance',
                label='tab:general_specialist',
                position='htbp',
                column_format='lccccccc',
                escape=False
            )
            
            with open('final_plots/tables/general_vs_specialist.txt', 'w') as f:
                f.write(latex_table)
    
    # Table 7: Provider Performance Summary (aggregated across all experiments)
    provider_summary = []
    for provider in sorted(df['provider'].unique()):
        provider_data = df[df['provider'] == provider]
        
        provider_summary.append({
            'Provider': provider,
            'Total Models': len(provider_data['model'].unique()),
            'General Models': len(provider_data[provider_data['is_specialist'] == False]['model'].unique()),
            'Specialist Models': len(provider_data[provider_data['is_specialist'] == True]['model'].unique()),
            'Avg Linear R²': f"{provider_data['linear_r2'].mean():.3f}",
            'Avg PCA R²': f"{provider_data['pca_r2'].mean():.3f}",
            'Avg PCA Var': f"{provider_data['pca_var_comp_1'].mean():.3f}"
        })
    
    provider_df = pd.DataFrame(provider_summary)
    
    latex_table = provider_df.to_latex(
        index=False,
        caption='Provider Performance Summary (All Experiments)',
        label='tab:provider_summary',
        position='htbp',
        column_format='lcccccc',
        escape=False
    )
    
    with open('final_plots/tables/provider_summary.txt', 'w') as f:
        f.write(latex_table)

# Generate LaTeX tables
create_latex_tables()

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nGENERAL MODELS:")
for experiment in general_models['experiment'].unique():
    exp_data = general_models[general_models['experiment'] == experiment]
    print(f"\n{experiment.upper()}:")
    
    for provider in exp_data['provider'].unique():
        provider_data = exp_data[exp_data['provider'] == provider]
        print(f"\n  {provider}:")
        print(f"    Models: {', '.join(provider_data['model'].unique())}")
        print(f"    Size range: {provider_data['size'].min()} - {provider_data['size'].max()}")
        print(f"    Linear R² range: {provider_data['linear_r2'].min():.4f} - {provider_data['linear_r2'].max():.4f}")
        print(f"    PCA R² range: {provider_data['pca_r2'].min():.4f} - {provider_data['pca_r2'].max():.4f}")

if len(specialist_models) > 0:
    print("\nSPECIALIST MODELS:")
    for experiment in specialist_models['experiment'].unique():
        exp_data = specialist_models[specialist_models['experiment'] == experiment]
        print(f"\n{experiment.upper()}:")
        
        for provider in exp_data['provider'].unique():
            provider_data = exp_data[exp_data['provider'] == provider]
            print(f"\n  {provider}:")
            print(f"    Models: {', '.join(provider_data['model'].unique())}")
            print(f"    Size range: {provider_data['size'].min()} - {provider_data['size'].max()}")
            print(f"    Linear R² range: {provider_data['linear_r2'].min():.4f} - {provider_data['linear_r2'].max():.4f}")
            print(f"    PCA R² range: {provider_data['pca_r2'].min():.4f} - {provider_data['pca_r2'].max():.4f}")

print(f"\n" + "="*60)
print("LATEX TABLES GENERATED (.txt files using pandas to_latex):")
print("final_plots/tables/")
print("├── model_overview.txt")
print("├── performance_summary_[experiment].txt")
print("├── best_performance_[experiment].txt")
print("├── model_correlations.txt (model-level precision correlations)")
print("├── overall_correlations.txt")
print("├── provider_summary.txt")
print("└── general_vs_specialist.txt (if applicable)")
print("="*60)

print(f"\n" + "="*60)
print("DIRECTORY STRUCTURE CREATED:")
print("final_plots/")
print("├── general_models/")
print("├── specialist_models/")
print("├── by_provider/")
print("│   ├── openai/")
print("│   ├── google/")
print("│   └── voyage/")
print("└── tables/")
print("="*60)