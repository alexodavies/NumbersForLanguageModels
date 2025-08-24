#!/usr/bin/env python3
"""
Simplified Sweep Experiments Module
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Your existing imports
from datasets import (
    real_positive_decimals, 
    real_positive_and_negative_decimals, 
    real_int_and_decimal
)
from linear_reconstruct import evaluate_linear_reconstruction
from pca_exp import evaluate_pca_reconstruction
from sklearn.decomposition import PCA


def vis_pca(embeddings, values, path):
    fig, ax = plt.subplots(figsize=(6,6))
    test_pca = PCA(n_components = 2).fit_transform(embeddings)
    ax.scatter(test_pca[:,0], test_pca[:,1], c = values)
    plt.tight_layout()
    plt.savefig(path)

@dataclass
class SweepConfig:
    """Simple sweep configuration."""
    n_samples: int = 300
    test_size: float = 0.2
    random_state: int = 42
    
    # Size ranges to test
    decimal_sizes: List[int] = None
    mixed_int_sizes: List[int] = None
    mixed_decimal_sizes: List[int] = None
    
    # Output directories
    plot_dir: str = "sweep_plots"
    results_dir: str = "sweep_results"
    
    def __post_init__(self):
        if self.decimal_sizes is None:
            self.decimal_sizes = [i for i in range(1,20)]
        if self.mixed_int_sizes is None:
            self.mixed_int_sizes = [i for i in range(1,20)]
        if self.mixed_decimal_sizes is None:
            self.mixed_decimal_sizes = [i for i in range(1,20)]
        
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


class NumberSizeSweep:
    """Test embedding performance as number sizes increase."""
    
    def __init__(self, embedding_wrapper, config: SweepConfig = None):
        self.wrapper = embedding_wrapper
        self.config = config or SweepConfig()
        
        # Get available models
        self.available_models = self._get_models()
        
        print(f"🔬 Sweep initialized")
        print(f"  Models: {self.available_models}")
        print(f"  Decimal sizes: {self.config.decimal_sizes}")
        
        self.results = {}
    
    def _get_models(self) -> List[str]:
        """Get available models."""
        try:
            available_services = self.wrapper.get_available_services()
            supported_models = self.wrapper.get_supported_models()
            
            models = []
            for service, is_available in available_services.items():
                if is_available:
                    models.extend(supported_models[service])
            
            # Limit to first 3 for efficiency
            return models if models else ["mock-model"]
        except:
            return ["mock-model"]
    
    def run_decimal_sweep(self, positive_only: bool = True) -> Dict[str, Any]:
        """Run sweep on decimal numbers of increasing size."""
        experiment_name = "positive_decimals" if positive_only else "mixed_sign_decimals"
        dataset_func = real_positive_decimals if positive_only else real_positive_and_negative_decimals
        
        print(f"\n{'='*50}")
        print(f"DECIMAL SWEEP: {experiment_name}")
        print(f"{'='*50}")
        
        results = {
            'experiment': experiment_name,
            'sizes': self.config.decimal_sizes,
            'models': {}
        }
        
        for model in self.available_models:
            print(f"\n🤖 Testing {model}")
            
            model_results = {
                'sizes': [],
                'linear_r2': [],
                'pca_r2': []
            }
            
            for size in self.config.decimal_sizes:
                print(f"  📏 Size {size}...")
                
                try:
                    # Generate dataset with fixed seed for reproducibility
                    import random
                    import numpy as np
                    seed = self.config.random_state + size  # Deterministic seed
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    texts = dataset_func(self.config.n_samples, size)
                    values = [float(x) for x in texts]
                    
                    # Get embeddings with parameter caching
                    if hasattr(self.wrapper, 'embed_with_params'):
                        embeddings = self.wrapper.embed_with_params(
                            texts, model, experiment_name, size, 
                            random_state=self.config.random_state
                        )
                    else:
                        embeddings = self.wrapper.embed(texts, model)
                    
                    print(f"    Range: {min(values):.3f} to {max(values):.3f}")
                    
                    # Run experiments
                    linear_results = evaluate_linear_reconstruction(
                        embeddings, values,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
                    
                    # Run PCA with 1 component first for R² score
                    pca_results_1 = evaluate_pca_reconstruction(
                        embeddings, values,
                        n_components=1,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
                    
                    # Run PCA with up to 5 components for explained variance analysis
                    max_components = min(5, len(embeddings[0]), len(values))
                    pca_results_5 = evaluate_pca_reconstruction(
                        embeddings, values,
                        n_components=max_components,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )

                    vis_pca(embeddings, values, f"plots/sweep/pca_vis/{experiment_name}-{model}-{size}.png")
                    
                    # Store results
                    model_results['sizes'].append(size)
                    model_results['linear_r2'].append(linear_results['test_r2'])
                    model_results['pca_r2'].append(pca_results_1['test_r2'])  # Use 1-component for R²
                    
                    # Store explained variance for up to 5 components
                    explained_var = pca_results_5['explained_variance_ratios']
                    for i in range(5):
                        key = f'pca_explained_var_{i+1}'
                        if key not in model_results:
                            model_results[key] = []
                        
                        if i < len(explained_var):
                            model_results[key].append(explained_var[i])
                        else:
                            model_results[key].append(0.0)  # If fewer components available
                    
                    # Calculate cumulative explained variance
                    if 'pca_cumulative_var' not in model_results:
                        model_results['pca_cumulative_var'] = []
                    
                    cumulative_var = sum(explained_var)
                    model_results['pca_cumulative_var'].append(cumulative_var)
                    
                    print(f"    Linear R²: {linear_results['test_r2']:.3f}, "
                          f"PCA R²: {pca_results_1['test_r2']:.3f}")
                    print(f"    PCA explained variance (top {len(explained_var)}): {[f'{v:.3f}' for v in explained_var]}")
                    print(f"    Cumulative: {cumulative_var:.3f}")
                
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    model_results['sizes'].append(size)
                    model_results['linear_r2'].append(np.nan)
                    model_results['pca_r2'].append(np.nan)
                    
                    # Add NaN for all PCA variance components
                    for i in range(5):
                        key = f'pca_explained_var_{i+1}'
                        if key not in model_results:
                            model_results[key] = []
                        model_results[key].append(np.nan)
                    
                    if 'pca_cumulative_var' not in model_results:
                        model_results['pca_cumulative_var'] = []
                    model_results['pca_cumulative_var'].append(np.nan)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    def run_mixed_sweep(self) -> Dict[str, Any]:
        """Run sweep on mixed integer/decimal numbers."""
        experiment_name = "mixed_int_decimal"
        
        print(f"\n{'='*50}")
        print(f"MIXED INT/DECIMAL SWEEP")
        print(f"{'='*50}")
        
        results = {
            'experiment': experiment_name,
            'models': {}
        }
        
        # Test varying integer size (decimal=3)
        fixed_decimal = 3
        
        for model in self.available_models:
            print(f"\n🤖 Testing {model}")
            
            model_results = {
                'sizes': [],
                'linear_r2': [],
                'pca_r2': []
            }
            
            for int_size in self.config.mixed_int_sizes:
                print(f"  📏 Int={int_size}, Dec={fixed_decimal}...")
                
                try:
                    # Generate mixed dataset with fixed seed
                    import random
                    import numpy as np
                    seed = self.config.random_state + int_size + 1000  # Different seed space
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    texts = real_int_and_decimal(
                        self.config.n_samples, 
                        int_size, 
                        int_size
                    )
                    values = [float(x) for x in texts]
                    
                    # Get embeddings with parameter caching
                    if hasattr(self.wrapper, 'embed_with_params'):
                        embeddings = self.wrapper.embed_with_params(
                            texts, model, experiment_name, int_size,
                            decimal_size=fixed_decimal,
                            random_state=self.config.random_state
                        )
                    else:
                        embeddings = self.wrapper.embed(texts, model)
                    
                    print(f"    Range: {min(values):.3f} to {max(values):.3f}")
                    
                    # Run experiments
                    linear_results = evaluate_linear_reconstruction(
                        embeddings, values,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
                    
                    # Run PCA with 1 component first for R² score
                    pca_results_1 = evaluate_pca_reconstruction(
                        embeddings, values,
                        n_components=1,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
                    
                    # Run PCA with up to 5 components for explained variance
                    max_components = min(5, len(embeddings[0]), len(values))
                    pca_results_5 = evaluate_pca_reconstruction(
                        embeddings, values,
                        n_components=max_components,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )

                    vis_pca(embeddings, values, f"plots/sweep/pca_vis/mixed-decimals-integers-{model}-{int_size}.png")
                    
                    # Store results
                    model_results['sizes'].append(int_size)
                    model_results['linear_r2'].append(linear_results['test_r2'])
                    model_results['pca_r2'].append(pca_results_1['test_r2'])
                    
                    # Store explained variance for up to 5 components
                    explained_var = pca_results_5['explained_variance_ratios']
                    for i in range(5):
                        key = f'pca_explained_var_{i+1}'
                        if key not in model_results:
                            model_results[key] = []
                        
                        if i < len(explained_var):
                            model_results[key].append(explained_var[i])
                        else:
                            model_results[key].append(0.0)
                    
                    # Cumulative explained variance
                    if 'pca_cumulative_var' not in model_results:
                        model_results['pca_cumulative_var'] = []
                    
                    cumulative_var = sum(explained_var)
                    model_results['pca_cumulative_var'].append(cumulative_var)
                    
                    print(f"    Linear R²: {linear_results['test_r2']:.3f}, "
                          f"PCA R²: {pca_results_1['test_r2']:.3f}")
                    print(f"    PCA explained variance (top {len(explained_var)}): {[f'{v:.3f}' for v in explained_var]}")
                    print(f"    Cumulative: {cumulative_var:.3f}")
                
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    model_results['sizes'].append(int_size)
                    model_results['linear_r2'].append(np.nan)
                    model_results['pca_r2'].append(np.nan)
                    
                    # Add NaN for all PCA variance components
                    for i in range(5):
                        key = f'pca_explained_var_{i+1}'
                        if key not in model_results:
                            model_results[key] = []
                        model_results[key].append(np.nan)
                    
                    if 'pca_cumulative_var' not in model_results:
                        model_results['pca_cumulative_var'] = []
                    model_results['pca_cumulative_var'].append(np.nan)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    def run_all_sweeps(self) -> Dict[str, Any]:
        """Run all sweep experiments."""
        print("🔬 Running all sweep experiments...")
        
        all_results = {}
        all_results['positive_decimals'] = self.run_decimal_sweep(positive_only=True)
        all_results['mixed_decimals'] = self.run_decimal_sweep(positive_only=False)
        all_results['mixed_int_decimal'] = self.run_mixed_sweep()
        
        return all_results
    
    def plot_sweep_results(self, experiment_name: str = None) -> None:
        """Create plots for results."""
        if not self.results:
            print("No results to plot")
            return
        
        experiments = [experiment_name] if experiment_name else list(self.results.keys())
        
        for exp_name in experiments:
            if exp_name not in self.results:
                continue
            
            self._plot_experiment(exp_name)
    
    def _plot_experiment(self, exp_name: str) -> None:
        """Plot results for one experiment with PCA component analysis."""
        results = self.results[exp_name]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: Linear performance
        for model, model_results in results['models'].items():
            ax1.plot(model_results['sizes'], model_results['linear_r2'], 
                    marker='o', label=model, linewidth=2)
        
        ax1.set_xlabel('Size (digits)')
        ax1.set_ylabel('Linear Test R²')
        ax1.set_title(f'Linear Performance - {exp_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: PCA performance (component 1 only)
        for model, model_results in results['models'].items():
            ax2.plot(model_results['sizes'], model_results['pca_r2'], 
                    marker='s', label=model, linewidth=2)
        
        ax2.set_xlabel('Size (digits)')
        ax2.set_ylabel('PCA Test R² (Component 1)')
        ax2.set_title(f'PCA Performance - {exp_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: First PCA component explained variance
        for model, model_results in results['models'].items():
            if 'pca_explained_var_1' in model_results:
                ax3.plot(model_results['sizes'], model_results['pca_explained_var_1'], 
                        marker='^', label=model, linewidth=2)
        
        ax3.set_xlabel('Size (digits)')
        ax3.set_ylabel('PCA Component 1 Explained Variance')
        ax3.set_title(f'First Component Explained Variance - {exp_name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Cumulative explained variance (top 5 components)
        for model, model_results in results['models'].items():
            if 'pca_cumulative_var' in model_results:
                ax4.plot(model_results['sizes'], model_results['pca_cumulative_var'], 
                        marker='d', label=model, linewidth=2)
        
        ax4.set_xlabel('Size (digits)')
        ax4.set_ylabel('Cumulative Explained Variance (Top 5)')
        ax4.set_title(f'Cumulative PCA Variance - {exp_name}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save
        filename = f"sweep_{exp_name}.png"
        filepath = os.path.join(self.config.plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Saved: {filepath}")
    
    def export_sweep_results(self, filename: str = None) -> None:
        """Export results to CSV with PCA component analysis."""
        if filename is None:
            filename = os.path.join(self.config.results_dir, "number_size_sweep_results.csv")
        
        if not self.results:
            print("No results to export")
            return
        
        rows = []
        
        for exp_name, results in self.results.items():
            for model, model_results in results['models'].items():
                for i, size in enumerate(model_results['sizes']):
                    if i < len(model_results['linear_r2']):
                        row = {
                            'experiment': exp_name,
                            'model': model,
                            'size': size,
                            'linear_r2': model_results['linear_r2'][i],
                            'pca_r2': model_results['pca_r2'][i]
                        }
                        
                        # Add PCA explained variance components
                        for j in range(1, 6):
                            var_key = f'pca_explained_var_{j}'
                            if var_key in model_results and i < len(model_results[var_key]):
                                row[f'pca_var_comp_{j}'] = model_results[var_key][i]
                            else:
                                row[f'pca_var_comp_{j}'] = np.nan
                        
                        # Add cumulative explained variance
                        if 'pca_cumulative_var' in model_results and i < len(model_results['pca_cumulative_var']):
                            row['pca_cumulative_var_5'] = model_results['pca_cumulative_var'][i]
                        else:
                            row['pca_cumulative_var_5'] = np.nan
                        
                        rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        print(f"📊 Exported {len(df)} results to {filename}")
        print("   Columns include PCA explained variance for components 1-5")
        print("   and cumulative explained variance")
    
    def generate_sweep_report(self) -> str:
        """Generate summary report."""
        if not self.results:
            return "No results available."
        
        lines = []
        lines.append("NUMBER SIZE SWEEP RESULTS")
        lines.append("=" * 40)
        lines.append("")
        
        for exp_name, results in self.results.items():
            lines.append(f"EXPERIMENT: {exp_name.upper()}")
            lines.append("-" * 30)
            
            # Calculate average performance
            all_linear = []
            for model_results in results['models'].values():
                all_linear.extend([x for x in model_results['linear_r2'] if not np.isnan(x)])
            
            if all_linear:
                avg_perf = np.mean(all_linear)
                lines.append(f"Average Linear R²: {avg_perf:.3f}")
                
                # Find trend
                sizes = results.get('sizes', results['models'][list(results['models'].keys())[0]]['sizes'])
                first_scores = []
                last_scores = []
                
                for model_results in results['models'].values():
                    linear_scores = [x for x in model_results['linear_r2'] if not np.isnan(x)]
                    if len(linear_scores) > 1:
                        first_scores.append(linear_scores[0])
                        last_scores.append(linear_scores[-1])
                
                if first_scores and last_scores:
                    trend_change = np.mean(last_scores) - np.mean(first_scores)
                    trend = "declining" if trend_change < -0.05 else "improving" if trend_change > 0.05 else "stable"
                    lines.append(f"Performance trend: {trend} (Δ = {trend_change:+.3f})")
            
            lines.append("")
        
        # Overall insights
        lines.append("KEY INSIGHTS:")
        lines.append("• Larger numbers generally become harder to embed")
        lines.append("• PCA Component 1 shows how much numerical info is in the primary direction")
        lines.append("• Cumulative variance (top 5) shows total numerical info captured")
        lines.append("• Check plots for detailed performance and variance curves")
        lines.append("• CSV contains all 5 PCA components for detailed analysis")
        
        return "\n".join(lines)
    
    def save_sweep_report(self, filename: str = None) -> None:
        """Save report to file."""
        if filename is None:
            filename = os.path.join(self.config.results_dir, "number_size_sweep_report.txt")
        
        report = self.generate_sweep_report()
        with open(filename, "w") as f:
            f.write(report)
        
        print(f"📋 Saved report: {filename}")


def run_quick_sweep_demo(embedding_wrapper, models_to_test: List[str] = None):
    """Quick demo of sweep functionality."""
    print("🚀 Quick Sweep Demo")
    print("=" * 40)
    
    # Small config for demo
    config = SweepConfig(
        n_samples=50,
        decimal_sizes=[i for i in range(1,10)],
        mixed_int_sizes=[i for i in range(1,10)],
        plot_dir="demo_sweep_plots",
        results_dir="demo_sweep_results"
    )
    
    # Initialize sweep
    sweep = NumberSizeSweep(embedding_wrapper, config)
    
    # Filter models if specified
    if models_to_test:
        available = set(sweep.available_models)
        requested = set(models_to_test)
        sweep.available_models = list(available & requested)
        
        if not sweep.available_models:
            print("❌ No valid models")
            return
    
    print(f"Testing: {sweep.available_models}")
    
    try:
        # Run just positive decimals
        sweep.run_decimal_sweep(positive_only=True)
        
        # Create outputs
        sweep.plot_sweep_results('positive_decimals')
        sweep.export_sweep_results()
        
        # Show report
        report = sweep.generate_sweep_report()
        print("\n" + report)
        
        print("\n✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()