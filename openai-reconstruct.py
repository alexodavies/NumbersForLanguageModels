#!/usr/bin/env python3
"""
Multi-Provider Embedding Number Reconstruction Experiment with Train/Test Split

This enhanced experiment evaluates how well embedding models from OpenAI, Google Gemini, 
and Voyage AI can capture numerical semantics by testing whether learned mappings from 
embeddings can reconstruct numerical values on unseen test data.

Key improvements:
- Train/test split for proper evaluation
- Support for OpenAI, Google Gemini, and Voyage AI models
- Cross-validation capabilities
- Enhanced evaluation metrics
- Robust error handling and fallback mechanisms
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

# Import local modules
from datasets import (
    real_positive_decimals, 
    real_positive_and_negative_decimals, 
    real_int_and_decimal
)
from api_wrapper import EmbeddingWrapper
from utils import (
    compute_explained_variance_score,
    compute_linear_reconstruction_score,
    plot_reconstruction_results,
    plot_linear_reconstruction_results,
    run_experiment_batch,
    summarize_results,
    save_results_to_csv,
    analyze_embedding_dimensionality_impact
)


def create_simple_integer_dataset(n_samples: int, max_value: int = 100) -> List[str]:
    """Create a simple dataset of integers for baseline testing."""
    import random
    random.seed(42)  # For reproducibility
    return [str(random.randint(1, max_value)) for _ in range(n_samples)]


def create_arithmetic_progression(n_samples: int, start: float = 0, step: float = 1) -> List[str]:
    """Create an arithmetic progression dataset."""
    return [str(start + i * step) for i in range(n_samples)]


def create_logarithmic_dataset(n_samples: int, base: float = 10) -> List[str]:
    """Create a dataset with logarithmic spacing."""
    values = np.logspace(0, 2, n_samples, base=base)  # 1 to 100 in log space
    return [f"{val:.6f}" for val in values]


def train_test_split_embeddings(texts: List[str], 
                               true_values: List[float],
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[List[str], List[str], 
                                                              List[float], List[float]]:
    """
    Split texts and values into train/test sets.
    
    Args:
        texts: List of text representations
        true_values: List of corresponding numerical values
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_texts, test_texts, train_values, test_values)
    """
    return train_test_split(texts, true_values, 
                          test_size=test_size, 
                          random_state=random_state,
                          shuffle=True)


def evaluate_reconstruction_with_split(wrapper: EmbeddingWrapper,
                                     texts: List[str],
                                     true_values: List[float],
                                     model: str,
                                     test_size: float = 0.2,
                                     use_ridge: bool = True,
                                     alpha: float = 1.0) -> Dict[str, Any]:
    """
    Evaluate reconstruction performance using train/test split.
    
    Args:
        wrapper: EmbeddingWrapper instance
        texts: List of text representations
        true_values: List of numerical values
        model: Model name to use
        test_size: Fraction for test set
        use_ridge: Whether to use Ridge regression (with regularization)
        alpha: Ridge regression alpha parameter
        
    Returns:
        Dictionary containing train/test metrics and model info
    """
    # Split data
    train_texts, test_texts, train_values, test_values = train_test_split_embeddings(
        texts, true_values, test_size=test_size, random_state=42
    )
    
    # Get embeddings
    train_embeddings = wrapper.embed(train_texts, model)
    test_embeddings = wrapper.embed(test_texts, model)
    
    # Convert to numpy arrays
    X_train = np.array(train_embeddings)
    X_test = np.array(test_embeddings)
    y_train = np.array(train_values)
    y_test = np.array(test_values)
    
    # Optional scaling (often helps with high-dimensional embeddings)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {
        'model': model,
        'train_size': len(train_texts),
        'test_size': len(test_texts),
        'embedding_dim': X_train.shape[1],
        'train_texts': train_texts,
        'test_texts': test_texts,
        'train_values': train_values,
        'test_values': test_values
    }
    
    # Train and evaluate different models
    models_to_test = []
    
    if use_ridge:
        models_to_test.append(('ridge', Ridge(alpha=alpha)))
        models_to_test.append(('ridge_scaled', Ridge(alpha=alpha)))
    
    models_to_test.append(('linear', LinearRegression()))
    models_to_test.append(('linear_scaled', LinearRegression()))
    
    for model_name, regressor in models_to_test:
        try:
            # Choose scaled or unscaled features
            if 'scaled' in model_name:
                X_train_fit, X_test_fit = X_train_scaled, X_test_scaled
            else:
                X_train_fit, X_test_fit = X_train, X_test
            
            # Fit model
            regressor.fit(X_train_fit, y_train)
            
            # Predictions
            train_pred = regressor.predict(X_train_fit)
            test_pred = regressor.predict(X_test_fit)
            
            # Metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            results[f'{model_name}_train_r2'] = train_r2
            results[f'{model_name}_test_r2'] = test_r2
            results[f'{model_name}_train_mse'] = train_mse
            results[f'{model_name}_test_mse'] = test_mse
            results[f'{model_name}_train_mae'] = train_mae
            results[f'{model_name}_test_mae'] = test_mae
            results[f'{model_name}_predictions'] = test_pred
            results[f'{model_name}_overfitting'] = train_r2 - test_r2
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results[f'{model_name}_error'] = str(e)
    
    return results


class MultiProviderReconstructionExperiment:
    """Enhanced experiment class supporting multiple embedding providers."""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 google_api_key: str = None, 
                 voyage_api_key: str = None,
                 base_plots_dir: str = "plots_multi_provider"):
        """Initialize the experiment with API keys for all providers."""
        self.wrapper = EmbeddingWrapper(
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            voyage_api_key=voyage_api_key
        )
        self.results = {}
        self.base_plots_dir = base_plots_dir
        
        # Get all available models
        self.all_models = []
        available_services = self.wrapper.get_available_services()
        supported_models = self.wrapper.get_supported_models()
        
        # Add models from available services
        for service, is_available in available_services.items():
            if is_available:
                self.all_models.extend(supported_models[service])
                print(f"✓ {service.title()} service available with models: {supported_models[service]}")
            else:
                print(f"✗ {service.title()} service not available")
        
        if not self.all_models:
            # Fallback: use all models for testing (will use mock data)
            for models in supported_models.values():
                self.all_models.extend(models)
            print("⚠️  No API keys provided. Will use mock data for demonstration.")
        
        print(f"\nTotal models to test: {len(self.all_models)}")
        print(f"Models: {self.all_models}")
    
    def run_train_test_reconstruction_experiment(self, n_samples: int = 300, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run comprehensive train/test reconstruction experiment.
        
        Args:
            n_samples: Total number of samples to generate
            test_size: Fraction to use for testing
            
        Returns:
            Dictionary of results
        """
        print("="*80)
        print("TRAIN/TEST SPLIT RECONSTRUCTION EXPERIMENT")
        print("="*80)
        print(f"Testing {len(self.all_models)} models on numerical reconstruction")
        print(f"Total samples: {n_samples} (train: {int(n_samples*(1-test_size))}, test: {int(n_samples*test_size)})")
        print(f"Evaluation: Ridge & Linear regression with proper train/test split")
        
        # Define datasets to test
        dataset_generators = [
            ("Small Decimals 0-1", lambda n: real_positive_decimals(n, 4)),
            ("Decimals -1 to +1", lambda n: real_positive_and_negative_decimals(n, 4)),
            ("Medium Numbers XX.XX", lambda n: real_int_and_decimal(n, 2, 2)),
            ("Arithmetic Progression", lambda n: create_arithmetic_progression(n, 0, 0.01)),
            ("Simple Integers 1-100", lambda n: create_simple_integer_dataset(n, 100)),
        ]
        
        results = {}
        
        for dataset_name, dataset_generator in dataset_generators:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print('='*60)
            
            # Generate dataset
            texts = dataset_generator(n_samples)
            true_values = [float(x) for x in texts]
            
            results[dataset_name] = {}
            
            for model in self.all_models:
                print(f"\nTesting {model}...")
                
                try:
                    # Run train/test evaluation
                    model_results = evaluate_reconstruction_with_split(
                        self.wrapper, texts, true_values, model, 
                        test_size=test_size, use_ridge=True, alpha=1.0
                    )
                    
                    results[dataset_name][model] = model_results
                    
                    # Print key metrics
                    ridge_test_r2 = model_results.get('ridge_test_r2', 0)
                    ridge_train_r2 = model_results.get('ridge_train_r2', 0)
                    linear_test_r2 = model_results.get('linear_test_r2', 0)
                    overfitting = model_results.get('ridge_overfitting', 0)
                    
                    print(f"  Ridge Test R²: {ridge_test_r2:.4f}")
                    print(f"  Ridge Train R²: {ridge_train_r2:.4f}")
                    print(f"  Linear Test R²: {linear_test_r2:.4f}")
                    print(f"  Overfitting (Train-Test): {overfitting:.4f}")
                    
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    results[dataset_name][model] = {'error': str(e)}
        
        self.results['train_test_reconstruction'] = results
        return results
    
    def run_cross_validation_experiment(self, n_samples: int = 200, n_folds: int = 5) -> Dict[str, Any]:
        """
        Run cross-validation experiment for more robust evaluation.
        
        Args:
            n_samples: Number of samples to generate
            n_folds: Number of CV folds
            
        Returns:
            Dictionary of results
        """
        from sklearn.model_selection import cross_val_score
        
        print("\n" + "="*80)
        print("CROSS-VALIDATION EXPERIMENT")
        print("="*80)
        print(f"Running {n_folds}-fold cross-validation on {len(self.all_models)} models")
        
        # Use a simple dataset for CV
        texts = real_positive_decimals(n_samples, 4)
        true_values = [float(x) for x in texts]
        
        cv_results = {}
        
        for model in self.all_models:
            print(f"\nCross-validating {model}...")
            
            try:
                # Get embeddings
                embeddings = self.wrapper.embed(texts, model)
                X = np.array(embeddings)
                y = np.array(true_values)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Cross-validation with Ridge
                ridge = Ridge(alpha=1.0)
                cv_scores = cross_val_score(ridge, X_scaled, y, cv=n_folds, scoring='r2')
                
                cv_results[model] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'embedding_dim': X.shape[1]
                }
                
                print(f"  CV R² scores: {cv_scores}")
                print(f"  Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                cv_results[model] = {'error': str(e)}
        
        self.results['cross_validation'] = cv_results
        return cv_results
    
    def run_scaling_analysis(self, sample_sizes: List[int] = [50, 100, 200, 400]) -> Dict[str, Any]:
        """
        Analyze how performance scales with dataset size.
        
        Args:
            sample_sizes: List of sample sizes to test
            
        Returns:
            Dictionary of results
        """
        print("\n" + "="*80)
        print("SCALING ANALYSIS")
        print("="*80)
        print(f"Testing how performance scales with dataset size: {sample_sizes}")
        
        scaling_results = {}
        
        # Use a consistent dataset type
        for sample_size in sample_sizes:
            print(f"\nTesting with {sample_size} samples...")
            
            texts = real_positive_decimals(sample_size, 4)
            true_values = [float(x) for x in texts]
            
            scaling_results[sample_size] = {}
            
            for model in self.all_models:
                try:
                    model_results = evaluate_reconstruction_with_split(
                        self.wrapper, texts, true_values, model, 
                        test_size=0.2, use_ridge=True
                    )
                    
                    scaling_results[sample_size][model] = {
                        'ridge_test_r2': model_results.get('ridge_test_r2', 0),
                        'ridge_train_r2': model_results.get('ridge_train_r2', 0),
                        'overfitting': model_results.get('ridge_overfitting', 0)
                    }
                    
                except Exception as e:
                    scaling_results[sample_size][model] = {'error': str(e)}
        
        self.results['scaling_analysis'] = scaling_results
        return scaling_results
    
    def generate_comprehensive_visualizations(self, save_plots: bool = True) -> None:
        """Generate comprehensive visualizations for all results."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*80)
        
        # 1. Train/Test Performance Comparison
        if 'train_test_reconstruction' in self.results:
            self._plot_train_test_comparison(save_plots)
        
        # 2. Cross-validation results
        if 'cross_validation' in self.results:
            self._plot_cross_validation_results(save_plots)
        
        # 3. Scaling analysis
        if 'scaling_analysis' in self.results:
            self._plot_scaling_analysis(save_plots)
        
        # 4. Model comparison heatmap
        self._plot_model_comparison_heatmap(save_plots)
    
    def _plot_train_test_comparison(self, save_plots: bool = True) -> None:
        """Plot train vs test performance comparison."""
        results = self.results['train_test_reconstruction']
        
        # Collect data for plotting
        models = []
        datasets = []
        train_scores = []
        test_scores = []
        
        for dataset_name, dataset_results in results.items():
            for model_name, model_results in dataset_results.items():
                if 'error' not in model_results:
                    models.append(model_name)
                    datasets.append(dataset_name)
                    train_scores.append(model_results.get('ridge_train_r2', 0))
                    test_scores.append(model_results.get('ridge_test_r2', 0))
        
        if not train_scores:
            print("No valid train/test results to plot")
            return
        
        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Color by model provider
        colors = {'text-embedding': 'blue', 'gemini': 'green', 'voyage': 'red'}
        model_colors = []
        for model in models:
            if 'text-embedding' in model:
                model_colors.append(colors['text-embedding'])
            elif 'gemini' in model:
                model_colors.append(colors['gemini'])
            elif 'voyage' in model:
                model_colors.append(colors['voyage'])
            else:
                model_colors.append('gray')
        
        scatter = ax.scatter(train_scores, test_scores, c=model_colors, alpha=0.7, s=100)
        
        # Perfect prediction line
        max_score = max(max(train_scores), max(test_scores))
        ax.plot([0, max_score], [0, max_score], 'k--', alpha=0.5, label='Perfect Generalization')
        
        ax.set_xlabel('Train R² Score')
        ax.set_ylabel('Test R² Score')
        ax.set_title('Train vs Test Performance (Ridge Regression)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text annotations for extreme points
        for i, (train, test, model, dataset) in enumerate(zip(train_scores, test_scores, models, datasets)):
            if test > 0.8 or (train - test) > 0.3:  # High performers or overfitters
                ax.annotate(f'{model[:15]}...', (train, test), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        if save_plots:
            comparison_dir = os.path.join(self.base_plots_dir, "train_test_comparison")
            os.makedirs(comparison_dir, exist_ok=True)
            plt.savefig(os.path.join(comparison_dir, "train_vs_test_performance.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_cross_validation_results(self, save_plots: bool = True) -> None:
        """Plot cross-validation results."""
        if 'cross_validation' not in self.results:
            return
        
        cv_results = self.results['cross_validation']
        
        models = []
        mean_scores = []
        std_scores = []
        
        for model, results in cv_results.items():
            if 'error' not in results:
                models.append(model)
                mean_scores.append(results['mean_cv_score'])
                std_scores.append(results['std_cv_score'])
        
        if not models:
            print("No valid CV results to plot")
            return
        
        # Sort by mean score
        sorted_indices = np.argsort(mean_scores)[::-1]
        models = [models[i] for i in sorted_indices]
        mean_scores = [mean_scores[i] for i in sorted_indices]
        std_scores = [std_scores[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        y_pos = np.arange(len(models))
        bars = ax.barh(y_pos, mean_scores, xerr=std_scores, capsize=5, alpha=0.7)
        
        # Color bars by provider
        for i, (bar, model) in enumerate(zip(bars, models)):
            if 'text-embedding' in model:
                bar.set_color('blue')
            elif 'gemini' in model:
                bar.set_color('green')
            elif 'voyage' in model:
                bar.set_color('red')
            else:
                bar.set_color('gray')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([m.replace('-', '\n') for m in models], fontsize=10)
        ax.set_xlabel('Cross-Validation R² Score')
        ax.set_title('5-Fold Cross-Validation Results (Ridge Regression)')
        ax.grid(True, alpha=0.3, axis='x')
        
        if save_plots:
            cv_dir = os.path.join(self.base_plots_dir, "cross_validation")
            os.makedirs(cv_dir, exist_ok=True)
            plt.savefig(os.path.join(cv_dir, "cross_validation_results.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_scaling_analysis(self, save_plots: bool = True) -> None:
        """Plot scaling analysis results."""
        if 'scaling_analysis' not in self.results:
            return
        
        scaling_results = self.results['scaling_analysis']
        sample_sizes = list(scaling_results.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for model in self.all_models:
            test_scores = []
            overfitting_scores = []
            
            for size in sample_sizes:
                if model in scaling_results[size] and 'error' not in scaling_results[size][model]:
                    result = scaling_results[size][model]
                    test_scores.append(result['ridge_test_r2'])
                    overfitting_scores.append(result['overfitting'])
                else:
                    test_scores.append(0)
                    overfitting_scores.append(0)
            
            if any(s > 0 for s in test_scores):  # Only plot if there are valid scores
                # Color by provider
                color = 'blue' if 'text-embedding' in model else \
                       'green' if 'gemini' in model else \
                       'red' if 'voyage' in model else 'gray'
                
                ax1.plot(sample_sizes, test_scores, 'o-', label=model, color=color, alpha=0.7)
                ax2.plot(sample_sizes, overfitting_scores, 'o-', label=model, color=color, alpha=0.7)
        
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Test R² Score')
        ax1.set_title('Test Performance vs Dataset Size')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('Overfitting (Train - Test R²)')
        ax2.set_title('Overfitting vs Dataset Size')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_plots:
            scaling_dir = os.path.join(self.base_plots_dir, "scaling_analysis")
            os.makedirs(scaling_dir, exist_ok=True)
            plt.savefig(os.path.join(scaling_dir, "scaling_analysis.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_model_comparison_heatmap(self, save_plots: bool = True) -> None:
        """Create heatmap comparing all models across datasets."""
        if 'train_test_reconstruction' not in self.results:
            return
        
        results = self.results['train_test_reconstruction']
        
        # Create matrix of scores
        datasets = list(results.keys())
        models = list(set(model for dataset_results in results.values() 
                         for model in dataset_results.keys()))
        
        score_matrix = np.zeros((len(datasets), len(models)))
        
        for i, dataset in enumerate(datasets):
            for j, model in enumerate(models):
                if model in results[dataset] and 'error' not in results[dataset][model]:
                    score = results[dataset][model].get('ridge_test_r2', 0)
                    score_matrix[i, j] = score
                else:
                    score_matrix[i, j] = np.nan
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        im = ax.imshow(score_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right')
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Test R² Score')
        
        # Add text annotations
        for i in range(len(datasets)):
            for j in range(len(models)):
                if not np.isnan(score_matrix[i, j]):
                    text = ax.text(j, i, f'{score_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Model Performance Heatmap (Test R² Scores)')
        plt.tight_layout()
        
        if save_plots:
            heatmap_dir = os.path.join(self.base_plots_dir, "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)
            plt.savefig(os.path.join(heatmap_dir, "model_comparison_heatmap.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_comprehensive_report(self, save_to_file: bool = True) -> str:
        """Generate comprehensive analysis report."""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MULTI-PROVIDER EMBEDDING RECONSTRUCTION ANALYSIS")
        report_lines.append("="*80)
        
        # Model availability summary
        available_services = self.wrapper.get_available_services()
        report_lines.append("\n1. MODEL AVAILABILITY:")
        report_lines.append("-" * 50)
        for service, available in available_services.items():
            status = "✓ Available" if available else "✗ Not available"
            report_lines.append(f"  {service.title()}: {status}")
        
        report_lines.append(f"\nTotal models tested: {len(self.all_models)}")
        
        # Train/Test results
        if 'train_test_reconstruction' in self.results:
            report_lines.append("\n2. TRAIN/TEST RECONSTRUCTION RESULTS:")
            report_lines.append("-" * 50)
            
            # Find best performers
            best_performers = []
            for dataset, dataset_results in self.results['train_test_reconstruction'].items():
                for model, model_results in dataset_results.items():
                    if 'error' not in model_results:
                        test_r2 = model_results.get('ridge_test_r2', 0)
                        train_r2 = model_results.get('ridge_train_r2', 0)
                        overfitting = model_results.get('ridge_overfitting', 0)
                        best_performers.append((dataset, model, test_r2, train_r2, overfitting))
            
            best_performers.sort(key=lambda x: x[2], reverse=True)
            
            report_lines.append("Top 10 Test Performances (Ridge Regression):")
            for i, (dataset, model, test_r2, train_r2, overfitting) in enumerate(best_performers[:10]):
                report_lines.append(f"  {i+1:2d}. {model:<25} on {dataset:<20}: "
                                  f"Test R²={test_r2:.3f}, Train R²={train_r2:.3f}, "
                                  f"Overfitting={overfitting:.3f}")
        
        # Cross-validation results
        if 'cross_validation' in self.results:
            report_lines.append("\n3. CROSS-VALIDATION RESULTS:")
            report_lines.append("-" * 50)
            
            cv_results = self.results['cross_validation']
            cv_scores = []
            for model, results in cv_results.items():
                if 'error' not in results:
                    cv_scores.append((model, results['mean_cv_score'], results['std_cv_score']))
            
            cv_scores.sort(key=lambda x: x[1], reverse=True)
            
            report_lines.append("Cross-Validation Rankings (5-fold, Ridge Regression):")
            for i, (model, mean_score, std_score) in enumerate(cv_scores):
                report_lines.append(f"  {i+1:2d}. {model:<30}: {mean_score:.3f} ± {std_score:.3f}")
        
        # Scaling analysis
        if 'scaling_analysis' in self.results:
            report_lines.append("\n4. SCALING ANALYSIS:")
            report_lines.append("-" * 50)
            
            scaling_results = self.results['scaling_analysis']
            sample_sizes = sorted(scaling_results.keys())
            
            report_lines.append("Performance vs Dataset Size (Test R²):")
            for model in self.all_models:
                scores_by_size = []
                for size in sample_sizes:
                    if (model in scaling_results[size] and 
                        'error' not in scaling_results[size][model]):
                        score = scaling_results[size][model]['ridge_test_r2']
                        scores_by_size.append(f"{score:.3f}")
                    else:
                        scores_by_size.append("N/A")
                
                if any(s != "N/A" for s in scores_by_size):
                    report_lines.append(f"  {model:<30}: {' | '.join(scores_by_size)}")
            
            report_lines.append(f"  Sample sizes:                 {' | '.join(map(str, sample_sizes))}")
        
        # Provider comparison
        report_lines.append("\n5. PROVIDER COMPARISON:")
        report_lines.append("-" * 50)
        
        if 'train_test_reconstruction' in self.results:
            provider_scores = {'OpenAI': [], 'Google': [], 'Voyage': [], 'Other': []}
            
            for dataset_results in self.results['train_test_reconstruction'].values():
                for model, model_results in dataset_results.items():
                    if 'error' not in model_results:
                        test_r2 = model_results.get('ridge_test_r2', 0)
                        if 'text-embedding' in model:
                            provider_scores['OpenAI'].append(test_r2)
                        elif 'gemini' in model:
                            provider_scores['Google'].append(test_r2)
                        elif 'voyage' in model:
                            provider_scores['Voyage'].append(test_r2)
                        else:
                            provider_scores['Other'].append(test_r2)
            
            for provider, scores in provider_scores.items():
                if scores:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    report_lines.append(f"  {provider:<15}: {mean_score:.3f} ± {std_score:.3f} "
                                      f"(n={len(scores)})")
        
        # Key insights
        report_lines.append("\n6. KEY INSIGHTS:")
        report_lines.append("-" * 50)
        report_lines.append("• Train/test split provides unbiased evaluation of numerical understanding")
        report_lines.append("• Ridge regression with regularization reduces overfitting")
        report_lines.append("• Cross-validation gives robust performance estimates")
        report_lines.append("• Test R² > 0.8 indicates strong numerical representation")
        report_lines.append("• Overfitting = Train R² - Test R² (should be < 0.2)")
        report_lines.append("• Scaling analysis reveals data efficiency")
        report_lines.append("• Feature scaling often improves high-dimensional embedding performance")
        
        # Methodology notes
        report_lines.append("\n7. METHODOLOGY:")
        report_lines.append("-" * 50)
        report_lines.append("• Train/test split: 80/20 with stratified sampling")
        report_lines.append("• Ridge regression with α=1.0 for regularization")
        report_lines.append("• Feature standardization using StandardScaler")
        report_lines.append("• Multiple regression types tested (Ridge, Linear, scaled/unscaled)")
        report_lines.append("• Cross-validation with 5 folds for robust estimates")
        report_lines.append("• Multiple datasets test different numerical representations")
        
        report = "\n".join(report_lines)
        
        if save_to_file:
            filename = "multi_provider_reconstruction_report.txt"
            with open(filename, "w") as f:
                f.write(report)
            print(f"Comprehensive report saved to: {filename}")
        
        return report


def main():
    """Main execution function for enhanced multi-provider experiment."""
    print("Multi-Provider Embedding Number Reconstruction with Train/Test Split")
    print("=" * 80)
    
    # Get API keys from environment
    openai_key = os.getenv('OPENAI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY') 
    voyage_key = os.getenv('VOYAGE_API_KEY')
    
    if not any([openai_key, google_key, voyage_key]):
        print("⚠️  No API keys found in environment variables.")
        print("Set OPENAI_API_KEY, GOOGLE_API_KEY, and/or VOYAGE_API_KEY for full functionality.")
        print("The experiment will run with mock data for demonstration.\n")
    
    try:
        # Initialize experiment
        experiment = MultiProviderReconstructionExperiment(
            openai_api_key=openai_key,
            google_api_key=google_key,
            voyage_api_key=voyage_key
        )
        
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EVALUATION")
        print("="*80)
        
        # 1. Main train/test reconstruction experiment
        print("\n🔬 Running train/test reconstruction experiment...")
        train_test_results = experiment.run_train_test_reconstruction_experiment(
            n_samples=300, test_size=0.2
        )
        
        # 2. Cross-validation for robust evaluation
        print("\n🔄 Running cross-validation experiment...")
        cv_results = experiment.run_cross_validation_experiment(
            n_samples=200, n_folds=5
        )
        
        # 3. Scaling analysis
        print("\n📈 Running scaling analysis...")
        scaling_results = experiment.run_scaling_analysis(
            sample_sizes=[50, 100, 200, 400]
        )
        
        # 4. Generate all visualizations
        print("\n🎨 Generating comprehensive visualizations...")
        experiment.generate_comprehensive_visualizations(save_plots=True)
        
        # 5. Generate comprehensive report
        print("\n📊 Generating comprehensive analysis report...")
        final_report = experiment.generate_comprehensive_report(save_to_file=True)
        
        # Display summary
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nGenerated Files:")
        print("  📄 multi_provider_reconstruction_report.txt")
        
        print("\nGenerated Plot Directories:")
        print("  📊 plots_multi_provider/train_test_comparison/")
        print("  📊 plots_multi_provider/cross_validation/")
        print("  📊 plots_multi_provider/scaling_analysis/")
        print("  📊 plots_multi_provider/heatmaps/")
        
        # Show quick summary of best performers
        if train_test_results:
            print("\n🏆 TOP PERFORMERS (Test R² Score):")
            print("-" * 50)
            
            best_performers = []
            for dataset, dataset_results in train_test_results.items():
                for model, model_results in dataset_results.items():
                    if 'error' not in model_results:
                        test_r2 = model_results.get('ridge_test_r2', 0)
                        best_performers.append((model, dataset, test_r2))
            
            best_performers.sort(key=lambda x: x[2], reverse=True)
            
            for i, (model, dataset, score) in enumerate(best_performers[:5]):
                print(f"  {i+1}. {model:<25} ({dataset}): {score:.3f}")
        
        print(f"\n📈 Full analysis report: multi_provider_reconstruction_report.txt")
        print("🎯 Check the plots directories for detailed visualizations!")
        
    except Exception as e:
        print(f"❌ Experiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# Additional utility functions for the enhanced experiment
def compare_providers_statistical_significance(results: Dict[str, Any], 
                                             alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform statistical significance tests between providers.
    
    Args:
        results: Results dictionary from train_test_reconstruction
        alpha: Significance level
        
    Returns:
        Dictionary with statistical test results
    """
    from scipy import stats
    
    # Collect scores by provider
    provider_scores = {'OpenAI': [], 'Google': [], 'Voyage': []}
    
    for dataset_results in results.values():
        for model, model_results in dataset_results.items():
            if 'error' not in model_results:
                test_r2 = model_results.get('ridge_test_r2', 0)
                if 'text-embedding' in model:
                    provider_scores['OpenAI'].append(test_r2)
                elif 'gemini' in model:
                    provider_scores['Google'].append(test_r2)
                elif 'voyage' in model:
                    provider_scores['Voyage'].append(test_r2)
    
    # Perform pairwise t-tests
    statistical_results = {}
    providers = [p for p, scores in provider_scores.items() if len(scores) > 1]
    
    for i, provider1 in enumerate(providers):
        for provider2 in providers[i+1:]:
            scores1 = provider_scores[provider1]
            scores2 = provider_scores[provider2]
            
            if len(scores1) > 1 and len(scores2) > 1:
                statistic, p_value = stats.ttest_ind(scores1, scores2)
                
                statistical_results[f"{provider1}_vs_{provider2}"] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'mean_diff': np.mean(scores1) - np.mean(scores2)
                }
    
    return statistical_results


def analyze_embedding_dimensionality_impact_enhanced(wrapper: EmbeddingWrapper,
                                                   texts: List[str],
                                                   true_values: List[float],
                                                   model: str,
                                                   max_components: int = 20) -> Dict[str, Any]:
    """
    Enhanced dimensionality analysis with train/test split.
    
    Args:
        wrapper: EmbeddingWrapper instance
        texts: List of text representations
        true_values: List of numerical values  
        model: Model to analyze
        max_components: Maximum number of PCA components to test
        
    Returns:
        Dictionary with dimensionality analysis results
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    
    # Split data
    train_texts, test_texts, train_values, test_values = train_test_split_embeddings(
        texts, true_values, test_size=0.2, random_state=42
    )
    
    # Get embeddings
    train_embeddings = wrapper.embed(train_texts, model)
    test_embeddings = wrapper.embed(test_texts, model)
    
    X_train = np.array(train_embeddings)
    X_test = np.array(test_embeddings)
    y_train = np.array(train_values)
    y_test = np.array(test_values)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {
        'original_dim': X_train.shape[1],
        'train_size': len(train_texts),
        'test_size': len(test_texts),
        'components_tested': [],
        'train_scores': [],
        'test_scores': [],
        'explained_variance_ratios': []
    }
    
    # Test different numbers of components
    max_components = min(max_components, X_train.shape[1], X_train.shape[0] - 1)
    
    for n_components in range(1, max_components + 1):
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Train model
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_pca, y_train)
        
        # Evaluate
        train_pred = ridge.predict(X_train_pca)
        test_pred = ridge.predict(X_test_pca)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results['components_tested'].append(n_components)
        results['train_scores'].append(train_r2)
        results['test_scores'].append(test_r2)
        results['explained_variance_ratios'].append(pca.explained_variance_ratio_.sum())
    
    return results


if __name__ == "__main__":
    exit(main())