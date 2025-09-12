#!/usr/bin/env python3
"""
K-Fold Cross-Validation Implementation for Embedding Experiments - Linear Models Only

This module extends the existing experiment framework to support k-fold cross-validation
with statistical reporting using only linear models (LinearRegression and Ridge).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataclasses import dataclass
import os

@dataclass
class KFoldConfig:
    """Configuration for k-fold cross-validation."""
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    

def evaluate_linear_reconstruction_kfold(embeddings: List[List[float]], 
                                       values: List[float],
                                       k_fold_config: KFoldConfig = None,
                                       use_ridge: bool = False,
                                       alpha: float = 0.01) -> Dict[str, Any]:
    """
    Evaluate numerical reconstruction using LinearRegression or Ridge with k-fold cross-validation.
    
    Args:
        embeddings: List of embedding vectors
        values: List of corresponding numerical values
        k_fold_config: K-fold configuration
        use_ridge: If True, use Ridge regression; if False, use LinearRegression
        alpha: Ridge regression regularization parameter (only used if use_ridge=True)
        
    Returns:
        Dictionary with k-fold results including means and standard deviations
    """
    if k_fold_config is None:
        k_fold_config = KFoldConfig()
    
    # Convert to numpy arrays
    X = np.array(embeddings)
    y = np.array(values)
    
    # Initialize k-fold splitter
    kf = KFold(
        n_splits=k_fold_config.n_splits,
        shuffle=k_fold_config.shuffle,
        random_state=k_fold_config.random_state
    )
    
    # Storage for fold results
    fold_results = {
        'train_r2': [],
        'test_r2': [],
        'train_mse': [],
        'test_mse': [],
        'train_mae': [],
        'test_mae': []
    }
    
    # Store predictions for later analysis
    all_train_predictions = []
    all_test_predictions = []
    all_train_values = []
    all_test_values = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features for Ridge regression (helps with regularization)
        # For LinearRegression, scaling doesn't change results but helps with numerical stability
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Choose model
        if use_ridge:
            model = Ridge(alpha=alpha, random_state=k_fold_config.random_state)
        else:
            model = LinearRegression()
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Store results
        fold_results['train_r2'].append(train_r2)
        fold_results['test_r2'].append(test_r2)
        fold_results['train_mse'].append(train_mse)
        fold_results['test_mse'].append(test_mse)
        fold_results['train_mae'].append(train_mae)
        fold_results['test_mae'].append(test_mae)
        
        # Store predictions for aggregation
        all_train_predictions.extend(train_pred)
        all_test_predictions.extend(test_pred)
        all_train_values.extend(y_train)
        all_test_values.extend(y_test)
    
    # Calculate summary statistics across folds
    summary_stats = {}
    for metric, values in fold_results.items():
        summary_stats[f'{metric}_mean'] = np.mean(values)
        summary_stats[f'{metric}_std'] = np.std(values)
        summary_stats[f'{metric}_all_folds'] = values
    
    return {
        'embedding_dim': X.shape[1],
        'n_samples': X.shape[0],
        'n_folds': k_fold_config.n_splits,
        'model_type': 'Ridge' if use_ridge else 'LinearRegression',
        'alpha': alpha if use_ridge else None,
        
        # Summary statistics (means and stds across folds)
        **summary_stats,
        
        # Raw fold results for detailed analysis
        'fold_results': fold_results,
        
        # Aggregated predictions (for plotting/analysis)
        'all_train_predictions': np.array(all_train_predictions),
        'all_test_predictions': np.array(all_test_predictions),
        'all_train_values': np.array(all_train_values),
        'all_test_values': np.array(all_test_values)
    }


def evaluate_pca_linear_kfold(embeddings: List[List[float]], 
                            values: List[float],
                            n_components: int = 1,
                            k_fold_config: KFoldConfig = None,
                            use_ridge: bool = False,
                            alpha: float = 0.01) -> Dict[str, Any]:
    """
    Evaluate numerical reconstruction using PCA + Linear regression with k-fold cross-validation.
    
    Args:
        embeddings: List of embedding vectors
        values: List of corresponding numerical values
        n_components: Number of PCA components to use
        k_fold_config: K-fold configuration
        use_ridge: If True, use Ridge regression; if False, use LinearRegression
        alpha: Ridge regression regularization parameter (only used if use_ridge=True)
        
    Returns:
        Dictionary with k-fold PCA results including means and standard deviations
    """
    if k_fold_config is None:
        k_fold_config = KFoldConfig()
    
    # Convert to numpy arrays
    X = np.array(embeddings)
    y = np.array(values)
    
    # Initialize k-fold splitter
    kf = KFold(
        n_splits=k_fold_config.n_splits,
        shuffle=k_fold_config.shuffle,
        random_state=k_fold_config.random_state
    )
    
    # Storage for fold results
    fold_results = {
        'train_r2': [],
        'test_r2': [],
        'train_mse': [],
        'test_mse': [],
        'train_mae': [],
        'test_mae': [],
        'explained_variance_ratios': [],
        'first_component_variance': []
    }
    
    # Store predictions and components for later analysis
    all_train_components = []
    all_test_components = []
    all_train_values = []
    all_test_values = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize features for PCA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit PCA
        pca = PCA(n_components=n_components, random_state=k_fold_config.random_state)
        train_components = pca.fit_transform(X_train_scaled)
        test_components = pca.transform(X_test_scaled)
        
        # Store explained variance ratios
        explained_var_ratios = pca.explained_variance_ratio_
        fold_results['explained_variance_ratios'].append(explained_var_ratios)
        fold_results['first_component_variance'].append(explained_var_ratios[0])
        
        # Choose model for fitting components to values
        if use_ridge:
            model = Ridge(alpha=alpha, random_state=k_fold_config.random_state)
        else:
            model = LinearRegression()
        
        # Fit linear model to map components to values
        model.fit(train_components, y_train)
        
        # Make predictions
        train_pred = model.predict(train_components)
        test_pred = model.predict(test_components)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Store results
        fold_results['train_r2'].append(train_r2)
        fold_results['test_r2'].append(test_r2)
        fold_results['train_mse'].append(train_mse)
        fold_results['test_mse'].append(test_mse)
        fold_results['train_mae'].append(train_mae)
        fold_results['test_mae'].append(test_mae)
        
        # Store components for aggregation
        if n_components == 1:
            all_train_components.extend(train_components.flatten())
            all_test_components.extend(test_components.flatten())
        else:
            all_train_components.extend(train_components.tolist())
            all_test_components.extend(test_components.tolist())
        all_train_values.extend(y_train)
        all_test_values.extend(y_test)
    
    # Calculate summary statistics across folds
    summary_stats = {}
    for metric, values in fold_results.items():
        if metric in ['explained_variance_ratios']:
            # Handle arrays differently
            if n_components == 1:
                summary_stats[f'{metric}_mean'] = np.mean([v[0] for v in values])
                summary_stats[f'{metric}_std'] = np.std([v[0] for v in values])
            else:
                # For multiple components, average across folds
                summary_stats[f'{metric}_mean'] = np.mean(values, axis=0)
                summary_stats[f'{metric}_std'] = np.std(values, axis=0)
            summary_stats[f'{metric}_all_folds'] = values
        else:
            summary_stats[f'{metric}_mean'] = np.mean(values)
            summary_stats[f'{metric}_std'] = np.std(values)
            summary_stats[f'{metric}_all_folds'] = values
    
    return {
        'embedding_dim': X.shape[1],
        'n_samples': X.shape[0],
        'n_components': n_components,
        'n_folds': k_fold_config.n_splits,
        'model_type': 'Ridge' if use_ridge else 'LinearRegression',
        'alpha': alpha if use_ridge else None,
        
        # Summary statistics (means and stds across folds)
        **summary_stats,
        
        # Raw fold results for detailed analysis
        'fold_results': fold_results,
        
        # Aggregated components and values (for plotting/analysis)
        'all_train_components': np.array(all_train_components),
        'all_test_components': np.array(all_test_components),
        'all_train_values': np.array(all_train_values),
        'all_test_values': np.array(all_test_values)
    }


class NumberSizeSweepKFold:
    """Extended NumberSizeSweep class with k-fold cross-validation support using only linear models."""
    
    def __init__(self, embedding_wrapper, config=None, k_fold_config: KFoldConfig = None, use_ridge: bool = False):
        # Import the original config class
        from sweep_experiments import SweepConfig
        
        self.wrapper = embedding_wrapper
        self.config = config or SweepConfig()
        self.k_fold_config = k_fold_config or KFoldConfig()
        self.use_ridge = use_ridge
        
        # Get available models
        self.available_models = self._get_models()
        
        model_type = "Ridge" if use_ridge else "LinearRegression"
        print(f"K-Fold Sweep initialized with {model_type}")
        print(f"  Models: {self.available_models}")
        print(f"  K-fold: {self.k_fold_config.n_splits} splits")
        print(f"  Decimal sizes: {self.config.decimal_sizes}")
        
        self.results = {}
    
    def _get_models(self) -> List[str]:
        """Get available models (copied from original)."""
        try:
            available_services = self.wrapper.get_available_services()
            supported_models = self.wrapper.get_supported_models()
            
            models = []
            for service, is_available in available_services.items():
                if is_available:
                    models.extend(supported_models[service])
            
            return models if models else ["mock-model"]
        except:
            return ["mock-model"]
    
    def run_decimal_sweep_kfold(self, positive_only: bool = True) -> Dict[str, Any]:
        """Run decimal sweep with k-fold cross-validation using linear models only."""
        # Import dataset functions
        from datasets import real_positive_decimals, real_positive_and_negative_decimals
        
        experiment_name = "positive_decimals_kfold" if positive_only else "mixed_sign_decimals_kfold"
        dataset_func = real_positive_decimals if positive_only else real_positive_and_negative_decimals
        
        print(f"\n{'='*50}")
        print(f"K-FOLD DECIMAL SWEEP: {experiment_name}")
        print(f"{'='*50}")
        
        results = {
            'experiment': experiment_name,
            'sizes': self.config.decimal_sizes,
            'k_folds': self.k_fold_config.n_splits,
            'model_type': 'Ridge' if self.use_ridge else 'LinearRegression',
            'models': {}
        }
        
        for model in self.available_models:
            print(f"\nTesting {model}")
            
            model_results = {
                'sizes': [],
                # Means across k-folds for direct linear regression
                'linear_r2_mean': [],
                'linear_r2_std': [],
                # Means across k-folds for PCA + linear regression
                'pca_r2_mean': [],
                'pca_r2_std': [],
                'pca_explained_var_mean': [],
                'pca_explained_var_std': [],
                # Store all fold results for detailed analysis
                'linear_r2_all_folds': [],
                'pca_r2_all_folds': [],
                'pca_explained_var_all_folds': []
            }
            
            for size in self.config.decimal_sizes:
                print(f"  Size {size} ({self.k_fold_config.n_splits} folds)...")
                
                try:
                    # Generate dataset with fixed seed for reproducibility
                    import random
                    import numpy as np
                    seed = self.config.random_state + size
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    texts = dataset_func(self.config.n_samples, size)
                    values = [float(x) for x in texts]
                    
                    # Get embeddings ONCE (using cache system)
                    print(f"    Generating embeddings (cached)...")
                    if hasattr(self.wrapper, 'embed_with_params'):
                        embeddings = self.wrapper.embed_with_params(
                            texts, model, experiment_name, size, 
                            random_state=self.config.random_state
                        )
                    else:
                        embeddings = self.wrapper.embed(texts, model)
                    
                    print(f"    Range: {min(values):.3f} to {max(values):.3f}")
                    print(f"    Running {self.k_fold_config.n_splits}-fold CV on pre-generated embeddings...")
                    
                    # Run k-fold experiments on the same embeddings
                    linear_results = evaluate_linear_reconstruction_kfold(
                        embeddings, values, self.k_fold_config, use_ridge=self.use_ridge
                    )
                    
                    pca_results = evaluate_pca_linear_kfold(
                        embeddings, values, n_components=1, k_fold_config=self.k_fold_config, 
                        use_ridge=self.use_ridge
                    )
                    
                    # Store results with means and standard deviations
                    model_results['sizes'].append(size)
                    
                    # Linear results
                    model_results['linear_r2_mean'].append(linear_results['test_r2_mean'])
                    model_results['linear_r2_std'].append(linear_results['test_r2_std'])
                    model_results['linear_r2_all_folds'].append(linear_results['test_r2_all_folds'])
                    
                    # PCA results
                    model_results['pca_r2_mean'].append(pca_results['test_r2_mean'])
                    model_results['pca_r2_std'].append(pca_results['test_r2_std'])
                    model_results['pca_r2_all_folds'].append(pca_results['test_r2_all_folds'])
                    
                    # PCA explained variance
                    model_results['pca_explained_var_mean'].append(pca_results['first_component_variance_mean'])
                    model_results['pca_explained_var_std'].append(pca_results['first_component_variance_std'])
                    model_results['pca_explained_var_all_folds'].append(pca_results['first_component_variance_all_folds'])
                    
                    print(f"    Linear R²: {linear_results['test_r2_mean']:.3f} ± {linear_results['test_r2_std']:.3f}")
                    print(f"    PCA R²: {pca_results['test_r2_mean']:.3f} ± {pca_results['test_r2_std']:.3f}")
                    print(f"    PCA Explained Var: {pca_results['first_component_variance_mean']:.3f} ± {pca_results['first_component_variance_std']:.3f}")
                
                except Exception as e:
                    print(f"    Error: {e}")
                    # Store NaN values for failed experiments
                    model_results['sizes'].append(size)
                    for key in ['linear_r2_mean', 'linear_r2_std', 'pca_r2_mean', 'pca_r2_std',
                               'pca_explained_var_mean', 'pca_explained_var_std']:
                        model_results[key].append(np.nan)
                    for key in ['linear_r2_all_folds', 'pca_r2_all_folds', 'pca_explained_var_all_folds']:
                        model_results[key].append([np.nan] * self.k_fold_config.n_splits)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    def run_all_sweeps_kfold(self) -> Dict[str, Any]:
        """Run all k-fold sweep experiments."""
        print("Running all k-fold sweep experiments...")
        
        all_results = {}
        all_results['positive_decimals_kfold'] = self.run_decimal_sweep_kfold(positive_only=True)
        all_results['mixed_decimals_kfold'] = self.run_decimal_sweep_kfold(positive_only=False)
        all_results['mixed_int_decimal_kfold'] = self.run_mixed_sweep_kfold()
        
        return all_results
    
    def run_mixed_sweep_kfold(self) -> Dict[str, Any]:
        """Run mixed integer/decimal sweep with k-fold cross-validation."""
        # Import dataset function
        from datasets import real_int_and_decimal
        
        experiment_name = "mixed_int_decimal_kfold"
        
        print(f"\n{'='*50}")
        print(f"K-FOLD MIXED INT/DECIMAL SWEEP")
        print(f"{'='*50}")
        
        results = {
            'experiment': experiment_name,
            'k_folds': self.k_fold_config.n_splits,
            'model_type': 'Ridge' if self.use_ridge else 'LinearRegression',
            'models': {}
        }
        
        for model in self.available_models:
            print(f"\nTesting {model}")
            
            model_results = {
                'sizes': [],
                # Means across k-folds
                'linear_r2_mean': [],
                'linear_r2_std': [],
                'pca_r2_mean': [],
                'pca_r2_std': [],
                'pca_explained_var_mean': [],
                'pca_explained_var_std': [],
                # Store all fold results for detailed analysis
                'linear_r2_all_folds': [],
                'pca_r2_all_folds': [],
                'pca_explained_var_all_folds': []
            }
            
            for size in self.config.mixed_int_sizes:
                print(f"  Int digits={size}, Dec digits={size} ({self.k_fold_config.n_splits} folds)...")
                
                try:
                    # Generate mixed dataset with fixed seed - same number of int and decimal digits
                    import random
                    import numpy as np
                    seed = self.config.random_state + size + 1000  # Different seed space
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    texts = real_int_and_decimal(
                        self.config.n_samples, 
                        size,  # integer digits
                        size   # decimal digits (same as integer)
                    )
                    values = [float(x) for x in texts]
                    
                    # Get embeddings ONCE (using cache system)
                    print(f"    Generating embeddings (cached)...")
                    if hasattr(self.wrapper, 'embed_with_params'):
                        embeddings = self.wrapper.embed_with_params(
                            texts, model, experiment_name, size,
                            decimal_size=size,  # Same as integer size
                            random_state=self.config.random_state
                        )
                    else:
                        embeddings = self.wrapper.embed(texts, model)
                    
                    print(f"    Range: {min(values):.3f} to {max(values):.3f}")
                    print(f"    Running {self.k_fold_config.n_splits}-fold CV on pre-generated embeddings...")
                    
                    # Run k-fold experiments on the same embeddings
                    linear_results = evaluate_linear_reconstruction_kfold(
                        embeddings, values, self.k_fold_config, use_ridge=self.use_ridge
                    )
                    
                    pca_results = evaluate_pca_linear_kfold(
                        embeddings, values, n_components=1, k_fold_config=self.k_fold_config,
                        use_ridge=self.use_ridge
                    )
                    
                    # Store results with means and standard deviations
                    model_results['sizes'].append(size)
                    
                    # Linear results
                    model_results['linear_r2_mean'].append(linear_results['test_r2_mean'])
                    model_results['linear_r2_std'].append(linear_results['test_r2_std'])
                    model_results['linear_r2_all_folds'].append(linear_results['test_r2_all_folds'])
                    
                    # PCA results
                    model_results['pca_r2_mean'].append(pca_results['test_r2_mean'])
                    model_results['pca_r2_std'].append(pca_results['test_r2_std'])
                    model_results['pca_r2_all_folds'].append(pca_results['test_r2_all_folds'])
                    
                    # PCA explained variance
                    model_results['pca_explained_var_mean'].append(pca_results['first_component_variance_mean'])
                    model_results['pca_explained_var_std'].append(pca_results['first_component_variance_std'])
                    model_results['pca_explained_var_all_folds'].append(pca_results['first_component_variance_all_folds'])
                    
                    print(f"    Linear R²: {linear_results['test_r2_mean']:.3f} ± {linear_results['test_r2_std']:.3f}")
                    print(f"    PCA R²: {pca_results['test_r2_mean']:.3f} ± {pca_results['test_r2_std']:.3f}")
                    print(f"    PCA Explained Var: {pca_results['first_component_variance_mean']:.3f} ± {pca_results['first_component_variance_std']:.3f}")
                
                except Exception as e:
                    print(f"    Error: {e}")
                    # Store NaN values for failed experiments
                    model_results['sizes'].append(size)
                    for key in ['linear_r2_mean', 'linear_r2_std', 'pca_r2_mean', 'pca_r2_std',
                               'pca_explained_var_mean', 'pca_explained_var_std']:
                        model_results[key].append(np.nan)
                    for key in ['linear_r2_all_folds', 'pca_r2_all_folds', 'pca_explained_var_all_folds']:
                        model_results[key].append([np.nan] * self.k_fold_config.n_splits)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    def plot_kfold_results(self, experiment_name: str = None) -> None:
        """Create plots with error bars showing k-fold standard deviations."""
        import matplotlib.pyplot as plt
        import os
        
        if not self.results:
            print("No results to plot")
            return
        
        experiments = [experiment_name] if experiment_name else list(self.results.keys())
        
        for exp_name in experiments:
            if exp_name not in self.results:
                print(f"Warning: Experiment '{exp_name}' not found in results")
                continue
            
            print(f"Creating k-fold plots for: {exp_name}")
            results = self.results[exp_name]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
            
            # Plot 1: Linear performance with error bars
            for model, model_results in results['models'].items():
                ax1.errorbar(model_results['sizes'], model_results['linear_r2_mean'],
                            yerr=model_results['linear_r2_std'], marker='o', label=model, 
                            linewidth=2, capsize=5)
            
            ax1.set_xlabel('Size (digits)')
            ax1.set_ylabel('Linear Test R² (Mean ± Std)')
            model_type = results.get('model_type', 'Linear')
            ax1.set_title(f'{model_type} Performance with K-Fold CV - {exp_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: PCA performance with error bars
            for model, model_results in results['models'].items():
                ax2.errorbar(model_results['sizes'], model_results['pca_r2_mean'],
                            yerr=model_results['pca_r2_std'], marker='s', label=model, 
                            linewidth=2, capsize=5)
            
            ax2.set_xlabel('Size (digits)')
            ax2.set_ylabel('PCA + Linear Test R² (Mean ± Std)')
            ax2.set_title(f'PCA + {model_type} Performance with K-Fold CV - {exp_name}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # Plot 3: PCA explained variance with error bars
            for model, model_results in results['models'].items():
                ax3.errorbar(model_results['sizes'], model_results['pca_explained_var_mean'],
                            yerr=model_results['pca_explained_var_std'], marker='^', label=model, 
                            linewidth=2, capsize=5)
            
            ax3.set_xlabel('Size (digits)')
            ax3.set_ylabel('PCA Explained Variance (Mean ± Std)')
            ax3.set_title(f'First Component Explained Variance with K-Fold CV - {exp_name}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # Plot 4: Comparison of methods with confidence intervals
            # Show the difference between linear and PCA performance
            for model, model_results in results['models'].items():
                linear_means = np.array(model_results['linear_r2_mean'])
                pca_means = np.array(model_results['pca_r2_mean'])
                # Combined standard error for difference (assuming independence)
                linear_stds = np.array(model_results['linear_r2_std'])
                pca_stds = np.array(model_results['pca_r2_std'])
                diff_stds = np.sqrt(linear_stds**2 + pca_stds**2)
                
                ax4.errorbar(model_results['sizes'], linear_means - pca_means,
                            yerr=diff_stds, marker='d', label=model, 
                            linewidth=2, capsize=5)
            
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Size (digits)')
            ax4.set_ylabel('Linear R² - PCA R² (Mean ± Std)')
            ax4.set_title(f'Method Performance Difference with K-Fold CV - {exp_name}')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'K-Fold Cross-Validation Results ({results["k_folds"]} folds) - {model_type}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save
            filename = f"kfold_sweep_{exp_name}.png"
            filepath = os.path.join(self.config.plot_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved K-fold plot: {filepath}")
    
    def export_kfold_results(self, filename: str = None) -> None:
        """Export k-fold results to CSV with detailed statistics."""
        if filename is None:
            filename = os.path.join(self.config.results_dir, "kfold_sweep_results.csv")
        
        if not self.results:
            print("No results to export")
            return
        
        rows = []
        
        for exp_name, results in self.results.items():
            for model, model_results in results['models'].items():
                for i, size in enumerate(model_results['sizes']):
                    if i < len(model_results['linear_r2_mean']):
                        row = {
                            'experiment': exp_name,
                            'model': model,
                            'size': size,
                            'k_folds': results['k_folds'],
                            'model_type': results.get('model_type', 'LinearRegression'),
                            
                            # Linear results
                            'linear_r2_mean': model_results['linear_r2_mean'][i],
                            'linear_r2_std': model_results['linear_r2_std'][i],
                            
                            # PCA results
                            'pca_r2_mean': model_results['pca_r2_mean'][i],
                            'pca_r2_std': model_results['pca_r2_std'][i],
                            
                            # PCA explained variance
                            'pca_explained_var_mean': model_results['pca_explained_var_mean'][i],
                            'pca_explained_var_std': model_results['pca_explained_var_std'][i],
                        }
                        
                        # Add individual fold results as separate columns
                        if i < len(model_results['linear_r2_all_folds']):
                            fold_results = model_results['linear_r2_all_folds'][i]
                            for fold_idx, fold_score in enumerate(fold_results):
                                row[f'linear_r2_fold_{fold_idx+1}'] = fold_score
                        
                        if i < len(model_results['pca_r2_all_folds']):
                            fold_results = model_results['pca_r2_all_folds'][i]
                            for fold_idx, fold_score in enumerate(fold_results):
                                row[f'pca_r2_fold_{fold_idx+1}'] = fold_score
                        
                        if i < len(model_results['pca_explained_var_all_folds']):
                            fold_results = model_results['pca_explained_var_all_folds'][i]
                            for fold_idx, fold_score in enumerate(fold_results):
                                row[f'pca_explained_var_fold_{fold_idx+1}'] = fold_score
                        
                        rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        print(f"Exported {len(df)} k-fold results to {filename}")
        print(f"   Includes means, standard deviations, and individual fold results")
        print(f"   K-folds: {df['k_folds'].iloc[0] if len(df) > 0 else 'N/A'}")
        print(f"   Model type: {df['model_type'].iloc[0] if len(df) > 0 else 'N/A'}")


def run_kfold_demo(embedding_wrapper, models_to_test: List[str] = None, use_ridge: bool = False):
    """Quick demo of k-fold functionality with linear models only."""
    print("K-Fold Cross-Validation Demo - Linear Models Only")
    print("=" * 50)
    
    # Import required classes
    from sweep_experiments import SweepConfig
    
    # Small config for demo
    config = SweepConfig(
        n_samples=100,  # Smaller for demo
        decimal_sizes=[3, 5, 8],  # Just a few sizes
        plot_dir="kfold_demo_plots",
        results_dir="kfold_demo_results"
    )
    
    # K-fold config
    k_fold_config = KFoldConfig(n_splits=3, random_state=42)  # 3 folds for demo
    
    # Initialize k-fold sweep
    kf_sweep = NumberSizeSweepKFold(embedding_wrapper, config, k_fold_config, use_ridge=use_ridge)
    
    # Filter models if specified
    if models_to_test:
        available = set(kf_sweep.available_models)
        requested = set(models_to_test)
        kf_sweep.available_models = list(available & requested)
        
        if not kf_sweep.available_models:
            print("No valid models")
            return
    
    model_type = "Ridge" if use_ridge else "LinearRegression"
    print(f"Testing: {kf_sweep.available_models} with {model_type}")
    
    try:
        # Run k-fold decimal sweep
        kf_sweep.run_decimal_sweep_kfold(positive_only=True)
        
        # Also run mixed sweep for demo
        if not models_to_test or len(kf_sweep.available_models) > 0:
            kf_sweep.run_mixed_sweep_kfold()
        
        # Create plots and exports
        for exp_name in kf_sweep.results.keys():
            kf_sweep.plot_kfold_results(exp_name)
        
        kf_sweep.export_kfold_results()
        
        print(f"\nK-fold demo completed successfully with {model_type}!")
        print("Check the generated plots and CSV for detailed k-fold statistics")
        
    except Exception as e:
        print(f"K-fold demo failed: {e}")
        import traceback
        traceback.print_exc()