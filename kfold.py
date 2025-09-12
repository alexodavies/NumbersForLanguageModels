#!/usr/bin/env python3
"""
K-Fold Cross-Validation Implementation with Statsmodels and Statistical Significance Analysis

This module extends the existing experiment framework to support k-fold cross-validation
with detailed statistical reporting using statsmodels for comprehensive regression analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from dataclasses import dataclass
import warnings
import os

@dataclass
class KFoldConfig:
    """Configuration for k-fold cross-validation."""
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    

@dataclass 
class SignificanceResults:
    """Container for statistical significance results."""
    n_significant_features: int
    total_features: int
    proportion_significant: float
    p_values: np.ndarray
    coefficients: np.ndarray
    std_errors: np.ndarray
    t_statistics: np.ndarray
    significant_mask: np.ndarray
    alpha: float = 0.05
    
    @property
    def significance_rate(self) -> float:
        """Alias for proportion_significant."""
        return self.proportion_significant


def analyze_model_significance(model_results, alpha: float = 0.05) -> SignificanceResults:
    """
    Analyze statistical significance of model coefficients.
    
    Args:
        model_results: Fitted statsmodels results object
        alpha: Significance level (default 0.05)
        
    Returns:
        SignificanceResults object with detailed significance analysis
    """
    # Get p-values, coefficients, etc. - these might be pandas Series or numpy arrays
    p_values = model_results.pvalues
    coefficients = model_results.params
    std_errors = model_results.bse
    t_statistics = model_results.tvalues
    
    # Convert to numpy arrays if they're pandas Series
    if hasattr(p_values, 'values'):
        p_values = p_values.values
    if hasattr(coefficients, 'values'):
        coefficients = coefficients.values
    if hasattr(std_errors, 'values'):
        std_errors = std_errors.values
    if hasattr(t_statistics, 'values'):
        t_statistics = t_statistics.values
    
    # Exclude intercept if present (last coefficient in statsmodels with add_constant)
    if hasattr(model_results, 'model') and hasattr(model_results.model, 'k_constant'):
        if model_results.model.k_constant:
            # Exclude the constant term from significance analysis
            p_values = p_values[:-1]
            coefficients = coefficients[:-1] 
            std_errors = std_errors[:-1]
            t_statistics = t_statistics[:-1]
    
    significant_mask = p_values < alpha
    n_significant = np.sum(significant_mask)
    total_features = len(p_values)
    proportion_significant = n_significant / total_features if total_features > 0 else 0.0
    
    return SignificanceResults(
        n_significant_features=n_significant,
        total_features=total_features,
        proportion_significant=proportion_significant,
        p_values=p_values,
        coefficients=coefficients,
        std_errors=std_errors,
        t_statistics=t_statistics,
        significant_mask=significant_mask,
        alpha=alpha
    )


def evaluate_statsmodels_reconstruction_kfold(embeddings: List[List[float]], 
                                            values: List[float],
                                            k_fold_config: KFoldConfig = None,
                                            alpha: float = 0.05,
                                            fit_intercept: bool = True) -> Dict[str, Any]:
    """
    Evaluate numerical reconstruction using statsmodels OLS with k-fold cross-validation.
    
    Args:
        embeddings: List of embedding vectors
        values: List of corresponding numerical values
        k_fold_config: K-fold configuration
        alpha: Significance level for statistical tests
        fit_intercept: Whether to fit an intercept term
        
    Returns:
        Dictionary with k-fold results including statistical significance analysis
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
        'train_adj_r2': [],
        'test_adj_r2': [],
        'aic': [],
        'bic': [],
        'f_statistic': [],
        'f_pvalue': [],
        'durbin_watson': [],
        'jarque_bera': [],
        'jarque_bera_pvalue': [],
        # Significance analysis
        'n_significant_features': [],
        'proportion_significant': [],
        'mean_p_value': [],
        'min_p_value': [],
        'max_coefficient_magnitude': []
    }
    
    # Store detailed results for analysis
    all_significance_results = []
    all_train_predictions = []
    all_test_predictions = []
    all_train_values = []
    all_test_values = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize features for numerical stability
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Add constant for intercept if requested
        if fit_intercept:
            X_train_sm = sm.add_constant(X_train_scaled)
            X_test_sm = sm.add_constant(X_test_scaled)
        else:
            X_train_sm = X_train_scaled
            X_test_sm = X_test_scaled
        
        # Fit statsmodels OLS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = sm.OLS(y_train, X_train_sm)
                results = model.fit()
                
                # Make predictions
                train_pred = results.predict(X_train_sm)
                test_pred = results.predict(X_test_sm)
                
                # Calculate basic metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                # Adjusted R-squared (approximate for test set)
                n_train = len(y_train)
                n_test = len(y_test)
                p = X_train_sm.shape[1] - (1 if fit_intercept else 0)  # Number of predictors excluding intercept
                
                train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p - 1) if n_train > p + 1 else train_r2
                test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p - 1) if n_test > p + 1 else test_r2
                
                # Statistical measures from training set
                aic = results.aic
                bic = results.bic
                f_statistic = results.fvalue
                f_pvalue = results.f_pvalue
                
                # Diagnostic statistics
                durbin_watson = sm.stats.stattools.durbin_watson(results.resid)
                jb_result = sm.stats.stattools.jarque_bera(results.resid)
                jb_stat, jb_pvalue = jb_result[0], jb_result[1]
                
                # Significance analysis
                sig_results = analyze_model_significance(results, alpha)
                all_significance_results.append(sig_results)
                
                # Store results
                fold_results['train_r2'].append(train_r2)
                fold_results['test_r2'].append(test_r2)
                fold_results['train_mse'].append(train_mse)
                fold_results['test_mse'].append(test_mse)
                fold_results['train_mae'].append(train_mae)
                fold_results['test_mae'].append(test_mae)
                fold_results['train_adj_r2'].append(train_adj_r2)
                fold_results['test_adj_r2'].append(test_adj_r2)
                fold_results['aic'].append(aic)
                fold_results['bic'].append(bic)
                fold_results['f_statistic'].append(f_statistic)
                fold_results['f_pvalue'].append(f_pvalue)
                fold_results['durbin_watson'].append(durbin_watson)
                fold_results['jarque_bera'].append(jb_stat)
                fold_results['jarque_bera_pvalue'].append(jb_pvalue)
                
                # Significance metrics
                fold_results['n_significant_features'].append(sig_results.n_significant_features)
                fold_results['proportion_significant'].append(sig_results.proportion_significant)
                fold_results['mean_p_value'].append(np.mean(sig_results.p_values))
                fold_results['min_p_value'].append(np.min(sig_results.p_values))
                fold_results['max_coefficient_magnitude'].append(np.max(np.abs(sig_results.coefficients)))
                
                # Store predictions for aggregation
                all_train_predictions.extend(train_pred)
                all_test_predictions.extend(test_pred)
                all_train_values.extend(y_train)
                all_test_values.extend(y_test)
                
            except Exception as e:
                print(f"    Warning: Fold {fold_idx} failed: {e}")
                # Store NaN values for failed fold
                for key in fold_results.keys():
                    fold_results[key].append(np.nan)
                
                all_significance_results.append(SignificanceResults(
                    n_significant_features=0,
                    total_features=X.shape[1],
                    proportion_significant=0.0,
                    p_values=np.full(X.shape[1], np.nan),
                    coefficients=np.full(X.shape[1], np.nan),
                    std_errors=np.full(X.shape[1], np.nan),
                    t_statistics=np.full(X.shape[1], np.nan),
                    significant_mask=np.full(X.shape[1], False),
                    alpha=alpha
                ))
    
    # Calculate summary statistics across folds
    summary_stats = {}
    for metric, values in fold_results.items():
        # Filter out NaN values for statistics
        clean_values = [v for v in values if not np.isnan(v)]
        if clean_values:
            summary_stats[f'{metric}_mean'] = np.mean(clean_values)
            summary_stats[f'{metric}_std'] = np.std(clean_values)
            summary_stats[f'{metric}_all_folds'] = values
        else:
            summary_stats[f'{metric}_mean'] = np.nan
            summary_stats[f'{metric}_std'] = np.nan
            summary_stats[f'{metric}_all_folds'] = values
    
    return {
        'embedding_dim': X.shape[1],
        'n_samples': X.shape[0],
        'n_folds': k_fold_config.n_splits,
        'model_type': 'Statsmodels_OLS',
        'fit_intercept': fit_intercept,
        'alpha': alpha,
        
        # Summary statistics (means and stds across folds)
        **summary_stats,
        
        # Raw fold results for detailed analysis
        'fold_results': fold_results,
        'significance_results': all_significance_results,
        
        # Aggregated predictions (for plotting/analysis)
        'all_train_predictions': np.array(all_train_predictions),
        'all_test_predictions': np.array(all_test_predictions),
        'all_train_values': np.array(all_train_values),
        'all_test_values': np.array(all_test_values)
    }


def evaluate_pca_statsmodels_kfold(embeddings: List[List[float]], 
                                 values: List[float],
                                 n_components: int = 1,
                                 k_fold_config: KFoldConfig = None,
                                 alpha: float = 0.05,
                                 fit_intercept: bool = True) -> Dict[str, Any]:
    """
    Evaluate numerical reconstruction using PCA + Statsmodels OLS with k-fold cross-validation.
    
    Args:
        embeddings: List of embedding vectors
        values: List of corresponding numerical values
        n_components: Number of PCA components to use
        k_fold_config: K-fold configuration
        alpha: Significance level for statistical tests
        fit_intercept: Whether to fit an intercept term
        
    Returns:
        Dictionary with k-fold PCA results including statistical significance analysis
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
    
    # Storage for fold results (similar to above but with PCA-specific metrics)
    fold_results = {
        'train_r2': [],
        'test_r2': [],
        'train_mse': [],
        'test_mse': [],
        'train_mae': [],
        'test_mae': [],
        'train_adj_r2': [],
        'test_adj_r2': [],
        'aic': [],
        'bic': [],
        'f_statistic': [],
        'f_pvalue': [],
        'durbin_watson': [],
        'jarque_bera': [],
        'jarque_bera_pvalue': [],
        'explained_variance_ratios': [],
        'first_component_variance': [],
        'cumulative_variance_explained': [],
        # Significance analysis for PCA components
        'n_significant_components': [],
        'proportion_significant_components': [],
        'component_p_values': [],
        'component_coefficients': []
    }
    
    # Store detailed results
    all_significance_results = []
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
        
        # Store PCA metrics
        explained_var_ratios = pca.explained_variance_ratio_
        fold_results['explained_variance_ratios'].append(explained_var_ratios)
        fold_results['first_component_variance'].append(explained_var_ratios[0])
        fold_results['cumulative_variance_explained'].append(np.sum(explained_var_ratios))
        
        # Prepare data for statsmodels
        if fit_intercept:
            train_components_sm = sm.add_constant(train_components)
            test_components_sm = sm.add_constant(test_components)
        else:
            train_components_sm = train_components
            test_components_sm = test_components
        
        # Fit statsmodels OLS on PCA components
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = sm.OLS(y_train, train_components_sm)
                results = model.fit()
                
                # Make predictions
                train_pred = results.predict(train_components_sm)
                test_pred = results.predict(test_components_sm)
                
                # Calculate metrics (same as above)
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                # Adjusted R-squared
                n_train = len(y_train)
                n_test = len(y_test)
                p = n_components
                
                train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p - 1) if n_train > p + 1 else train_r2
                test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p - 1) if n_test > p + 1 else test_r2
                
                # Statistical measures
                aic = results.aic
                bic = results.bic
                f_statistic = results.fvalue
                f_pvalue = results.f_pvalue
                durbin_watson = sm.stats.stattools.durbin_watson(results.resid)
                jb_stat, jb_pvalue, _, _ = sm.stats.stattools.jarque_bera(results.resid)
                
                # Significance analysis for PCA components
                sig_results = analyze_model_significance(results, alpha)
                all_significance_results.append(sig_results)
                
                # Store all results
                fold_results['train_r2'].append(train_r2)
                fold_results['test_r2'].append(test_r2)
                fold_results['train_mse'].append(train_mse)
                fold_results['test_mse'].append(test_mse)
                fold_results['train_mae'].append(train_mae)
                fold_results['test_mae'].append(test_mae)
                fold_results['train_adj_r2'].append(train_adj_r2)
                fold_results['test_adj_r2'].append(test_adj_r2)
                fold_results['aic'].append(aic)
                fold_results['bic'].append(bic)
                fold_results['f_statistic'].append(f_statistic)
                fold_results['f_pvalue'].append(f_pvalue)
                fold_results['durbin_watson'].append(durbin_watson)
                fold_results['jarque_bera'].append(jb_stat)
                fold_results['jarque_bera_pvalue'].append(jb_pvalue)
                
                # PCA component significance
                fold_results['n_significant_components'].append(sig_results.n_significant_features)
                fold_results['proportion_significant_components'].append(sig_results.proportion_significant)
                fold_results['component_p_values'].append(sig_results.p_values)
                fold_results['component_coefficients'].append(sig_results.coefficients)
                
                # Store components for aggregation
                if n_components == 1:
                    all_train_components.extend(train_components.flatten())
                    all_test_components.extend(test_components.flatten())
                else:
                    all_train_components.extend(train_components.tolist())
                    all_test_components.extend(test_components.tolist())
                all_train_values.extend(y_train)
                all_test_values.extend(y_test)
                
            except Exception as e:
                print(f"    Warning: PCA Fold {fold_idx} failed: {e}")
                # Store NaN values for failed fold
                for key in fold_results.keys():
                    if key in ['explained_variance_ratios', 'component_p_values', 'component_coefficients']:
                        fold_results[key].append(np.full(n_components, np.nan))
                    else:
                        fold_results[key].append(np.nan)
                
                all_significance_results.append(SignificanceResults(
                    n_significant_features=0,
                    total_features=n_components,
                    proportion_significant=0.0,
                    p_values=np.full(n_components, np.nan),
                    coefficients=np.full(n_components, np.nan),
                    std_errors=np.full(n_components, np.nan),
                    t_statistics=np.full(n_components, np.nan),
                    significant_mask=np.full(n_components, False),
                    alpha=alpha
                ))
    
    # Calculate summary statistics across folds
    summary_stats = {}
    for metric, values in fold_results.items():
        if metric in ['explained_variance_ratios', 'component_p_values', 'component_coefficients']:
            # Handle arrays differently
            clean_values = [v for v in values if not np.any(np.isnan(v))]
            if clean_values:
                if n_components == 1:
                    summary_stats[f'{metric}_mean'] = np.mean([v[0] if len(v) > 0 else np.nan for v in clean_values])
                    summary_stats[f'{metric}_std'] = np.std([v[0] if len(v) > 0 else np.nan for v in clean_values])
                else:
                    summary_stats[f'{metric}_mean'] = np.mean(clean_values, axis=0)
                    summary_stats[f'{metric}_std'] = np.std(clean_values, axis=0)
                summary_stats[f'{metric}_all_folds'] = values
            else:
                summary_stats[f'{metric}_mean'] = np.full(n_components, np.nan)
                summary_stats[f'{metric}_std'] = np.full(n_components, np.nan)
                summary_stats[f'{metric}_all_folds'] = values
        else:
            clean_values = [v for v in values if not np.isnan(v)]
            if clean_values:
                summary_stats[f'{metric}_mean'] = np.mean(clean_values)
                summary_stats[f'{metric}_std'] = np.std(clean_values)
                summary_stats[f'{metric}_all_folds'] = values
            else:
                summary_stats[f'{metric}_mean'] = np.nan
                summary_stats[f'{metric}_std'] = np.nan
                summary_stats[f'{metric}_all_folds'] = values
    
    return {
        'embedding_dim': X.shape[1],
        'n_samples': X.shape[0],
        'n_components': n_components,
        'n_folds': k_fold_config.n_splits,
        'model_type': 'Statsmodels_PCA_OLS',
        'fit_intercept': fit_intercept,
        'alpha': alpha,
        
        # Summary statistics (means and stds across folds)
        **summary_stats,
        
        # Raw fold results for detailed analysis
        'fold_results': fold_results,
        'significance_results': all_significance_results,
        
        # Aggregated components and values (for plotting/analysis)
        'all_train_components': np.array(all_train_components),
        'all_test_components': np.array(all_test_components),
        'all_train_values': np.array(all_train_values),
        'all_test_values': np.array(all_test_values)
    }


class NumberSizeSweepStatsmodels:
    """NumberSizeSweep class with statsmodels and statistical significance analysis."""
    
    def __init__(self, embedding_wrapper, config=None, k_fold_config: KFoldConfig = None, alpha: float = 0.05):
        # Import the original config class
        from sweep_experiments import SweepConfig
        
        self.wrapper = embedding_wrapper
        self.config = config or SweepConfig()
        self.k_fold_config = k_fold_config or KFoldConfig()
        self.alpha = alpha
        
        # Get available models
        self.available_models = self._get_models()
        
        print(f"Statsmodels K-Fold Sweep initialized")
        print(f"  Models: {self.available_models}")
        print(f"  K-fold: {self.k_fold_config.n_splits} splits")
        print(f"  Significance level: {self.alpha}")
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
            
            return models if models else ["mock-model"]
        except:
            return ["mock-model"]
    
    def run_decimal_sweep_kfold(self, positive_only: bool = True) -> Dict[str, Any]:
        """Run decimal sweep with statsmodels and significance analysis."""
        from datasets import real_positive_decimals, real_positive_and_negative_decimals
        
        experiment_name = "positive_decimals_statsmodels" if positive_only else "mixed_sign_decimals_statsmodels"
        dataset_func = real_positive_decimals if positive_only else real_positive_and_negative_decimals
        
        print(f"\n{'='*50}")
        print(f"STATSMODELS DECIMAL SWEEP: {experiment_name}")
        print(f"{'='*50}")
        
        results = {
            'experiment': experiment_name,
            'sizes': self.config.decimal_sizes,
            'k_folds': self.k_fold_config.n_splits,
            'model_type': 'Statsmodels_OLS',
            'alpha': self.alpha,
            'models': {}
        }
        
        for model in self.available_models:
            print(f"\nTesting {model}")
            
            model_results = {
                'sizes': [],
                # Performance metrics
                'linear_r2_mean': [],
                'linear_r2_std': [],
                'linear_adj_r2_mean': [],
                'linear_adj_r2_std': [],
                'pca_r2_mean': [],
                'pca_r2_std': [],
                'pca_adj_r2_mean': [],
                'pca_adj_r2_std': [],
                'pca_explained_var_mean': [],
                'pca_explained_var_std': [],
                # Statistical significance metrics
                'linear_proportion_significant_mean': [],
                'linear_proportion_significant_std': [],
                'pca_proportion_significant_mean': [],
                'pca_proportion_significant_std': [],
                'linear_f_pvalue_mean': [],
                'linear_f_pvalue_std': [],
                'pca_f_pvalue_mean': [],
                'pca_f_pvalue_std': [],
                # Model diagnostics
                'linear_aic_mean': [],
                'linear_bic_mean': [],
                'pca_aic_mean': [],
                'pca_bic_mean': [],
                # Store all fold results for detailed analysis
                'linear_all_fold_results': [],
                'pca_all_fold_results': []
            }
            
            for size in self.config.decimal_sizes:
                print(f"  Size {size} ({self.k_fold_config.n_splits} folds)...")
                
                try:
                    # Generate dataset with fixed seed
                    import random
                    import numpy as np
                    seed = self.config.random_state + size
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    texts = dataset_func(self.config.n_samples, size)
                    values = [float(x) for x in texts]
                    
                    # Get embeddings
                    print(f"    Generating embeddings (cached)...")
                    if hasattr(self.wrapper, 'embed_with_params'):
                        embeddings = self.wrapper.embed_with_params(
                            texts, model, experiment_name, size,
                            random_state=self.config.random_state
                        )
                    else:
                        embeddings = self.wrapper.embed(texts, model)
                    
                    print(f"    Range: {min(values):.3f} to {max(values):.3f}")
                    print(f"    Running {self.k_fold_config.n_splits}-fold CV with statsmodels...")
                    
                    # Run statsmodels k-fold experiments
                    linear_results = evaluate_statsmodels_reconstruction_kfold(
                        embeddings, values, self.k_fold_config, alpha=self.alpha
                    )
                    
                    pca_results = evaluate_pca_statsmodels_kfold(
                        embeddings, values, n_components=1, k_fold_config=self.k_fold_config,
                        alpha=self.alpha
                    )
                    
                    # Store results with means and standard deviations
                    model_results['sizes'].append(size)
                    
                    # Performance metrics
                    model_results['linear_r2_mean'].append(linear_results['test_r2_mean'])
                    model_results['linear_r2_std'].append(linear_results['test_r2_std'])
                    model_results['linear_adj_r2_mean'].append(linear_results['test_adj_r2_mean'])
                    model_results['linear_adj_r2_std'].append(linear_results['test_adj_r2_std'])
                    
                    model_results['pca_r2_mean'].append(pca_results['test_r2_mean'])
                    model_results['pca_r2_std'].append(pca_results['test_r2_std'])
                    model_results['pca_adj_r2_mean'].append(pca_results['test_adj_r2_mean'])
                    model_results['pca_adj_r2_std'].append(pca_results['test_adj_r2_std'])
                    model_results['pca_explained_var_mean'].append(pca_results['first_component_variance_mean'])
                    model_results['pca_explained_var_std'].append(pca_results['first_component_variance_std'])
                    
                    # Statistical significance metrics
                    model_results['linear_proportion_significant_mean'].append(linear_results['proportion_significant_mean'])
                    model_results['linear_proportion_significant_std'].append(linear_results['proportion_significant_std'])
                    model_results['pca_proportion_significant_mean'].append(pca_results['proportion_significant_components_mean'])
                    model_results['pca_proportion_significant_std'].append(pca_results['proportion_significant_components_std'])
                    
                    # F-test significance
                    model_results['linear_f_pvalue_mean'].append(linear_results['f_pvalue_mean'])
                    model_results['linear_f_pvalue_std'].append(linear_results['f_pvalue_std'])
                    model_results['pca_f_pvalue_mean'].append(pca_results['f_pvalue_mean'])
                    model_results['pca_f_pvalue_std'].append(pca_results['f_pvalue_std'])
                    
                    # Model selection criteria
                    model_results['linear_aic_mean'].append(linear_results['aic_mean'])
                    model_results['linear_bic_mean'].append(linear_results['bic_mean'])
                    model_results['pca_aic_mean'].append(pca_results['aic_mean'])
                    model_results['pca_bic_mean'].append(pca_results['bic_mean'])
                    
                    # Store detailed results
                    model_results['linear_all_fold_results'].append(linear_results)
                    model_results['pca_all_fold_results'].append(pca_results)
                    
                    print(f"    Linear R²: {linear_results['test_r2_mean']:.3f} ± {linear_results['test_r2_std']:.3f}")
                    print(f"    Linear Sig%: {linear_results['proportion_significant_mean']:.1%} ± {linear_results['proportion_significant_std']:.1%}")
                    print(f"    PCA R²: {pca_results['test_r2_mean']:.3f} ± {pca_results['test_r2_std']:.3f}")
                    print(f"    PCA Sig%: {pca_results['proportion_significant_components_mean']:.1%} ± {pca_results['proportion_significant_components_std']:.1%}")
                    print(f"    PCA Explained Var: {pca_results['first_component_variance_mean']:.3f} ± {pca_results['first_component_variance_std']:.3f}")
                
                except Exception as e:
                    print(f"    Error: {e}")
                    # Store NaN values for failed experiments
                    model_results['sizes'].append(size)
                    for key in model_results.keys():
                        if key not in ['sizes', 'linear_all_fold_results', 'pca_all_fold_results']:
                            model_results[key].append(np.nan)
                    model_results['linear_all_fold_results'].append(None)
                    model_results['pca_all_fold_results'].append(None)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    def run_all_sweeps_kfold(self) -> Dict[str, Any]:
        """Run all statsmodels sweep experiments."""
        print("Running all statsmodels sweep experiments...")
        
        all_results = {}
        all_results['positive_decimals_statsmodels'] = self.run_decimal_sweep_kfold(positive_only=True)
        all_results['mixed_decimals_statsmodels'] = self.run_decimal_sweep_kfold(positive_only=False)
        all_results['mixed_int_decimal_statsmodels'] = self.run_mixed_sweep_kfold()
        
        return all_results
    
    def run_mixed_sweep_kfold(self) -> Dict[str, Any]:
        """Run mixed integer/decimal sweep with statsmodels."""
        from datasets import real_int_and_decimal
        
        experiment_name = "mixed_int_decimal_statsmodels"
        
        print(f"\n{'='*50}")
        print(f"STATSMODELS MIXED INT/DECIMAL SWEEP")
        print(f"{'='*50}")
        
        results = {
            'experiment': experiment_name,
            'k_folds': self.k_fold_config.n_splits,
            'model_type': 'Statsmodels_OLS',
            'alpha': self.alpha,
            'models': {}
        }
        
        for model in self.available_models:
            print(f"\nTesting {model}")
            
            model_results = {
                'sizes': [],
                # Performance metrics
                'linear_r2_mean': [],
                'linear_r2_std': [],
                'linear_adj_r2_mean': [],
                'linear_adj_r2_std': [],
                'pca_r2_mean': [],
                'pca_r2_std': [],
                'pca_adj_r2_mean': [],
                'pca_adj_r2_std': [],
                'pca_explained_var_mean': [],
                'pca_explained_var_std': [],
                # Statistical significance metrics
                'linear_proportion_significant_mean': [],
                'linear_proportion_significant_std': [],
                'pca_proportion_significant_mean': [],
                'pca_proportion_significant_std': [],
                'linear_f_pvalue_mean': [],
                'linear_f_pvalue_std': [],
                'pca_f_pvalue_mean': [],
                'pca_f_pvalue_std': [],
                # Model diagnostics
                'linear_aic_mean': [],
                'linear_bic_mean': [],
                'pca_aic_mean': [],
                'pca_bic_mean': [],
                # Store all fold results for detailed analysis
                'linear_all_fold_results': [],
                'pca_all_fold_results': []
            }
            
            for size in self.config.mixed_int_sizes:
                print(f"  Int digits={size}, Dec digits={size} ({self.k_fold_config.n_splits} folds)...")
                
                try:
                    # Generate mixed dataset with fixed seed
                    import random
                    import numpy as np
                    seed = self.config.random_state + size + 1000  # Different seed space
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    texts = real_int_and_decimal(
                        self.config.n_samples,
                        size,  # integer digits
                        0   # decimal digits (same as integer)
                    )
                    values = [float(x) for x in texts]
                    
                    # Get embeddings
                    print(f"    Generating embeddings (cached)...")
                    if hasattr(self.wrapper, 'embed_with_params'):
                        embeddings = self.wrapper.embed_with_params(
                            texts, model, experiment_name, size,
                            decimal_size=0,  # Same as integer size
                            random_state=self.config.random_state
                        )
                    else:
                        embeddings = self.wrapper.embed(texts, model)
                    
                    print(f"    Range: {min(values):.3f} to {max(values):.3f}")
                    print(f"    Running {self.k_fold_config.n_splits}-fold CV with statsmodels...")
                    
                    # Run statsmodels k-fold experiments
                    linear_results = evaluate_statsmodels_reconstruction_kfold(
                        embeddings, values, self.k_fold_config, alpha=self.alpha
                    )
                    
                    pca_results = evaluate_pca_statsmodels_kfold(
                        embeddings, values, n_components=1, k_fold_config=self.k_fold_config,
                        alpha=self.alpha
                    )
                    
                    # Store results (same pattern as above)
                    model_results['sizes'].append(size)
                    
                    # Performance metrics
                    model_results['linear_r2_mean'].append(linear_results['test_r2_mean'])
                    model_results['linear_r2_std'].append(linear_results['test_r2_std'])
                    model_results['linear_adj_r2_mean'].append(linear_results['test_adj_r2_mean'])
                    model_results['linear_adj_r2_std'].append(linear_results['test_adj_r2_std'])
                    
                    model_results['pca_r2_mean'].append(pca_results['test_r2_mean'])
                    model_results['pca_r2_std'].append(pca_results['test_r2_std'])
                    model_results['pca_adj_r2_mean'].append(pca_results['test_adj_r2_mean'])
                    model_results['pca_adj_r2_std'].append(pca_results['test_adj_r2_std'])
                    model_results['pca_explained_var_mean'].append(pca_results['first_component_variance_mean'])
                    model_results['pca_explained_var_std'].append(pca_results['first_component_variance_std'])
                    
                    # Statistical significance metrics
                    model_results['linear_proportion_significant_mean'].append(linear_results['proportion_significant_mean'])
                    model_results['linear_proportion_significant_std'].append(linear_results['proportion_significant_std'])
                    model_results['pca_proportion_significant_mean'].append(pca_results['proportion_significant_components_mean'])
                    model_results['pca_proportion_significant_std'].append(pca_results['proportion_significant_components_std'])
                    
                    # F-test significance
                    model_results['linear_f_pvalue_mean'].append(linear_results['f_pvalue_mean'])
                    model_results['linear_f_pvalue_std'].append(linear_results['f_pvalue_std'])
                    model_results['pca_f_pvalue_mean'].append(pca_results['f_pvalue_mean'])
                    model_results['pca_f_pvalue_std'].append(pca_results['f_pvalue_std'])
                    
                    # Model selection criteria
                    model_results['linear_aic_mean'].append(linear_results['aic_mean'])
                    model_results['linear_bic_mean'].append(linear_results['bic_mean'])
                    model_results['pca_aic_mean'].append(pca_results['aic_mean'])
                    model_results['pca_bic_mean'].append(pca_results['bic_mean'])
                    
                    # Store detailed results
                    model_results['linear_all_fold_results'].append(linear_results)
                    model_results['pca_all_fold_results'].append(pca_results)
                    
                    print(f"    Linear R²: {linear_results['test_r2_mean']:.3f} ± {linear_results['test_r2_std']:.3f}")
                    print(f"    Linear Sig%: {linear_results['proportion_significant_mean']:.1%} ± {linear_results['proportion_significant_std']:.1%}")
                    print(f"    PCA R²: {pca_results['test_r2_mean']:.3f} ± {pca_results['test_r2_std']:.3f}")
                    print(f"    PCA Sig%: {pca_results['proportion_significant_components_mean']:.1%} ± {pca_results['proportion_significant_components_std']:.1%}")
                    print(f"    PCA Explained Var: {pca_results['first_component_variance_mean']:.3f} ± {pca_results['first_component_variance_std']:.3f}")
                
                except Exception as e:
                    print(f"    Error: {e}")
                    # Store NaN values for failed experiments
                    model_results['sizes'].append(size)
                    for key in model_results.keys():
                        if key not in ['sizes', 'linear_all_fold_results', 'pca_all_fold_results']:
                            model_results[key].append(np.nan)
                    model_results['linear_all_fold_results'].append(None)
                    model_results['pca_all_fold_results'].append(None)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    def plot_kfold_results(self, experiment_name: str = None) -> None:
        """Create comprehensive plots including significance analysis."""
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
            
            print(f"Creating statsmodels plots for: {exp_name}")
            results = self.results[exp_name]
            
            # Create a large figure with multiple subplots
            fig, axes = plt.subplots(3, 3, figsize=(20, 16))
            fig.suptitle(f'Statsmodels Analysis ({results["k_folds"]} folds, α={results["alpha"]}) - {exp_name}', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: R² Performance Comparison
            ax1 = axes[0, 0]
            for model, model_results in results['models'].items():
                ax1.errorbar(model_results['sizes'], model_results['linear_r2_mean'],
                            yerr=model_results['linear_r2_std'], marker='o', label=f'{model} Linear', 
                            linewidth=2, capsize=5)
                ax1.errorbar(model_results['sizes'], model_results['pca_r2_mean'],
                            yerr=model_results['pca_r2_std'], marker='s', label=f'{model} PCA', 
                            linewidth=2, capsize=5, linestyle='--')
            ax1.set_xlabel('Size (digits)')
            ax1.set_ylabel('Test R² (Mean ± Std)')
            ax1.set_title('R² Performance Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Adjusted R² Performance
            ax2 = axes[0, 1]
            for model, model_results in results['models'].items():
                ax2.errorbar(model_results['sizes'], model_results['linear_adj_r2_mean'],
                            yerr=model_results['linear_adj_r2_std'], marker='o', label=f'{model} Linear', 
                            linewidth=2, capsize=5)
                ax2.errorbar(model_results['sizes'], model_results['pca_adj_r2_mean'],
                            yerr=model_results['pca_adj_r2_std'], marker='s', label=f'{model} PCA', 
                            linewidth=2, capsize=5, linestyle='--')
            ax2.set_xlabel('Size (digits)')
            ax2.set_ylabel('Adjusted R² (Mean ± Std)')
            ax2.set_title('Adjusted R² Performance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Statistical Significance - Proportion of Significant Features
            ax3 = axes[0, 2]
            for model, model_results in results['models'].items():
                ax3.errorbar(model_results['sizes'], model_results['linear_proportion_significant_mean'],
                            yerr=model_results['linear_proportion_significant_std'], marker='o', 
                            label=f'{model} Linear Features', linewidth=2, capsize=5)
                ax3.errorbar(model_results['sizes'], model_results['pca_proportion_significant_mean'],
                            yerr=model_results['pca_proportion_significant_std'], marker='s', 
                            label=f'{model} PCA Components', linewidth=2, capsize=5, linestyle='--')
            ax3.set_xlabel('Size (digits)')
            ax3.set_ylabel('Proportion Significant (Mean ± Std)')
            ax3.set_title(f'Feature Significance (α={results["alpha"]})')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # Plot 4: F-test P-values (Overall Model Significance)
            ax4 = axes[1, 0]
            for model, model_results in results['models'].items():
                ax4.errorbar(model_results['sizes'], model_results['linear_f_pvalue_mean'],
                            yerr=model_results['linear_f_pvalue_std'], marker='o', label=f'{model} Linear', 
                            linewidth=2, capsize=5)
                ax4.errorbar(model_results['sizes'], model_results['pca_f_pvalue_mean'],
                            yerr=model_results['pca_f_pvalue_std'], marker='s', label=f'{model} PCA', 
                            linewidth=2, capsize=5, linestyle='--')
            ax4.axhline(y=results["alpha"], color='red', linestyle=':', alpha=0.7, label=f'α={results["alpha"]}')
            ax4.set_xlabel('Size (digits)')
            ax4.set_ylabel('F-test P-value (Mean ± Std)')
            ax4.set_title('Overall Model Significance (F-test)')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Model Selection - AIC
            ax5 = axes[1, 1]
            for model, model_results in results['models'].items():
                ax5.plot(model_results['sizes'], model_results['linear_aic_mean'], 
                        marker='o', label=f'{model} Linear AIC', linewidth=2)
                ax5.plot(model_results['sizes'], model_results['pca_aic_mean'], 
                        marker='s', label=f'{model} PCA AIC', linewidth=2, linestyle='--')
            ax5.set_xlabel('Size (digits)')
            ax5.set_ylabel('AIC (Mean)')
            ax5.set_title('Model Selection Criterion (AIC)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Model Selection - BIC
            ax6 = axes[1, 2]
            for model, model_results in results['models'].items():
                ax6.plot(model_results['sizes'], model_results['linear_bic_mean'], 
                        marker='o', label=f'{model} Linear BIC', linewidth=2)
                ax6.plot(model_results['sizes'], model_results['pca_bic_mean'], 
                        marker='s', label=f'{model} PCA BIC', linewidth=2, linestyle='--')
            ax6.set_xlabel('Size (digits)')
            ax6.set_ylabel('BIC (Mean)')
            ax6.set_title('Model Selection Criterion (BIC)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # Plot 7: PCA Explained Variance
            ax7 = axes[2, 0]
            for model, model_results in results['models'].items():
                ax7.errorbar(model_results['sizes'], model_results['pca_explained_var_mean'],
                            yerr=model_results['pca_explained_var_std'], marker='^', label=model, 
                            linewidth=2, capsize=5)
            ax7.set_xlabel('Size (digits)')
            ax7.set_ylabel('PCA Explained Variance (Mean ± Std)')
            ax7.set_title('First Component Explained Variance')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            ax7.set_ylim(0, 1)
            
            # Plot 8: Significance vs Performance Scatter
            ax8 = axes[2, 1]
            for model, model_results in results['models'].items():
                ax8.scatter(model_results['linear_proportion_significant_mean'], 
                           model_results['linear_r2_mean'], 
                           s=100, alpha=0.7, label=f'{model} Linear')
                ax8.scatter(model_results['pca_proportion_significant_mean'], 
                           model_results['pca_r2_mean'], 
                           s=100, alpha=0.7, marker='s', label=f'{model} PCA')
            ax8.set_xlabel('Proportion Features Significant')
            ax8.set_ylabel('Test R²')
            ax8.set_title('Significance vs Performance')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            
            # Plot 9: Method Performance Difference with Significance Info
            ax9 = axes[2, 2]
            for model, model_results in results['models'].items():
                linear_means = np.array(model_results['linear_r2_mean'])
                pca_means = np.array(model_results['pca_r2_mean'])
                linear_stds = np.array(model_results['linear_r2_std'])
                pca_stds = np.array(model_results['pca_r2_std'])
                diff_stds = np.sqrt(linear_stds**2 + pca_stds**2)
                
                ax9.errorbar(model_results['sizes'], linear_means - pca_means,
                            yerr=diff_stds, marker='d', label=model, 
                            linewidth=2, capsize=5)
            ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax9.set_xlabel('Size (digits)')
            ax9.set_ylabel('Linear R² - PCA R² (Mean ± Std)')
            ax9.set_title('Method Performance Difference')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save
            filename = f"statsmodels_sweep_{exp_name}.png"
            filepath = os.path.join(self.config.plot_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved Statsmodels plot: {filepath}")
    
    def export_kfold_results(self, filename: str = None) -> None:
        """Export statsmodels results to CSV with comprehensive statistics."""
        if filename is None:
            filename = os.path.join(self.config.results_dir, "statsmodels_sweep_results.csv")
        
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
                            'model_type': results.get('model_type', 'Statsmodels_OLS'),
                            'alpha': results.get('alpha', 0.05),
                            
                            # Performance metrics
                            'linear_r2_mean': model_results['linear_r2_mean'][i],
                            'linear_r2_std': model_results['linear_r2_std'][i],
                            'linear_adj_r2_mean': model_results['linear_adj_r2_mean'][i],
                            'linear_adj_r2_std': model_results['linear_adj_r2_std'][i],
                            
                            'pca_r2_mean': model_results['pca_r2_mean'][i],
                            'pca_r2_std': model_results['pca_r2_std'][i],
                            'pca_adj_r2_mean': model_results['pca_adj_r2_mean'][i],
                            'pca_adj_r2_std': model_results['pca_adj_r2_std'][i],
                            
                            'pca_explained_var_mean': model_results['pca_explained_var_mean'][i],
                            'pca_explained_var_std': model_results['pca_explained_var_std'][i],
                            
                            # Statistical significance metrics
                            'linear_proportion_significant_mean': model_results['linear_proportion_significant_mean'][i],
                            'linear_proportion_significant_std': model_results['linear_proportion_significant_std'][i],
                            'pca_proportion_significant_mean': model_results['pca_proportion_significant_mean'][i],
                            'pca_proportion_significant_std': model_results['pca_proportion_significant_std'][i],
                            
                            # Model significance (F-test)
                            'linear_f_pvalue_mean': model_results['linear_f_pvalue_mean'][i],
                            'linear_f_pvalue_std': model_results['linear_f_pvalue_std'][i],
                            'pca_f_pvalue_mean': model_results['pca_f_pvalue_mean'][i],
                            'pca_f_pvalue_std': model_results['pca_f_pvalue_std'][i],
                            
                            # Model selection criteria
                            'linear_aic_mean': model_results['linear_aic_mean'][i],
                            'linear_bic_mean': model_results['linear_bic_mean'][i],
                            'pca_aic_mean': model_results['pca_aic_mean'][i],
                            'pca_bic_mean': model_results['pca_bic_mean'][i],
                        }
                        
                        rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        print(f"Exported {len(df)} statsmodels results to {filename}")
        print(f"   Includes performance metrics, significance analysis, and model diagnostics")
        print(f"   K-folds: {df['k_folds'].iloc[0] if len(df) > 0 else 'N/A'}")
        print(f"   Significance level: {df['alpha'].iloc[0] if len(df) > 0 else 'N/A'}")
        print(f"   Model type: {df['model_type'].iloc[0] if len(df) > 0 else 'N/A'}")

# Backward compatibility alias
NumberSizeSweepKFold = NumberSizeSweepStatsmodels


def run_statsmodels_demo(embedding_wrapper, models_to_test: List[str] = None, alpha: float = 0.05):
    """Quick demo of statsmodels functionality with significance analysis."""
    print("Statsmodels K-Fold Cross-Validation Demo")
    print("=" * 50)
    
    # Import required classes
    from sweep_experiments import SweepConfig
    
    # Small config for demo
    config = SweepConfig(
        n_samples=100,  # Smaller for demo
        decimal_sizes=[3, 5, 8],  # Just a few sizes
        plot_dir="statsmodels_demo_plots",
        results_dir="statsmodels_demo_results"
    )
    
    # K-fold config
    k_fold_config = KFoldConfig(n_splits=3, random_state=42)  # 3 folds for demo
    
    # Initialize statsmodels sweep
    sm_sweep = NumberSizeSweepStatsmodels(embedding_wrapper, config, k_fold_config, alpha=alpha)
    
    # Filter models if specified
    if models_to_test:
        available = set(sm_sweep.available_models)
        requested = set(models_to_test)
        sm_sweep.available_models = list(available & requested)
        
        if not sm_sweep.available_models:
            print("No valid models")
            return
    
    print(f"Testing: {sm_sweep.available_models} with Statsmodels OLS (α={alpha})")
    
    try:
        # Run statsmodels decimal sweep
        sm_sweep.run_decimal_sweep_kfold(positive_only=True)
        
        # Also run mixed sweep for demo
        if not models_to_test or len(sm_sweep.available_models) > 0:
            sm_sweep.run_mixed_sweep_kfold()
        
        # Create plots and exports
        for exp_name in sm_sweep.results.keys():
            sm_sweep.plot_kfold_results(exp_name)
        
        sm_sweep.export_kfold_results()
        
        print(f"\nStatsmodels demo completed successfully!")
        print("Check the generated plots and CSV for detailed statistical analysis including:")
        print("  - R² and Adjusted R² with confidence intervals")
        print("  - Proportion of statistically significant features/components")
        print("  - F-test results for overall model significance")
        print("  - AIC/BIC model selection criteria")
        print("  - Comprehensive diagnostic statistics")
        
    except Exception as e:
        print(f"Statsmodels demo failed: {e}")
        import traceback
        traceback.print_exc()


# Additional utility function for detailed significance reporting
def generate_significance_report(results_dict: Dict[str, Any], experiment_name: str, model_name: str) -> str:
    """Generate a detailed text report of statistical significance findings."""
    if experiment_name not in results_dict or model_name not in results_dict[experiment_name]['models']:
        return "Results not found"
    
    model_results = results_dict[experiment_name]['models'][model_name]
    experiment_results = results_dict[experiment_name]
    
    report = f"""
STATISTICAL SIGNIFICANCE REPORT
===============================
Experiment: {experiment_name}
Model: {model_name}
K-Folds: {experiment_results['k_folds']}
Significance Level (α): {experiment_results['alpha']}

SUMMARY STATISTICS ACROSS SIZES:
-------------------------------
"""
    
    sizes = model_results['sizes']
    
    # Safety check for empty results
    if not sizes or len(model_results['linear_r2_mean']) == 0:
        return f"No results available for {experiment_name} - {model_name}"
    
    for i, size in enumerate(sizes):
        if i < len(model_results['linear_r2_mean']):
            report += f"""
Size {size} digits:
  LINEAR MODEL:
    - R² = {model_results['linear_r2_mean'][i]:.3f} ± {model_results['linear_r2_std'][i]:.3f}
    - Adjusted R² = {model_results['linear_adj_r2_mean'][i]:.3f} ± {model_results['linear_adj_r2_std'][i]:.3f}
    - Significant features: {model_results['linear_proportion_significant_mean'][i]:.1%} ± {model_results['linear_proportion_significant_std'][i]:.1%}
    - F-test p-value: {model_results['linear_f_pvalue_mean'][i]:.2e} ± {model_results['linear_f_pvalue_std'][i]:.2e}
    - AIC: {model_results['linear_aic_mean'][i]:.1f}, BIC: {model_results['linear_bic_mean'][i]:.1f}

  PCA MODEL (1 component):
    - R² = {model_results['pca_r2_mean'][i]:.3f} ± {model_results['pca_r2_std'][i]:.3f}
    - Adjusted R² = {model_results['pca_adj_r2_mean'][i]:.3f} ± {model_results['pca_adj_r2_std'][i]:.3f}
    - Explained variance: {model_results['pca_explained_var_mean'][i]:.3f} ± {model_results['pca_explained_var_std'][i]:.3f}
    - Component significant: {model_results['pca_proportion_significant_mean'][i]:.1%} ± {model_results['pca_proportion_significant_std'][i]:.1%}
    - F-test p-value: {model_results['pca_f_pvalue_mean'][i]:.2e} ± {model_results['pca_f_pvalue_std'][i]:.2e}
    - AIC: {model_results['pca_aic_mean'][i]:.1f}, BIC: {model_results['pca_bic_mean'][i]:.1f}
"""
    
    # Add interpretation
    report += f"""
INTERPRETATION:
--------------
1. Model Performance: 
   - Best linear R²: {max(model_results['linear_r2_mean']):.3f} at size {sizes[np.argmax(model_results['linear_r2_mean'])]}
   - Best PCA R²: {max(model_results['pca_r2_mean']):.3f} at size {sizes[np.argmax(model_results['pca_r2_mean'])]}

2. Statistical Significance:
   - Linear models show significant overall effect (F-test p < {experiment_results['alpha']}) in {sum(1 for p in model_results['linear_f_pvalue_mean'] if p < experiment_results['alpha'])}/{len(model_results['linear_f_pvalue_mean'])} size conditions
   - PCA models show significant overall effect in {sum(1 for p in model_results['pca_f_pvalue_mean'] if p < experiment_results['alpha'])}/{len(model_results['pca_f_pvalue_mean'])} size conditions

3. Feature Importance:
   - Average proportion of significant linear features: {np.mean(model_results['linear_proportion_significant_mean']):.1%}
   - PCA component significance rate: {np.mean(model_results['pca_proportion_significant_mean']):.1%}

4. Model Selection (Lower is better):
   - Linear models preferred by AIC in {sum(1 for i in range(len(sizes)) if model_results['linear_aic_mean'][i] < model_results['pca_aic_mean'][i])}/{len(sizes)} cases
   - Linear models preferred by BIC in {sum(1 for i in range(len(sizes)) if model_results['linear_bic_mean'][i] < model_results['pca_bic_mean'][i])}/{len(sizes)} cases
"""
    
    return report


def compare_significance_across_models(results_dict: Dict[str, Any], experiment_name: str) -> pd.DataFrame:
    """Create a comparison table of significance metrics across all models."""
    if experiment_name not in results_dict:
        return pd.DataFrame()
    
    experiment_results = results_dict[experiment_name]
    comparison_data = []
    
    for model_name, model_results in experiment_results['models'].items():
        for i, size in enumerate(model_results['sizes']):
            if i < len(model_results['linear_r2_mean']):
                comparison_data.append({
                    'Model': model_name,
                    'Size': size,
                    'Linear_R2': model_results['linear_r2_mean'][i],
                    'Linear_Sig_Prop': model_results['linear_proportion_significant_mean'][i],
                    'Linear_F_pvalue': model_results['linear_f_pvalue_mean'][i],
                    'PCA_R2': model_results['pca_r2_mean'][i],
                    'PCA_Sig_Prop': model_results['pca_proportion_significant_mean'][i],
                    'PCA_F_pvalue': model_results['pca_f_pvalue_mean'][i],
                    'PCA_Explained_Var': model_results['pca_explained_var_mean'][i],
                    'Linear_AIC': model_results['linear_aic_mean'][i],
                    'PCA_AIC': model_results['pca_aic_mean'][i],
                    'Significant_Linear': model_results['linear_f_pvalue_mean'][i] < experiment_results['alpha'],
                    'Significant_PCA': model_results['pca_f_pvalue_mean'][i] < experiment_results['alpha']
                })
    
    return pd.DataFrame(comparison_data)


# Example usage and testing functions
def validate_statsmodels_implementation():
    """Validate the statsmodels implementation with synthetic data."""
    print("Validating statsmodels implementation...")
    
    # Create synthetic data with known relationships
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    # Create features with some having true relationships to target
    X = np.random.randn(n_samples, n_features)
    
    # Create target with known coefficients (only first 5 features matter)
    true_coefficients = np.zeros(n_features)
    true_coefficients[:5] = [2.0, -1.5, 0.8, -0.3, 1.2]  # First 5 features are significant
    
    y = X @ true_coefficients + 0.1 * np.random.randn(n_samples)  # Small noise
    
    # Test our implementation
    k_fold_config = KFoldConfig(n_splits=3, random_state=42)
    
    results = evaluate_statsmodels_reconstruction_kfold(
        X.tolist(), y.tolist(), k_fold_config, alpha=0.05
    )
    
    print(f"Validation Results:")
    print(f"  R² = {results['test_r2_mean']:.3f} ± {results['test_r2_std']:.3f}")
    print(f"  Proportion significant = {results['proportion_significant_mean']:.1%} ± {results['proportion_significant_std']:.1%}")
    print(f"  F-test p-value = {results['f_pvalue_mean']:.2e}")
    print(f"  Expected ~10% significant features (5/50), got {results['proportion_significant_mean']:.1%}")
    
    # Validate that we detect the right number of significant features
    expected_significant_rate = 5 / n_features  # 5 true features out of 50
    actual_significant_rate = results['proportion_significant_mean']
    
    if abs(actual_significant_rate - expected_significant_rate) < 0.1:  # Within 10%
        print("✓ Significance detection working correctly")
    else:
        print(f"⚠ Significance detection may be off: expected ~{expected_significant_rate:.1%}, got {actual_significant_rate:.1%}")
    
    return results


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_statsmodels_implementation()