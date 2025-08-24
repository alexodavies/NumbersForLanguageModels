"""
Enhanced Utilities for Multi-Provider Embedding Reconstruction Experiments

This module provides enhanced utility functions for the embedding reconstruction
experiments, including train/test split support, advanced visualizations,
and comprehensive evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os


def compute_explained_variance_score(embeddings: List[List[float]], 
                                   true_values: List[float],
                                   n_components: int = 2) -> Dict[str, Any]:
    """
    Compute PCA explained variance score for embeddings.
    
    Args:
        embeddings: List of embedding vectors
        true_values: List of corresponding numerical values
        n_components: Number of PCA components to use
        
    Returns:
        Dictionary containing PCA results and scores
    """
    X = np.array(embeddings)
    y = np.array(true_values)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Compute R² score using first principal component
    first_component = X_pca[:, 0]
    r2_score_pca = r2_score(y, first_component)
    
    # If R² is negative, try flipping the component
    if r2_score_pca < 0:
        r2_score_pca = r2_score(y, -first_component)
        first_component = -first_component
    
    results = {
        'explained_variance_score': r2_score_pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        'first_component': first_component,
        'second_component': X_pca[:, 1] if n_components > 1 else None,
        'true_values': y,
        'pca_components': pca.components_,
        'embedding_dim': X.shape[1]
    }
    
    return results


def compute_linear_reconstruction_score(embeddings: List[List[float]], 
                                      true_values: List[float],
                                      use_ridge: bool = True,
                                      alpha: float = 1.0) -> Dict[str, Any]:
    """
    Compute linear reconstruction score using all embedding dimensions.
    
    Args:
        embeddings: List of embedding vectors
        true_values: List of corresponding numerical values
        use_ridge: Whether to use Ridge regression (with regularization)
        alpha: Ridge regression alpha parameter
        
    Returns:
        Dictionary containing linear regression results
    """
    X = np.array(embeddings)
    y = np.array(true_values)
    
    # Scale features for better numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit linear model
    if use_ridge:
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()
    
    model.fit(X_scaled, y)
    predictions = model.predict(X_scaled)
    
    # Compute metrics
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    
    results = {
        'linear_r2_score': r2,
        'linear_mse': mse,
        'linear_mae': mae,
        'predictions': predictions,
        'true_values': y,
        'model_type': 'Ridge' if use_ridge else 'Linear',
        'regularization_alpha': alpha if use_ridge else None,
        'embedding_dim': X.shape[1],
        'coefficients': model.coef_ if hasattr(model, 'coef_') else None
    }
    
    return results


def compute_train_test_reconstruction_score(embeddings_train: List[List[float]],
                                          embeddings_test: List[List[float]],
                                          values_train: List[float],
                                          values_test: List[float],
                                          use_ridge: bool = True,
                                          alpha: float = 1.0) -> Dict[str, Any]:
    """
    Compute reconstruction scores with proper train/test split.
    
    Args:
        embeddings_train: Training embeddings
        embeddings_test: Test embeddings
        values_train: Training values
        values_test: Test values
        use_ridge: Whether to use Ridge regression
        alpha: Ridge regularization parameter
        
    Returns:
        Dictionary containing train/test results
    """
    X_train = np.array(embeddings_train)
    X_test = np.array(embeddings_test)
    y_train = np.array(values_train)
    y_test = np.array(values_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model on training data
    if use_ridge:
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Compute metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'overfitting': train_r2 - test_r2,
        'train_predictions': train_pred,
        'test_predictions': test_pred,
        'train_values': y_train,
        'test_values': y_test,
        'model_type': 'Ridge' if use_ridge else 'Linear',
        'regularization_alpha': alpha if use_ridge else None,
        'embedding_dim': X_train.shape[1]
    }
    
    return results


def plot_reconstruction_results(results: Dict[str, Any], 
                              title: str = "PCA Reconstruction Results",
                              save_path: Optional[str] = None,
                              show_plot: bool = False) -> None:
    """
    Enhanced plotting function for PCA reconstruction results.
    
    Args:
        results: Results dictionary from compute_explained_variance_score
        title: Plot title
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot
    """
    if 'first_component' not in results:
        print("No PCA results found in results dictionary")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    first_comp = results['first_component']
    true_vals = results['true_values']
    
    # 1. First component vs true values
    ax1.scatter(true_vals, first_comp, alpha=0.6, color='blue')
    ax1.plot([min(true_vals), max(true_vals)], 
             [min(true_vals), max(true_vals)], 'r--', alpha=0.8)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('First Principal Component')
    ax1.set_title(f'PCA Reconstruction (R² = {results["explained_variance_score"]:.4f})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = true_vals - first_comp
    ax2.scatter(first_comp, residuals, alpha=0.6, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predictions')
    ax2.grid(True, alpha=0.3)
    
    # 3. Explained variance ratio
    if 'explained_variance_ratio' in results:
        components = range(1, len(results['explained_variance_ratio']) + 1)
        ax3.bar(components, results['explained_variance_ratio'], alpha=0.7, color='orange')
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.set_title('Explained Variance by Component')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 2D PCA visualization (if available)
    if results.get('second_component') is not None:
        second_comp = results['second_component']
        scatter = ax4.scatter(first_comp, second_comp, c=true_vals, cmap='viridis', alpha=0.7)
        ax4.set_xlabel('First Principal Component')
        ax4.set_ylabel('Second Principal Component') 
        ax4.set_title('2D PCA Colored by True Values')
        plt.colorbar(scatter, ax=ax4)
    else:
        # Histogram of first component
        ax4.hist(first_comp, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('First Principal Component')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of First Component')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show_plot:
            plt.close()
    
    if show_plot:
        plt.show()


def plot_linear_reconstruction_results(results: Dict[str, Any],
                                     title: str = "Linear Reconstruction Results", 
                                     save_path: Optional[str] = None,
                                     show_plot: bool = False) -> None:
    """
    Enhanced plotting function for linear reconstruction results.
    
    Args:
        results: Results dictionary from compute_linear_reconstruction_score
        title: Plot title
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot
    """
    if 'predictions' not in results:
        print("No linear reconstruction results found in results dictionary")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    predictions = results['predictions']
    true_vals = results['true_values']
    
    # 1. Predictions vs true values
    ax1.scatter(true_vals, predictions, alpha=0.6, color='blue')
    min_val = min(min(true_vals), min(predictions))
    max_val = max(max(true_vals), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title(f'Linear Reconstruction (R² = {results["linear_r2_score"]:.4f})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = true_vals - predictions
    ax2.scatter(predictions, residuals, alpha=0.6, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Predictions')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predictions')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    ax3.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    # 4. Model coefficients importance (if available)
    if 'coefficients' in results and results['coefficients'] is not None:
        coeffs = results['coefficients']
        # Show top 20 most important coefficients
        coeff_indices = np.argsort(np.abs(coeffs))[-20:]
        ax4.barh(range(len(coeff_indices)), coeffs[coeff_indices])
        ax4.set_xlabel('Coefficient Value')
        ax4.set_ylabel('Feature Index')
        ax4.set_title('Top 20 Feature Coefficients')
        ax4.grid(True, alpha=0.3)
    else:
        # Scatter plot of predictions vs true values (alternative view)
        ax4.scatter(true_vals, predictions, alpha=0.6, color='purple')
        ax4.plot([min(true_vals), max(true_vals)], 
                [min(true_vals), max(true_vals)], 'r--', alpha=0.8)
        ax4.set_xlabel('True Values')
        ax4.set_ylabel('Predictions')
        ax4.set_title('Predictions vs True (Alternative View)')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show_plot:
            plt.close()
    
    if show_plot:
        plt.show()


def plot_train_test_reconstruction_results(results: Dict[str, Any],
                                         title: str = "Train/Test Reconstruction Results",
                                         save_path: Optional[str] = None,
                                         show_plot: bool = False) -> None:
    """
    Plot train/test reconstruction results with comprehensive analysis.
    
    Args:
        results: Results dictionary from compute_train_test_reconstruction_score
        title: Plot title
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot
    """
    if 'train_predictions' not in results:
        print("No train/test reconstruction results found in results dictionary")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    train_pred = results['train_predictions']
    test_pred = results['test_predictions']
    train_vals = results['train_values']
    test_vals = results['test_values']
    
    # 1. Train vs Test Performance
    ax1.scatter(train_vals, train_pred, alpha=0.6, color='blue', label='Train', s=30)
    ax1.scatter(test_vals, test_pred, alpha=0.6, color='red', label='Test', s=30)
    
    all_vals = np.concatenate([train_vals, test_vals])
    all_preds = np.concatenate([train_pred, test_pred])
    min_val = min(min(all_vals), min(all_preds))
    max_val = max(max(all_vals), max(all_preds))
    
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title(f'Train R²: {results["train_r2"]:.3f}, Test R²: {results["test_r2"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals Analysis
    train_residuals = train_vals - train_pred
    test_residuals = test_vals - test_pred
    
    ax2.scatter(train_pred, train_residuals, alpha=0.6, color='blue', label='Train', s=30)
    ax2.scatter(test_pred, test_residuals, alpha=0.6, color='red', label='Test', s=30)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Predictions')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'Residuals (Overfitting: {results["overfitting"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of Residuals
    ax3.hist(train_residuals, bins=15, alpha=0.6, color='blue', label='Train', density=True)
    ax3.hist(test_residuals, bins=15, alpha=0.6, color='red', label='Test', density=True)
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Density')
    ax3.set_title('Residuals Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Comparison
    metrics = ['R²', 'MSE', 'MAE']
    train_metrics = [results['train_r2'], results['train_mse'], results['train_mae']]
    test_metrics = [results['test_r2'], results['test_mse'], results['test_mae']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize MSE and MAE for better visualization
    max_mse = max(results['train_mse'], results['test_mse'])
    max_mae = max(results['train_mae'], results['test_mae'])
    
    train_metrics_norm = [train_metrics[0], train_metrics[1]/max_mse, train_metrics[2]/max_mae]
    test_metrics_norm = [test_metrics[0], test_metrics[1]/max_mse, test_metrics[2]/max_mae]
    
    ax4.bar(x - width/2, train_metrics_norm, width, label='Train', alpha=0.8, color='blue')
    ax4.bar(x + width/2, test_metrics_norm, width, label='Test', alpha=0.8, color='red')
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Normalized Values')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['R² (raw)', 'MSE (norm)', 'MAE (norm)'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show_plot:
            plt.close()
    
    if show_plot:
        plt.show()


def run_experiment_batch_with_splits(embedding_wrapper,
                                   dataset_generators: List[Tuple[str, callable]],
                                   models: List[str],
                                   n_samples: int = 200,
                                   test_size: float = 0.2,
                                   use_ridge: bool = True,
                                   alpha: float = 1.0,
                                   **generator_kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Run experiments with train/test splits across multiple datasets and models.
    
    Args:
        embedding_wrapper: Initialized EmbeddingWrapper instance
        dataset_generators: List of (name, generator_function) tuples
        models: List of model names to test
        n_samples: Total number of samples per dataset
        test_size: Fraction to use for testing
        use_ridge: Whether to use Ridge regression
        alpha: Ridge regression alpha parameter
        **generator_kwargs: Additional kwargs for dataset generators
        
    Returns:
        Nested dictionary of results: {dataset_name: {model_name: results}}
    """
    all_results = {}
    
    for dataset_name, generator_func in dataset_generators:
        print(f"\n{'='*60}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Generate dataset
        dataset = generator_func(n_samples, **generator_kwargs)
        true_values = [float(x) for x in dataset]
        
        dataset_results = {}
        
        for model in models:
            print(f"\nTesting model: {model}")
            try:
                # Train/test split
                train_texts, test_texts, train_values, test_values = train_test_split(
                    dataset, true_values, test_size=test_size, random_state=42, shuffle=True
                )
                
                # Get embeddings
                train_embeddings = embedding_wrapper.embed(train_texts, model)
                test_embeddings = embedding_wrapper.embed(test_texts, model)
                
                # Compute train/test reconstruction scores
                split_results = compute_train_test_reconstruction_score(
                    train_embeddings, test_embeddings, train_values, test_values,
                    use_ridge=use_ridge, alpha=alpha
                )
                
                # Also compute traditional PCA results for comparison
                all_embeddings = embedding_wrapper.embed(dataset, model)
                pca_results = compute_explained_variance_score(all_embeddings, true_values, n_components=2)
                
                # Combine results
                combined_results = {**split_results, **pca_results}
                dataset_results[model] = combined_results
                
                print(f"  Train R²: {split_results['train_r2']:.4f}")
                print(f"  Test R²: {split_results['test_r2']:.4f}")
                print(f"  Overfitting: {split_results['overfitting']:.4f}")
                print(f"  PCA R²: {pca_results['explained_variance_score']:.4f}")
                
            except Exception as e:
                print(f"  Error with {model}: {str(e)}")
                dataset_results[model] = {'error': str(e)}
        
        all_results[dataset_name] = dataset_results
    
    return all_results


def create_comprehensive_comparison_plot(results: Dict[str, Dict[str, Any]],
                                        save_path: Optional[str] = None,
                                        show_plot: bool = False) -> None:
    """
    Create a comprehensive comparison plot across all models and datasets.
    
    Args:
        results: Results from run_experiment_batch_with_splits
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot
    """
    # Extract data for plotting
    datasets = list(results.keys())
    models = list(set(model for dataset_results in results.values() 
                     for model in dataset_results.keys() if 'error' not in dataset_results[model]))
    
    # Create matrices for different metrics
    train_r2_matrix = np.zeros((len(datasets), len(models)))
    test_r2_matrix = np.zeros((len(datasets), len(models)))
    overfitting_matrix = np.zeros((len(datasets), len(models)))
    pca_r2_matrix = np.zeros((len(datasets), len(models)))
    
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            if model in results[dataset] and 'error' not in results[dataset][model]:
                result = results[dataset][model]
                train_r2_matrix[i, j] = result.get('train_r2', 0)
                test_r2_matrix[i, j] = result.get('test_r2', 0)
                overfitting_matrix[i, j] = result.get('overfitting', 0)
                pca_r2_matrix[i, j] = result.get('explained_variance_score', 0)
            else:
                train_r2_matrix[i, j] = np.nan
                test_r2_matrix[i, j] = np.nan
                overfitting_matrix[i, j] = np.nan
                pca_r2_matrix[i, j] = np.nan
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Test R² Heatmap
    im1 = ax1.imshow(test_r2_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Test R² Scores', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right')
    ax1.set_yticks(range(len(datasets)))
    ax1.set_yticklabels(datasets)
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(models)):
            if not np.isnan(test_r2_matrix[i, j]):
                ax1.text(j, i, f'{test_r2_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im1, ax=ax1, label='Test R² Score')
    
    # 2. Train vs Test R² Scatter
    train_scores = train_r2_matrix[~np.isnan(train_r2_matrix)]
    test_scores = test_r2_matrix[~np.isnan(test_r2_matrix)]
    
    ax2.scatter(train_scores, test_scores, alpha=0.7, s=60)
    max_score = max(max(train_scores), max(test_scores))
    ax2.plot([0, max_score], [0, max_score], 'r--', alpha=0.8, label='Perfect Generalization')
    ax2.set_xlabel('Train R² Score')
    ax2.set_ylabel('Test R² Score')
    ax2.set_title('Train vs Test Performance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Overfitting Analysis
    im3 = ax3.imshow(overfitting_matrix, cmap='RdYlBu', aspect='auto', vmin=-0.2, vmax=0.5)
    ax3.set_title('Overfitting (Train R² - Test R²)', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right')
    ax3.set_yticks(range(len(datasets)))
    ax3.set_yticklabels(datasets)
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(models)):
            if not np.isnan(overfitting_matrix[i, j]):
                ax3.text(j, i, f'{overfitting_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im3, ax=ax3, label='Overfitting Score')
    
    # 4. PCA vs Linear Regression Comparison
    pca_scores = pca_r2_matrix[~np.isnan(pca_r2_matrix)]
    linear_scores = test_r2_matrix[~np.isnan(test_r2_matrix)]
    
    ax4.scatter(pca_scores, linear_scores, alpha=0.7, s=60)
    max_score = max(max(pca_scores), max(linear_scores))
    ax4.plot([0, max_score], [0, max_score], 'r--', alpha=0.8, label='Equal Performance')
    ax4.set_xlabel('PCA R² Score')
    ax4.set_ylabel('Linear Regression Test R² Score')
    ax4.set_title('PCA vs Linear Regression Performance', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Provider Embedding Reconstruction Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show_plot:
            plt.close()
    
    if show_plot:
        plt.show()


def save_results_to_csv_enhanced(results: Dict[str, Dict[str, Any]], 
                               filename: str = "enhanced_experiment_results.csv") -> None:
    """
    Save enhanced experiment results to CSV file.
    
    Args:
        results: Results from run_experiment_batch_with_splits
        filename: Output CSV filename
    """
    import pandas as pd
    
    rows = []
    for dataset_name, dataset_results in results.items():
        for model_name, model_results in dataset_results.items():
            if 'error' not in model_results:
                row = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'provider': 'OpenAI' if 'text-embedding' in model_name else
                               'Google' if 'gemini' in model_name else
                               'Voyage' if 'voyage' in model_name else 'Other',
                    'train_r2': model_results.get('train_r2', np.nan),
                    'test_r2': model_results.get('test_r2', np.nan),
                    'overfitting': model_results.get('overfitting', np.nan),
                    'train_mse': model_results.get('train_mse', np.nan),
                    'test_mse': model_results.get('test_mse', np.nan),
                    'train_mae': model_results.get('train_mae', np.nan),
                    'test_mae': model_results.get('test_mae', np.nan),
                    'pca_explained_variance_score': model_results.get('explained_variance_score', np.nan),
                    'pca_correlation': model_results.get('correlation', np.nan),
                    'embedding_dim': model_results.get('embedding_dim', np.nan),
                    'model_type': model_results.get('model_type', 'Unknown')
                }
                rows.append(row)
            else:
                row = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'provider': 'OpenAI' if 'text-embedding' in model_name else
                               'Google' if 'gemini' in model_name else
                               'Voyage' if 'voyage' in model_name else 'Other',
                    'error': model_results['error']
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Enhanced results saved to {filename}")


def compute_explained_variance_score(embeddings: List[List[float]], 
                                   true_values: List[float],
                                   n_components: int = 1,
                                   standardize_embeddings: bool = True) -> Dict[str, Any]:
    """
    Compute explained variance score using PCA projection.
    
    Args:
        embeddings: List of embedding vectors
        true_values: List of true numerical values
        n_components: Number of PCA components to use
        standardize_embeddings: Whether to standardize embeddings before PCA
        
    Returns:
        Dictionary with explained variance score and additional metrics
    """
    # Convert to numpy arrays
    X = np.array(embeddings)
    y = np.array(true_values)
    
    # Standardize embeddings if requested
    if standardize_embeddings:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Use first component for reconstruction
    pca_component = X_pca[:, 0]
    
    # Calculate explained variance score (R²) using sklearn's r2_score
    explained_var = r2_score(y, pca_component)
    
    # Additional metrics
    correlation = np.corrcoef(y, pca_component)[0, 1]
    mse = np.mean((y - pca_component) ** 2)
    
    # Get sklearn's explained variance ratio
    sklearn_explained_variance_ratio = pca.explained_variance_ratio_[0]
    
    return {
        'explained_variance_score': explained_var,
        'correlation': correlation,
        'mse': mse,
        'pca_explained_variance_ratio': sklearn_explained_variance_ratio,
        'pca_component': pca_component,
        'true_values': y,
        'pca_object': pca,
        'pca_components': X_pca  # Store all components for visualization
    }


def compute_linear_reconstruction_score(embeddings: List[List[float]], 
                                      true_values: List[float],
                                      standardize_embeddings: bool = True) -> Dict[str, Any]:
    """
    Compute reconstruction score using linear regression on all embedding dimensions.
    
    Args:
        embeddings: List of embedding vectors
        true_values: List of true numerical values
        standardize_embeddings: Whether to standardize embeddings before regression
        
    Returns:
        Dictionary with linear reconstruction metrics
    """
    # Convert to numpy arrays
    X = np.array(embeddings)
    y = np.array(true_values)
    
    # Standardize embeddings if requested
    if standardize_embeddings:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Make predictions
    y_pred = reg.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    correlation = np.corrcoef(y, y_pred)[0, 1]
    mse = np.mean((y - y_pred) ** 2)
    
    return {
        'linear_r2_score': r2,
        'linear_correlation': correlation,
        'linear_mse': mse,
        'linear_predictions': y_pred,
        'true_values': y,
        'linear_model': reg
    }


def plot_reconstruction_results(results: Dict[str, Any], 
                              title: str = "PCA Reconstruction Results",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot PCA reconstruction results with enhanced visualizations.
    
    Args:
        results: Results from compute_explained_variance_score
        title: Plot title
        save_path: Path to save the plot (optional)
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Create a 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. True vs PCA Component 1 (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(results['true_values'], results['pca_component'], alpha=0.6, s=30)
    ax1.plot([results['true_values'].min(), results['true_values'].max()], 
             [results['true_values'].min(), results['true_values'].max()], 
             'r--', alpha=0.8, label='Perfect reconstruction')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('PCA Component 1')
    ax1.set_title(f'True vs PCA Reconstruction\nR² = {results["explained_variance_score"]:.4f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Residuals plot (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = results['true_values'] - results['pca_component']
    ax2.scatter(results['pca_component'], residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA Components scatter (PC1 vs PC2) if available (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if results['pca_components'].shape[1] >= 2:
        scatter = ax3.scatter(results['pca_components'][:, 0], 
                             results['pca_components'][:, 1], 
                             c=results['true_values'], 
                             cmap='viridis', alpha=0.7, s=30)
        ax3.set_xlabel('PCA Component 1')
        ax3.set_ylabel('PCA Component 2')
        ax3.set_title('PCA Space (colored by true value)')
        plt.colorbar(scatter, ax=ax3, label='True Value')
    else:
        ax3.text(0.5, 0.5, 'Only 1 PCA component\navailable', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('PCA Component 2 (N/A)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution of true values (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(results['true_values'], bins=20, alpha=0.7, density=True, color='skyblue')
    ax4.set_xlabel('True Values')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of True Values')
    ax4.grid(True, alpha=0.3)
    
    # 5. Distribution of PCA component (bottom middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(results['pca_component'], bins=20, alpha=0.7, density=True, color='lightcoral')
    ax5.set_xlabel('PCA Component 1')
    ax5.set_ylabel('Density')
    ax5.set_title('Distribution of PCA Component 1')
    ax5.grid(True, alpha=0.3)
    
    # 6. Correlation matrix heatmap (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])
    if results['pca_components'].shape[1] >= 2:
        corr_data = np.column_stack([results['true_values'], 
                                   results['pca_components'][:, 0], 
                                   results['pca_components'][:, 1]])
        corr_matrix = np.corrcoef(corr_data.T)
        labels = ['True Values', 'PC1', 'PC2']
        im = ax6.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax6.set_xticks(range(len(labels)))
        ax6.set_yticks(range(len(labels)))
        ax6.set_xticklabels(labels)
        ax6.set_yticklabels(labels)
        
        # Add correlation values to heatmap
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax6.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                        ha='center', va='center', color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
        
        ax6.set_title('Correlation Matrix')
        plt.colorbar(im, ax=ax6)
    else:
        corr_val = results['correlation']
        ax6.text(0.5, 0.5, f'Correlation:\nTrue vs PC1\n{corr_val:.3f}', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)
        ax6.set_title('Correlation (True vs PC1)')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_linear_reconstruction_results(linear_results: Dict[str, Any],
                                     title: str = "Linear Reconstruction Results",
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot linear reconstruction results.
    
    Args:
        linear_results: Results from compute_linear_reconstruction_score
        title: Plot title
        save_path: Path to save the plot (optional)
        figsize: Figure size
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. True vs Predicted values
    ax1.scatter(linear_results['true_values'], linear_results['linear_predictions'], alpha=0.6, s=30)
    ax1.plot([linear_results['true_values'].min(), linear_results['true_values'].max()], 
             [linear_results['true_values'].min(), linear_results['true_values'].max()], 
             'r--', alpha=0.8, label='Perfect reconstruction')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'True vs Linear Predictions\nR² = {linear_results["linear_r2_score"]:.4f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Residuals plot
    residuals = linear_results['true_values'] - linear_results['linear_predictions']
    ax2.scatter(linear_results['linear_predictions'], residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of true values
    ax3.hist(linear_results['true_values'], bins=20, alpha=0.7, density=True, color='skyblue', label='True')
    ax3.hist(linear_results['linear_predictions'], bins=20, alpha=0.7, density=True, color='lightcoral', label='Predicted')
    ax3.set_xlabel('Values')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals distribution
    ax4.hist(residuals, bins=20, alpha=0.7, density=True, color='lightgreen')
    ax4.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Density')
    ax4.set_title('Residuals Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def run_experiment_batch(embedding_wrapper,
                        dataset_generators: List[Tuple[str, callable]],
                        models: List[str],
                        n_samples: int = 100,
                        include_linear_reconstruction: bool = True,
                        **generator_kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Run experiments across multiple datasets and models.
    
    Args:
        embedding_wrapper: Initialized EmbeddingWrapper instance
        dataset_generators: List of (name, generator_function) tuples
        models: List of model names to test
        n_samples: Number of samples per dataset
        include_linear_reconstruction: Whether to include linear reconstruction experiments
        **generator_kwargs: Additional kwargs for dataset generators
        
    Returns:
        Nested dictionary of results: {dataset_name: {model_name: results}}
    """
    all_results = {}
    
    for dataset_name, generator_func in dataset_generators:
        print(f"\n{'='*60}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Generate dataset
        dataset = generator_func(n_samples, **generator_kwargs)
        true_values = [float(x) for x in dataset]
        
        dataset_results = {}
        
        for model in models:
            print(f"\nTesting model: {model}")
            try:
                # Get embeddings
                embeddings = embedding_wrapper.embed(dataset, model)
                
                # Compute PCA-based explained variance score
                pca_results = compute_explained_variance_score(embeddings, true_values, n_components=2)
                
                # Compute linear reconstruction score if requested
                if include_linear_reconstruction:
                    linear_results = compute_linear_reconstruction_score(embeddings, true_values)
                    # Combine results
                    combined_results = {**pca_results, **linear_results}
                else:
                    combined_results = pca_results
                
                dataset_results[model] = combined_results
                
                print(f"  PCA Explained Variance Score (R²): {pca_results['explained_variance_score']:.4f}")
                print(f"  PCA Correlation: {pca_results['correlation']:.4f}")
                print(f"  PCA Explained Variance Ratio: {pca_results['pca_explained_variance_ratio']:.4f}")
                
                if include_linear_reconstruction:
                    print(f"  Linear Reconstruction R²: {linear_results['linear_r2_score']:.4f}")
                    print(f"  Linear Correlation: {linear_results['linear_correlation']:.4f}")
                
            except Exception as e:
                print(f"  Error with {model}: {str(e)}")
                dataset_results[model] = {'error': str(e)}
        
        all_results[dataset_name] = dataset_results
    
    return all_results


def summarize_results(results: Dict[str, Dict[str, Any]], include_linear: bool = True) -> None:
    """
    Print a summary table of all experiment results.
    
    Args:
        results: Results from run_experiment_batch
        include_linear: Whether to include linear reconstruction results
    """
    print(f"\n{'='*100}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*100}")
    
    # Create summary table
    if include_linear:
        print(f"{'Dataset':<25} {'Model':<25} {'PCA R²':<10} {'PCA Corr':<10} {'Linear R²':<12} {'Linear Corr':<12}")
        print("-" * 100)
    else:
        print(f"{'Dataset':<25} {'Model':<25} {'PCA R²':<15} {'PCA Corr':<12}")
        print("-" * 80)
    
    for dataset_name, dataset_results in results.items():
        for i, (model_name, model_results) in enumerate(dataset_results.items()):
            dataset_display = dataset_name if i == 0 else ""
            
            if 'error' in model_results:
                if include_linear:
                    print(f"{dataset_display:<25} {model_name:<25} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<12}")
                else:
                    print(f"{dataset_display:<25} {model_name:<25} {'ERROR':<15} {'ERROR':<12}")
            else:
                pca_r2 = f"{model_results['explained_variance_score']:.4f}"
                pca_corr = f"{model_results['correlation']:.4f}"
                
                if include_linear and 'linear_r2_score' in model_results:
                    linear_r2 = f"{model_results['linear_r2_score']:.4f}"
                    linear_corr = f"{model_results['linear_correlation']:.4f}"
                    print(f"{dataset_display:<25} {model_name:<25} {pca_r2:<10} {pca_corr:<10} {linear_r2:<12} {linear_corr:<12}")
                else:
                    print(f"{dataset_display:<25} {model_name:<25} {pca_r2:<15} {pca_corr:<12}")


def save_results_to_csv(results: Dict[str, Dict[str, Any]], 
                       filename: str = "experiment_results.csv") -> None:
    """
    Save experiment results to CSV file.
    
    Args:
        results: Results from run_experiment_batch
        filename: Output CSV filename
    """
    import pandas as pd
    
    rows = []
    for dataset_name, dataset_results in results.items():
        for model_name, model_results in dataset_results.items():
            if 'error' not in model_results:
                row = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'pca_explained_variance_score': model_results['explained_variance_score'],
                    'pca_correlation': model_results['correlation'],
                    'pca_mse': model_results['mse'],
                    'pca_explained_variance_ratio': model_results['pca_explained_variance_ratio']
                }
                
                # Add linear reconstruction results if available
                if 'linear_r2_score' in model_results:
                    row.update({
                        'linear_r2_score': model_results['linear_r2_score'],
                        'linear_correlation': model_results['linear_correlation'],
                        'linear_mse': model_results['linear_mse']
                    })
                
                rows.append(row)
            else:
                row = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'error': model_results['error']
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def analyze_embedding_dimensionality_impact(embedding_wrapper,
                                          dataset: List[str],
                                          true_values: List[float],
                                          model: str,
                                          max_components: int = 10) -> Dict[str, Any]:
    """
    Analyze how the number of PCA components affects reconstruction quality.
    
    Args:
        embedding_wrapper: Initialized EmbeddingWrapper instance
        dataset: List of string representations
        true_values: List of true numerical values
        model: Model name to test
        max_components: Maximum number of PCA components to test
        
    Returns:
        Dictionary with results for different numbers of components
    """
    # Get embeddings
    embeddings = embedding_wrapper.embed(dataset, model)
    
    results = {}
    for n_comp in range(1, max_components + 1):
        comp_results = compute_explained_variance_score(
            embeddings, true_values, n_components=n_comp
        )
        results[n_comp] = comp_results['explained_variance_score']
    
    return results