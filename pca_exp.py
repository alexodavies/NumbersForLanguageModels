#!/usr/bin/env python3
"""
PCA Experiment Module

Provides functionality for analyzing embeddings using Principal Component Analysis
and reconstructing numerical values from PCA components.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


class PCAAnalyzer:
    """Handles PCA analysis and reconstruction from embeddings."""
    
    def __init__(self, n_components: int = None, random_state: int = 42):
        """
        Initialize the PCA analyzer.
        
        Args:
            n_components: Number of PCA components to keep (None for all)
            random_state: Random state for reproducible results
        """
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = None
        self.pca = None
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray) -> 'PCAAnalyzer':
        """
        Fit the PCA model on embeddings.
        
        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
            
        Returns:
            Self for method chaining
        """
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(embeddings)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to PCA space.
        
        Args:
            embeddings: Array of embeddings to transform
            
        Returns:
            PCA-transformed embeddings
            
        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted before transformation")
        
        X_scaled = self.scaler.transform(embeddings)
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit PCA and transform embeddings in one step."""
        return self.fit(embeddings).transform(embeddings)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted first")
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_explained_variance(self) -> np.ndarray:
        """Get cumulative explained variance."""
        return np.cumsum(self.get_explained_variance_ratio())
    
    def get_components(self) -> np.ndarray:
        """Get the principal components (loadings)."""
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted first")
        return self.pca.components_


def evaluate_pca_reconstruction(embeddings: List[List[float]], 
                              values: List[float],
                              n_components: int = 1,
                              test_size: float = 0.2,
                              random_state: int = 42) -> Dict[str, Any]:
    """
    Evaluate numerical reconstruction using PCA components.
    
    Args:
        embeddings: List of embedding vectors
        values: List of corresponding numerical values
        n_components: Number of PCA components to use for reconstruction
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducible splits
        
    Returns:
        Dictionary with comprehensive PCA results
    """
    # Convert to numpy arrays
    X = np.array(embeddings)
    y = np.array(values)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Fit PCA
    pca_analyzer = PCAAnalyzer(n_components=n_components, random_state=random_state)
    train_pca = pca_analyzer.fit_transform(X_train)
    test_pca = pca_analyzer.transform(X_test)
    
    # Get explained variance ratios
    explained_var_ratios = pca_analyzer.get_explained_variance_ratio()
    cumulative_var = pca_analyzer.get_cumulative_explained_variance()
    
    # For single component, try direct correlation
    if n_components == 1:
        train_component = train_pca.flatten()
        test_component = test_pca.flatten()
        
        # Calculate R² for reconstruction
        train_r2 = r2_score(y_train, train_component)
        test_r2 = r2_score(y_test, test_component)
        
        # If negative, try flipping the component
        if test_r2 < 0:
            train_component = -train_component
            test_component = -test_component
            train_r2 = r2_score(y_train, train_component)
            test_r2 = r2_score(y_test, test_component)
        
        # Calculate other metrics
        train_mse = np.mean((y_train - train_component) ** 2)
        test_mse = np.mean((y_test - test_component) ** 2)
        train_mae = np.mean(np.abs(y_train - train_component))
        test_mae = np.mean(np.abs(y_test - test_component))
        
        return {
            'train_size': len(y_train),
            'test_size': len(y_test),
            'embedding_dim': X.shape[1],
            'n_components': n_components,
            'explained_variance_ratios': explained_var_ratios,
            'cumulative_explained_variance': cumulative_var,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_components': train_pca,
            'test_components': test_pca,
            'train_values': y_train,
            'test_values': y_test,
            'pca_analyzer': pca_analyzer
        }
    
    else:
        # For multiple components, we'd need to fit a linear model
        # This is a placeholder for more complex multi-component reconstruction
        return {
            'train_size': len(y_train),
            'test_size': len(y_test),
            'embedding_dim': X.shape[1],
            'n_components': n_components,
            'explained_variance_ratios': explained_var_ratios,
            'cumulative_explained_variance': cumulative_var,
            'train_components': train_pca,
            'test_components': test_pca,
            'train_values': y_train,
            'test_values': y_test,
            'pca_analyzer': pca_analyzer
        }


def analyze_pca_spectrum(embeddings: List[List[float]], 
                        max_components: int = 50) -> Dict[str, Any]:
    """
    Analyze the PCA spectrum of embeddings to understand dimensionality.
    
    Args:
        embeddings: List of embedding vectors
        max_components: Maximum number of components to analyze
        
    Returns:
        Dictionary with PCA spectrum analysis
    """
    X = np.array(embeddings)
    
    # Limit max_components to the actual dimensionality
    max_components = min(max_components, X.shape[1], X.shape[0] - 1)
    
    # Fit PCA with all requested components
    pca_analyzer = PCAAnalyzer(n_components=max_components)
    pca_analyzer.fit(X)
    
    explained_var_ratios = pca_analyzer.get_explained_variance_ratio()
    cumulative_var = pca_analyzer.get_cumulative_explained_variance()
    
    # Find number of components for different variance thresholds
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    components_for_threshold = {}
    
    for threshold in thresholds:
        idx = np.argmax(cumulative_var >= threshold)
        if cumulative_var[idx] >= threshold:
            components_for_threshold[f"{threshold:.0%}"] = idx + 1
        else:
            components_for_threshold[f"{threshold:.0%}"] = max_components
    
    return {
        'embedding_dim': X.shape[1],
        'n_samples': X.shape[0],
        'max_components_analyzed': max_components,
        'explained_variance_ratios': explained_var_ratios,
        'cumulative_explained_variance': cumulative_var,
        'components_for_variance_thresholds': components_for_threshold,
        'first_component_variance': explained_var_ratios[0] if len(explained_var_ratios) > 0 else 0,
        'effective_rank': np.sum(cumulative_var < 0.99) + 1  # Rough estimate
    }


def compare_pca_components(embeddings: List[List[float]], 
                          values: List[float],
                          component_counts: List[int] = [1, 2, 5, 10],
                          test_size: float = 0.2,
                          random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Compare reconstruction performance with different numbers of PCA components.
    
    Args:
        embeddings: List of embedding vectors
        values: List of corresponding numerical values
        component_counts: List of component counts to try
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducible results
        
    Returns:
        Dictionary mapping component counts to results
    """
    results = {}
    
    # Filter component counts to be reasonable
    max_possible = min(len(embeddings) - 1, len(embeddings[0]))
    valid_counts = [n for n in component_counts if n <= max_possible]
    
    for n_comp in valid_counts:
        results[f"{n_comp}_components"] = evaluate_pca_reconstruction(
            embeddings, values, 
            n_components=n_comp, 
            test_size=test_size, 
            random_state=random_state
        )
    
    return results