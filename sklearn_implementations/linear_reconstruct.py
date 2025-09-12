#!/usr/bin/env python3
"""
Linear Reconstruction Module

Provides functionality for reconstructing numerical values from embeddings
using linear models (Ridge regression).
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


class LinearReconstructor:
    """Handles linear reconstruction of numerical values from embeddings."""
    
    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        """
        Initialize the linear reconstructor.
        
        Args:
            alpha: Ridge regression regularization parameter
            random_state: Random state for reproducible results
        """
        self.alpha = alpha
        self.random_state = random_state
        self.scaler = None
        self.model = None
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray, values: np.ndarray) -> 'LinearReconstructor':
        """
        Fit the linear reconstruction model.
        
        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
            values: Array of target values (n_samples,)
            
        Returns:
            Self for method chaining
        """
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(embeddings)
        
        # Fit Ridge regression
        self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.model.fit(X_scaled, values)
        
        self.is_fitted = True
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict values from embeddings.
        
        Args:
            embeddings: Array of embeddings to predict from
            
        Returns:
            Predicted values
            
        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(embeddings)
        return self.model.predict(X_scaled)
    
    def evaluate(self, embeddings: np.ndarray, true_values: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            embeddings: Test embeddings
            true_values: True values
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(embeddings)
        r2 = r2_score(true_values, predictions)
        
        mse = np.mean((true_values - predictions) ** 2)
        mae = np.mean(np.abs(true_values - predictions))
        
        return {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'predictions': predictions
        }


def evaluate_linear_reconstruction(embeddings: List[List[float]], 
                                 values: List[float],
                                 test_size: float = 0.2,
                                 alpha: float = 1.0,
                                 random_state: int = 42) -> Dict[str, Any]:
    """
    Evaluate linear reconstruction performance with train/test split.
    
    Args:
        embeddings: List of embedding vectors
        values: List of corresponding numerical values
        test_size: Fraction of data to use for testing
        alpha: Ridge regression regularization parameter
        random_state: Random state for reproducible splits
        
    Returns:
        Dictionary with comprehensive results
    """
    # Convert to numpy arrays
    X = np.array(embeddings)
    y = np.array(values)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Fit model
    reconstructor = LinearReconstructor(alpha=alpha, random_state=random_state)
    reconstructor.fit(X_train, y_train)
    
    # Evaluate on both sets
    train_results = reconstructor.evaluate(X_train, y_train)
    test_results = reconstructor.evaluate(X_test, y_test)
    
    return {
        'train_size': len(y_train),
        'test_size': len(y_test),
        'embedding_dim': X.shape[1],
        'train_r2': train_results['r2'],
        'test_r2': test_results['r2'],
        'train_mse': train_results['mse'],
        'test_mse': test_results['mse'],
        'train_mae': train_results['mae'],
        'test_mae': test_results['mae'],
        'train_predictions': train_results['predictions'],
        'test_predictions': test_results['predictions'],
        'train_values': y_train,
        'test_values': y_test,
        'reconstructor': reconstructor
    }


def compare_linear_models(embeddings: List[List[float]], 
                         values: List[float],
                         alphas: List[float] = [0.1, 1.0, 10.0],
                         test_size: float = 0.2,
                         random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Compare linear reconstruction with different regularization parameters.
    
    Args:
        embeddings: List of embedding vectors
        values: List of corresponding numerical values
        alphas: List of Ridge regression alpha values to try
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducible results
        
    Returns:
        Dictionary mapping alpha values to results
    """
    results = {}
    
    for alpha in alphas:
        results[f"alpha_{alpha}"] = evaluate_linear_reconstruction(
            embeddings, values, test_size=test_size, 
            alpha=alpha, random_state=random_state
        )
    
    return results