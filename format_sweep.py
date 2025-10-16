#!/usr/bin/env python3
"""
Sweep Experiments Module - Extended with Format Comparison

Original: Tests how number SIZE affects embedding quality
New: Tests how number FORMAT affects embedding quality
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import random

# Your existing imports
from datasets import (
    real_positive_decimals, 
    real_positive_and_negative_decimals, 
    real_int_and_decimal
)
from linear_reconstruct import evaluate_linear_reconstruction
from pca_exp import evaluate_pca_reconstruction
from sklearn.decomposition import PCA


# ============================================================================
# FORMAT CONVERSION FUNCTIONS (New)
# ============================================================================

def generate_random_integers(n_samples: int, random_state: int = 42) -> List[int]:
    """Generate random integers from -1,000,000,000,000 to 1,000,000,000,000."""
    random.seed(random_state)
    np.random.seed(random_state)
    return [random.randint(-1_000_000_000_000, 1_000_000_000_000) for _ in range(n_samples)]


def format_scientific_3sf(num: int) -> str:
    """Scientific notation with 3 significant figures."""
    return f"{num:.2e}"


def format_rounded_3sf(num: int) -> str:
    """Round to 3 significant figures."""
    if num == 0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(num))))
    rounded = round(num, -magnitude + 2)
    return str(int(rounded))


def number_to_words(num: int) -> str:
    """Convert an integer to its written English form (supports up to 10^20)."""
    if num == 0:
        return "zero"
    
    negative = num < 0
    num = abs(num)
    
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
             "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    def convert_hundreds(n):
        if n == 0:
            return ""
        elif n < 10:
            return ones[n]
        elif n < 20:
            return teens[n - 10]
        elif n < 100:
            return tens[n // 10] + (" " + ones[n % 10] if n % 10 != 0 else "")
        else:
            hundred_part = ones[n // 100] + " hundred"
            rest = n % 100
            return hundred_part if rest == 0 else hundred_part + " " + convert_hundreds(rest)
    
    # Break down into quintillions, quadrillions, trillions, billions, millions, thousands, and hundreds
    quintillions = num // 1_000_000_000_000_000_000
    quadrillions = (num % 1_000_000_000_000_000_000) // 1_000_000_000_000_000
    trillions = (num % 1_000_000_000_000_000) // 1_000_000_000_000
    billions = (num % 1_000_000_000_000) // 1_000_000_000
    millions = (num % 1_000_000_000) // 1_000_000
    thousands = (num % 1_000_000) // 1_000
    hundreds = num % 1_000
    
    parts = []
    if quintillions > 0:
        parts.append(convert_hundreds(quintillions) + " quintillion")
    if quadrillions > 0:
        parts.append(convert_hundreds(quadrillions) + " quadrillion")
    if trillions > 0:
        parts.append(convert_hundreds(trillions) + " trillion")
    if billions > 0:
        parts.append(convert_hundreds(billions) + " billion")
    if millions > 0:
        parts.append(convert_hundreds(millions) + " million")
    if thousands > 0:
        parts.append(convert_hundreds(thousands) + " thousand")
    if hundreds > 0:
        parts.append(convert_hundreds(hundreds))
    
    result = " ".join(parts) if parts else "zero"
    if negative:
        result = "negative " + result
    
    return result


# ============================================================================
# EXISTING CODE
# ============================================================================

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
    
    # ========================================================================
    # NEW METHOD: Format Comparison
    # ========================================================================
    
    def run_format_comparison(self) -> Dict[str, Any]:
        """
        NEW: Compare how different formats affect embedding quality.
        Tests standard, scientific (3sf), rounded (3sf), and written words.
        """
        experiment_name = "format_comparison"
        
        print(f"\n{'='*50}")
        print(f"FORMAT COMPARISON EXPERIMENT")
        print(f"{'='*50}")
        
        # Generate base integers
        base_values = generate_random_integers(
            self.config.n_samples, 
            self.config.random_state
        )
        print(f"Generated {len(base_values)} integers")
        print(f"  Range: {min(base_values)} to {max(base_values)}")
        
        # Define formats
        formats = {
            'standard': lambda x: str(x),
            'scientific_3sf': lambda x: format_scientific_3sf(x),
            'rounded_3sf': lambda x: format_rounded_3sf(x),
            'written_words': lambda x: number_to_words(x),
        }
        
        # Show examples
        print(f"\nExample conversions (first value = {base_values[0]}):")
        for fmt_name, fmt_func in formats.items():
            print(f"  {fmt_name:20s} → {fmt_func(base_values[0])}")
        
        results = {
            'experiment': experiment_name,
            'n_samples': len(base_values),
            'formats': list(formats.keys()),
            'models': {}
        }
        
        # Test each model
        for model in self.available_models:
            print(f"\n🤖 Testing {model}")
            
            model_results = {
                'formats': [],
                'linear_r2': [],
                'pca_r2': [],
            }
            
            # Test each format
            for format_name, format_func in formats.items():
                print(f"  📝 Format: {format_name}")
                
                try:
                    # Convert all values to this format
                    formatted_texts = [format_func(val) for val in base_values]
                    
                    # Show examples
                    print(f"     Examples: {formatted_texts[:3]}")
                    
                    # Get embeddings
                    if hasattr(self.wrapper, 'embed_with_params'):
                        embeddings = self.wrapper.embed_with_params(
                            formatted_texts, model, experiment_name, format_name,
                            random_state=self.config.random_state
                        )
                    else:
                        embeddings = self.wrapper.embed(formatted_texts, model)
                    
                    # Evaluate reconstruction
                    linear_results = evaluate_linear_reconstruction(
                        embeddings, base_values,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
                    
                    # Run PCA with 1 component first for R² score
                    pca_results_1 = evaluate_pca_reconstruction(
                        embeddings, base_values,
                        n_components=1,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
                    
                    # Run PCA with up to 5 components for explained variance analysis
                    max_components = min(5, len(embeddings[0]), len(base_values))
                    pca_results_5 = evaluate_pca_reconstruction(
                        embeddings, base_values,
                        n_components=max_components,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
                    
                    # Store results
                    model_results['formats'].append(format_name)
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
                            model_results[key].append(0.0)  # If fewer components available
                    
                    # Calculate cumulative explained variance
                    if 'pca_cumulative_var' not in model_results:
                        model_results['pca_cumulative_var'] = []
                    
                    cumulative_var = sum(explained_var)
                    model_results['pca_cumulative_var'].append(cumulative_var)
                    
                    print(f"     Linear R²: {linear_results['test_r2']:.3f}, "
                          f"PCA R²: {pca_results_1['test_r2']:.3f}")
                    print(f"     PCA explained variance (top {len(explained_var)}): {[f'{v:.3f}' for v in explained_var]}")
                    print(f"     Cumulative: {cumulative_var:.3f}")
                
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    model_results['formats'].append(format_name)
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
    
    # ========================================================================
    # NEW METHOD: Format × Size Sweep
    # ========================================================================
    
    def run_format_size_sweep(self, size_range: List[int] = None) -> Dict[str, Any]:
        """
        NEW: Sweep across BOTH format AND size.
        For each format, test numbers of increasing magnitude (1-20 places).
        
        Args:
            size_range: List of number sizes (in places/digits) to test.
                       Default is [1, 2, ..., 20]
        """
        experiment_name = "format_size_sweep"
        
        if size_range is None:
            size_range = list(range(1, 21))
        
        print(f"\n{'='*60}")
        print(f"FORMAT × SIZE SWEEP EXPERIMENT")
        print(f"{'='*60}")
        print(f"Testing {len(size_range)} sizes across 4 formats")
        print(f"Size range: {min(size_range)} to {max(size_range)} places")
        
        # Define formats
        formats = {
            'standard': lambda x: str(x),
            'scientific_3sf': lambda x: format_scientific_3sf(x),
            'rounded_3sf': lambda x: format_rounded_3sf(x),
            'written_words': lambda x: number_to_words(x),
        }
        
        results = {
            'experiment': experiment_name,
            'n_samples': self.config.n_samples,
            'size_range': size_range,
            'formats': list(formats.keys()),
            'models': {}
        }
        
        # Test each model
        for model in self.available_models:
            print(f"\n🤖 Testing {model}")
            
            model_results = {
                'format_name': [],
                'size': [],
                'linear_r2': [],
                'pca_r2': [],
                'pca_explained_var_1': [],
                'pca_cumulative_var': []
            }
            
            # Test each format
            for format_name, format_func in formats.items():
                print(f"\n  📋 Format: {format_name}")
                
                # Test each size
                for size in size_range:
                    print(f"    📏 Size {size}...", end=" ")
                    
                    try:
                        # Generate numbers of specific magnitude
                        import random
                        import numpy as np
                        seed = self.config.random_state + size * 100
                        random.seed(seed)
                        np.random.seed(seed)
                        
                        # Generate numbers with 'size' digits
                        # For size=1: -9 to 9
                        # For size=2: -99 to 99
                        # For size=20: ~±10^19
                        if size == 1:
                            min_val, max_val = -9, 9
                        else:
                            max_val = 10**size - 1
                            min_val = -(10**size - 1)
                        
                        base_values = [
                            random.randint(min_val, max_val) 
                            for _ in range(self.config.n_samples)
                        ]
                        
                        # Convert to format
                        formatted_texts = [format_func(val) for val in base_values]
                        
                        # Show one example
                        if size in [1, 5, 10, 15, 20]:
                            print(f"Ex: {base_values[0]} → {formatted_texts[0][:50]}...")
                        
                        # Get embeddings
                        if hasattr(self.wrapper, 'embed_with_params'):
                            embeddings = self.wrapper.embed_with_params(
                                formatted_texts, model, experiment_name, 
                                f"{format_name}_size{size}",
                                random_state=self.config.random_state
                            )
                        else:
                            embeddings = self.wrapper.embed(formatted_texts, model)
                        
                        # Evaluate reconstruction
                        linear_results = evaluate_linear_reconstruction(
                            embeddings, base_values,
                            test_size=self.config.test_size,
                            random_state=self.config.random_state
                        )
                        
                        # PCA with 1 component for R² score
                        pca_results_1 = evaluate_pca_reconstruction(
                            embeddings, base_values,
                            n_components=1,
                            test_size=self.config.test_size,
                            random_state=self.config.random_state
                        )
                        
                        # PCA with up to 5 components for explained variance
                        max_components = min(5, len(embeddings[0]), len(base_values))
                        pca_results_5 = evaluate_pca_reconstruction(
                            embeddings, base_values,
                            n_components=max_components,
                            test_size=self.config.test_size,
                            random_state=self.config.random_state
                        )
                        
                        # Store results
                        model_results['format_name'].append(format_name)
                        model_results['size'].append(size)
                        model_results['linear_r2'].append(linear_results['test_r2'])
                        model_results['pca_r2'].append(pca_results_1['test_r2'])
                        
                        # Store first component explained variance
                        explained_var = pca_results_5['explained_variance_ratios']
                        model_results['pca_explained_var_1'].append(
                            explained_var[0] if len(explained_var) > 0 else 0.0
                        )
                        
                        # Store cumulative variance
                        cumulative_var = sum(explained_var)
                        model_results['pca_cumulative_var'].append(cumulative_var)
                        
                        print(f"Linear R²={linear_results['test_r2']:.3f}, PCA R²={pca_results_1['test_r2']:.3f}")
                    
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        model_results['format_name'].append(format_name)
                        model_results['size'].append(size)
                        model_results['linear_r2'].append(np.nan)
                        model_results['pca_r2'].append(np.nan)
                        model_results['pca_explained_var_1'].append(np.nan)
                        model_results['pca_cumulative_var'].append(np.nan)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    def run_format_size_sweep_kfold(self, size_range: List[int] = None, n_splits: int = 5) -> Dict[str, Any]:
        """
        K-fold cross-validation version of format×size sweep.
        
        Args:
            size_range: List of number sizes (in places/digits) to test.
                       Default is [1, 2, ..., 20]
            n_splits: Number of k-fold splits (default 5)
        """
        from sklearn.model_selection import KFold
        
        experiment_name = "format_size_sweep_kfold"
        
        if size_range is None:
            size_range = list(range(1, 21))
        
        print(f"\n{'='*60}")
        print(f"FORMAT × SIZE SWEEP (K-FOLD) EXPERIMENT")
        print(f"{'='*60}")
        print(f"Testing {len(size_range)} sizes across 4 formats")
        print(f"Size range: {min(size_range)} to {max(size_range)} places")
        print(f"K-fold splits: {n_splits}")
        
        # Define formats
        formats = {
            'standard': lambda x: str(x),
            'scientific_3sf': lambda x: format_scientific_3sf(x),
            'rounded_3sf': lambda x: format_rounded_3sf(x),
            'written_words': lambda x: number_to_words(x),
        }
        
        results = {
            'experiment': experiment_name,
            'n_samples': self.config.n_samples,
            'n_splits': n_splits,
            'size_range': size_range,
            'formats': list(formats.keys()),
            'models': {}
        }
        
        # Test each model
        for model in self.available_models:
            print(f"\n🤖 Testing {model}")
            
            model_results = {
                'format_name': [],
                'size': [],
                'linear_r2_mean': [],
                'linear_r2_std': [],
                'pca_r2_mean': [],
                'pca_r2_std': [],
                'pca_explained_var_1_mean': [],
                'pca_explained_var_1_std': [],
                'pca_cumulative_var_mean': [],
                'pca_cumulative_var_std': []
            }
            
            # Test each format
            for format_name, format_func in formats.items():
                print(f"\n  📋 Format: {format_name}")
                
                # Test each size
                for size in size_range:
                    print(f"    📏 Size {size}...", end=" ")
                    
                    try:
                        # Generate numbers of specific magnitude
                        import random
                        import numpy as np
                        seed = self.config.random_state + size * 100
                        random.seed(seed)
                        np.random.seed(seed)
                        
                        # Generate numbers with 'size' digits
                        if size == 1:
                            min_val, max_val = -9, 9
                        else:
                            max_val = 10**size - 1
                            min_val = -(10**size - 1)
                        
                        base_values = [
                            random.randint(min_val, max_val) 
                            for _ in range(self.config.n_samples)
                        ]
                        
                        # Convert to format
                        formatted_texts = [format_func(val) for val in base_values]
                        
                        # Get embeddings
                        if hasattr(self.wrapper, 'embed_with_params'):
                            embeddings = self.wrapper.embed_with_params(
                                formatted_texts, model, experiment_name, 
                                f"{format_name}_size{size}",
                                random_state=self.config.random_state
                            )
                        else:
                            embeddings = self.wrapper.embed(formatted_texts, model)
                        
                        # K-fold cross-validation
                        embeddings_array = np.array(embeddings)
                        values_array = np.array(base_values)
                        
                        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
                        
                        fold_linear_r2 = []
                        fold_pca_r2 = []
                        fold_pca_var_1 = []
                        fold_pca_cumulative = []
                        
                        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(embeddings_array)):
                            X_train = embeddings_array[train_idx]
                            X_test = embeddings_array[test_idx]
                            y_train = values_array[train_idx]
                            y_test = values_array[test_idx]
                            
                            # Linear regression
                            from sklearn.linear_model import LinearRegression
                            from sklearn.metrics import r2_score
                            
                            lr = LinearRegression()
                            lr.fit(X_train, y_train)
                            y_pred = lr.predict(X_test)
                            linear_r2 = r2_score(y_test, y_pred)
                            fold_linear_r2.append(linear_r2)
                            
                            # PCA with 1 component
                            pca_1 = PCA(n_components=1)
                            X_train_pca = pca_1.fit_transform(X_train)
                            X_test_pca = pca_1.transform(X_test)
                            
                            lr_pca = LinearRegression()
                            lr_pca.fit(X_train_pca, y_train)
                            y_pred_pca = lr_pca.predict(X_test_pca)
                            pca_r2 = r2_score(y_test, y_pred_pca)
                            fold_pca_r2.append(pca_r2)
                            
                            # PCA with up to 5 components for variance analysis
                            max_components = min(5, X_train.shape[1], X_train.shape[0])
                            pca_5 = PCA(n_components=max_components)
                            pca_5.fit(X_train)
                            
                            explained_var = pca_5.explained_variance_ratio_
                            fold_pca_var_1.append(explained_var[0] if len(explained_var) > 0 else 0.0)
                            fold_pca_cumulative.append(sum(explained_var))
                        
                        # Store mean and std across folds
                        model_results['format_name'].append(format_name)
                        model_results['size'].append(size)
                        model_results['linear_r2_mean'].append(np.mean(fold_linear_r2))
                        model_results['linear_r2_std'].append(np.std(fold_linear_r2))
                        model_results['pca_r2_mean'].append(np.mean(fold_pca_r2))
                        model_results['pca_r2_std'].append(np.std(fold_pca_r2))
                        model_results['pca_explained_var_1_mean'].append(np.mean(fold_pca_var_1))
                        model_results['pca_explained_var_1_std'].append(np.std(fold_pca_var_1))
                        model_results['pca_cumulative_var_mean'].append(np.mean(fold_pca_cumulative))
                        model_results['pca_cumulative_var_std'].append(np.std(fold_pca_cumulative))
                        
                        print(f"Linear R²={np.mean(fold_linear_r2):.3f}±{np.std(fold_linear_r2):.3f}, "
                              f"PCA R²={np.mean(fold_pca_r2):.3f}±{np.std(fold_pca_r2):.3f}")
                    
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        model_results['format_name'].append(format_name)
                        model_results['size'].append(size)
                        model_results['linear_r2_mean'].append(np.nan)
                        model_results['linear_r2_std'].append(np.nan)
                        model_results['pca_r2_mean'].append(np.nan)
                        model_results['pca_r2_std'].append(np.nan)
                        model_results['pca_explained_var_1_mean'].append(np.nan)
                        model_results['pca_explained_var_1_std'].append(np.nan)
                        model_results['pca_cumulative_var_mean'].append(np.nan)
                        model_results['pca_cumulative_var_std'].append(np.nan)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    # ========================================================================
    # EXISTING METHODS (keeping all your original code)
    # ========================================================================
    
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
                print(f"  🔢 Size {size}...")
                
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
                    
                    pca_results_1 = evaluate_pca_reconstruction(
                        embeddings, values,
                        n_components=1,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
                    
                    # Store results
                    model_results['sizes'].append(size)
                    model_results['linear_r2'].append(linear_results['test_r2'])
                    model_results['pca_r2'].append(pca_results_1['test_r2'])
                    
                    print(f"    Linear R²: {linear_results['test_r2']:.3f}, "
                          f"PCA R²: {pca_results_1['test_r2']:.3f}")
                
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    model_results['sizes'].append(size)
                    model_results['linear_r2'].append(np.nan)
                    model_results['pca_r2'].append(np.nan)
            
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
        
        fixed_decimal = 3
        
        for model in self.available_models:
            print(f"\n🤖 Testing {model}")
            
            model_results = {
                'sizes': [],
                'linear_r2': [],
                'pca_r2': []
            }
            
            for int_size in self.config.mixed_int_sizes:
                print(f"  🔢 Int={int_size}, Dec={fixed_decimal}...")
                
                try:
                    # Generate mixed dataset with fixed seed
                    import random
                    import numpy as np
                    seed = self.config.random_state + int_size + 1000
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    texts = real_int_and_decimal(
                        self.config.n_samples, 
                        int_size, 
                        int_size
                    )
                    values = [float(x) for x in texts]
                    
                    # Get embeddings
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
                    
                    pca_results_1 = evaluate_pca_reconstruction(
                        embeddings, values,
                        n_components=1,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
                    
                    # Store results
                    model_results['sizes'].append(int_size)
                    model_results['linear_r2'].append(linear_results['test_r2'])
                    model_results['pca_r2'].append(pca_results_1['test_r2'])
                    
                    print(f"    Linear R²: {linear_results['test_r2']:.3f}, "
                          f"PCA R²: {pca_results_1['test_r2']:.3f}")
                
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    model_results['sizes'].append(int_size)
                    model_results['linear_r2'].append(np.nan)
                    model_results['pca_r2'].append(np.nan)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    def run_all_sweeps(self, include_format_comparison: bool = True, 
                      include_format_size_sweep: bool = False) -> Dict[str, Any]:
        """
        Run all sweep experiments.
        
        Args:
            include_format_comparison: If True, run format comparison (fixed size)
            include_format_size_sweep: If True, run format × size sweep (1-20 places)
        """
        print("🔬 Running all sweep experiments...")
        
        all_results = {}
        all_results['positive_decimals'] = self.run_decimal_sweep(positive_only=True)
        all_results['mixed_decimals'] = self.run_decimal_sweep(positive_only=False)
        all_results['mixed_int_decimal'] = self.run_mixed_sweep()
        
        if include_format_comparison:
            all_results['format_comparison'] = self.run_format_comparison()
        
        if include_format_size_sweep:
            all_results['format_size_sweep'] = self.run_format_size_sweep()
        
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
            
            # Use different plotting method based on experiment type
            if exp_name == 'format_comparison':
                self._plot_format_comparison(exp_name)
            elif exp_name == 'format_size_sweep':
                self._plot_format_size_sweep(exp_name)
            elif exp_name == 'format_size_sweep_kfold':
                self._plot_format_size_sweep_kfold(exp_name)
            else:
                self._plot_experiment(exp_name)
    
    def _plot_format_comparison(self, exp_name: str) -> None:
        """Plot format comparison results (bar chart) with PCA variance analysis."""
        results = self.results[exp_name]
        n_models = len(results['models'])
        
        # Create 2x2 subplot grid for each model
        fig, axes = plt.subplots(n_models, 4, figsize=(20, 5*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model, model_results) in enumerate(results['models'].items()):
            x_pos = np.arange(len(model_results['formats']))
            
            # Plot 1: Linear performance
            ax1 = axes[idx, 0]
            bars1 = ax1.bar(x_pos, model_results['linear_r2'], alpha=0.7)
            ax1.set_xlabel('Format')
            ax1.set_ylabel('Linear Test R²')
            ax1.set_title(f'Linear Reconstruction - {model}')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(model_results['formats'], rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, 1)
            
            # Add value labels
            for bar, val in zip(bars1, model_results['linear_r2']):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 2: PCA performance
            ax2 = axes[idx, 1]
            bars2 = ax2.bar(x_pos, model_results['pca_r2'], alpha=0.7, color='orange')
            ax2.set_xlabel('Format')
            ax2.set_ylabel('PCA Test R²')
            ax2.set_title(f'PCA Reconstruction - {model}')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(model_results['formats'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1)
            
            # Add value labels
            for bar, val in zip(bars2, model_results['pca_r2']):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 3: First PCA component explained variance
            ax3 = axes[idx, 2]
            if 'pca_explained_var_1' in model_results:
                bars3 = ax3.bar(x_pos, model_results['pca_explained_var_1'], alpha=0.7, color='green')
                ax3.set_xlabel('Format')
                ax3.set_ylabel('PCA Component 1 Explained Variance')
                ax3.set_title(f'First Component Variance - {model}')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(model_results['formats'], rotation=45, ha='right')
                ax3.grid(True, alpha=0.3, axis='y')
                ax3.set_ylim(0, 1)
                
                # Add value labels
                for bar, val in zip(bars3, model_results['pca_explained_var_1']):
                    if not np.isnan(val):
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 4: Cumulative explained variance (top 5 components)
            ax4 = axes[idx, 3]
            if 'pca_cumulative_var' in model_results:
                bars4 = ax4.bar(x_pos, model_results['pca_cumulative_var'], alpha=0.7, color='red')
                ax4.set_xlabel('Format')
                ax4.set_ylabel('Cumulative Explained Variance (Top 5)')
                ax4.set_title(f'Cumulative PCA Variance - {model}')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(model_results['formats'], rotation=45, ha='right')
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.set_ylim(0, 1)
                
                # Add value labels
                for bar, val in zip(bars4, model_results['pca_cumulative_var']):
                    if not np.isnan(val):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        filename = f"sweep_{exp_name}.png"
        filepath = os.path.join(self.config.plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Saved: {filepath}")
    
    def _plot_format_size_sweep(self, exp_name: str) -> None:
        """Plot format×size sweep results (line charts, one per format)."""
        results = self.results[exp_name]
        n_models = len(results['models'])
        
        # Create 2x2 subplot grid: Linear R², PCA R², PCA Var Component 1, Cumulative Var
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define colors and markers for each format
        format_styles = {
            'standard': {'color': 'blue', 'marker': 'o', 'label': 'Standard'},
            'scientific_3sf': {'color': 'green', 'marker': 's', 'label': 'Scientific (3sf)'},
            'rounded_3sf': {'color': 'orange', 'marker': '^', 'label': 'Rounded (3sf)'},
            'written_words': {'color': 'red', 'marker': 'D', 'label': 'Written Words'}
        }
        
        for model, model_results in results['models'].items():
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(model_results)
            
            # Plot 1: Linear R² vs Size for each format
            ax1 = axes[0, 0]
            for format_name, style in format_styles.items():
                format_data = df[df['format_name'] == format_name]
                if not format_data.empty:
                    ax1.plot(format_data['size'], format_data['linear_r2'],
                            color=style['color'], marker=style['marker'], 
                            label=f"{style['label']} ({model})", 
                            linewidth=2, markersize=6, alpha=0.7)
            
            ax1.set_xlabel('Number Size (digits)', fontsize=11)
            ax1.set_ylabel('Linear Test R²', fontsize=11)
            ax1.set_title('Linear Reconstruction Performance', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.05, 1.05)
            
            # Plot 2: PCA R² vs Size for each format
            ax2 = axes[0, 1]
            for format_name, style in format_styles.items():
                format_data = df[df['format_name'] == format_name]
                if not format_data.empty:
                    ax2.plot(format_data['size'], format_data['pca_r2'],
                            color=style['color'], marker=style['marker'], 
                            label=f"{style['label']} ({model})", 
                            linewidth=2, markersize=6, alpha=0.7)
            
            ax2.set_xlabel('Number Size (digits)', fontsize=11)
            ax2.set_ylabel('PCA Test R²', fontsize=11)
            ax2.set_title('PCA Reconstruction Performance', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.05, 1.05)
            
            # Plot 3: First PCA Component Explained Variance
            ax3 = axes[1, 0]
            for format_name, style in format_styles.items():
                format_data = df[df['format_name'] == format_name]
                if not format_data.empty:
                    ax3.plot(format_data['size'], format_data['pca_explained_var_1'],
                            color=style['color'], marker=style['marker'], 
                            label=f"{style['label']} ({model})", 
                            linewidth=2, markersize=6, alpha=0.7)
            
            ax3.set_xlabel('Number Size (digits)', fontsize=11)
            ax3.set_ylabel('First Component Explained Variance', fontsize=11)
            ax3.set_title('PCA Component 1 Variance', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-0.05, 1.05)
            
            # Plot 4: Cumulative Explained Variance (top 5 components)
            ax4 = axes[1, 1]
            for format_name, style in format_styles.items():
                format_data = df[df['format_name'] == format_name]
                if not format_data.empty:
                    ax4.plot(format_data['size'], format_data['pca_cumulative_var'],
                            color=style['color'], marker=style['marker'], 
                            label=f"{style['label']} ({model})", 
                            linewidth=2, markersize=6, alpha=0.7)
            
            ax4.set_xlabel('Number Size (digits)', fontsize=11)
            ax4.set_ylabel('Cumulative Explained Variance (5 comp)', fontsize=11)
            ax4.set_title('Cumulative PCA Variance', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        filename = f"sweep_{exp_name}.png"
        filepath = os.path.join(self.config.plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Saved: {filepath}")

    def _plot_format_size_sweep_kfold(self, exp_name: str) -> None:
        """Plot format×size sweep k-fold results with error bars."""
        results = self.results[exp_name]
        
        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define colors and markers for each format
        format_styles = {
            'standard': {'color': 'blue', 'marker': 'o', 'label': 'Standard'},
            'scientific_3sf': {'color': 'green', 'marker': 's', 'label': 'Scientific (3sf)'},
            'rounded_3sf': {'color': 'orange', 'marker': '^', 'label': 'Rounded (3sf)'},
            'written_words': {'color': 'red', 'marker': 'D', 'label': 'Written Words'}
        }
        
        for model, model_results in results['models'].items():
            df = pd.DataFrame(model_results)
            
            # Plot 1: Linear R² with error bars
            ax1 = axes[0, 0]
            for format_name, style in format_styles.items():
                format_data = df[df['format_name'] == format_name]
                if not format_data.empty:
                    ax1.errorbar(format_data['size'], format_data['linear_r2_mean'],
                                yerr=format_data['linear_r2_std'],
                                color=style['color'], marker=style['marker'], 
                                label=f"{style['label']} ({model})", 
                                linewidth=2, markersize=6, alpha=0.7, capsize=3)
            
            ax1.set_xlabel('Number Size (digits)', fontsize=11)
            ax1.set_ylabel('Linear R² (mean ± std)', fontsize=11)
            ax1.set_title(f'Linear Reconstruction (K-Fold, n={results["n_splits"]})', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.05, 1.05)
            
            # Plot 2: PCA R² with error bars
            ax2 = axes[0, 1]
            for format_name, style in format_styles.items():
                format_data = df[df['format_name'] == format_name]
                if not format_data.empty:
                    ax2.errorbar(format_data['size'], format_data['pca_r2_mean'],
                                yerr=format_data['pca_r2_std'],
                                color=style['color'], marker=style['marker'], 
                                label=f"{style['label']} ({model})", 
                                linewidth=2, markersize=6, alpha=0.7, capsize=3)
            
            ax2.set_xlabel('Number Size (digits)', fontsize=11)
            ax2.set_ylabel('PCA R² (mean ± std)', fontsize=11)
            ax2.set_title(f'PCA Reconstruction (K-Fold, n={results["n_splits"]})', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.05, 1.05)
            
            # Plot 3: First PCA Component Variance with error bars
            ax3 = axes[1, 0]
            for format_name, style in format_styles.items():
                format_data = df[df['format_name'] == format_name]
                if not format_data.empty:
                    ax3.errorbar(format_data['size'], format_data['pca_explained_var_1_mean'],
                                yerr=format_data['pca_explained_var_1_std'],
                                color=style['color'], marker=style['marker'], 
                                label=f"{style['label']} ({model})", 
                                linewidth=2, markersize=6, alpha=0.7, capsize=3)
            
            ax3.set_xlabel('Number Size (digits)', fontsize=11)
            ax3.set_ylabel('Component 1 Var (mean ± std)', fontsize=11)
            ax3.set_title(f'PCA Component 1 Variance (K-Fold)', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-0.05, 1.05)
            
            # Plot 4: Cumulative Variance with error bars
            ax4 = axes[1, 1]
            for format_name, style in format_styles.items():
                format_data = df[df['format_name'] == format_name]
                if not format_data.empty:
                    ax4.errorbar(format_data['size'], format_data['pca_cumulative_var_mean'],
                                yerr=format_data['pca_cumulative_var_std'],
                                color=style['color'], marker=style['marker'], 
                                label=f"{style['label']} ({model})", 
                                linewidth=2, markersize=6, alpha=0.7, capsize=3)
            
            ax4.set_xlabel('Number Size (digits)', fontsize=11)
            ax4.set_ylabel('Cumulative Var (mean ± std)', fontsize=11)
            ax4.set_title(f'Cumulative PCA Variance (K-Fold)', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        filename = f"sweep_{exp_name}.png"
        filepath = os.path.join(self.config.plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Saved: {filepath}")
    
    def _plot_experiment(self, exp_name: str) -> None:
        """Plot results for one size-sweep experiment (line chart)."""
        results = self.results[exp_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
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
        
        # Plot 2: PCA performance
        for model, model_results in results['models'].items():
            ax2.plot(model_results['sizes'], model_results['pca_r2'], 
                    marker='s', label=model, linewidth=2)
        
        ax2.set_xlabel('Size (digits)')
        ax2.set_ylabel('PCA Test R²')
        ax2.set_title(f'PCA Performance - {exp_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        filename = f"sweep_{exp_name}.png"
        filepath = os.path.join(self.config.plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Saved: {filepath}")
    
    def export_sweep_results(self, filename: str = None) -> None:
        """Export results to CSV."""
        if filename is None:
            filename = os.path.join(self.config.results_dir, "number_size_sweep_results.csv")
        
        if not self.results:
            print("No results to export")
            return
        
        rows = []
        
        for exp_name, results in self.results.items():
            for model, model_results in results['models'].items():
                if exp_name == 'format_comparison':
                    # Format comparison has different structure
                    for i, fmt in enumerate(model_results['formats']):
                        if i < len(model_results['linear_r2']):
                            row = {
                                'experiment': exp_name,
                                'model': model,
                                'format': fmt,
                                'size': fmt,  # Use format name as "size" for consistency
                                'linear_r2': model_results['linear_r2'][i],
                                'pca_r2': model_results['pca_r2'][i]
                            }
                            # Add PCA explained variance for all 5 components
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
                
                elif exp_name == 'format_size_sweep':
                    # Format×size sweep has format_name, size, and metrics
                    for i in range(len(model_results['size'])):
                        row = {
                            'experiment': exp_name,
                            'model': model,
                            'format': model_results['format_name'][i],
                            'size': model_results['size'][i],
                            'linear_r2': model_results['linear_r2'][i],
                            'pca_r2': model_results['pca_r2'][i]
                        }
                        
                        # Add PCA metrics if available
                        if 'pca_explained_var_1' in model_results:
                            row['pca_var_comp_1'] = model_results['pca_explained_var_1'][i]
                        if 'pca_cumulative_var' in model_results:
                            row['pca_cumulative_var_5'] = model_results['pca_cumulative_var'][i]
                        
                        rows.append(row)
                
                elif exp_name == 'format_size_sweep_kfold':
                    # K-fold format×size sweep has mean and std for each metric
                    for i in range(len(model_results['size'])):
                        row = {
                            'experiment': exp_name,
                            'model': model,
                            'format': model_results['format_name'][i],
                            'size': model_results['size'][i],
                            'linear_r2_mean': model_results['linear_r2_mean'][i],
                            'linear_r2_std': model_results['linear_r2_std'][i],
                            'pca_r2_mean': model_results['pca_r2_mean'][i],
                            'pca_r2_std': model_results['pca_r2_std'][i]
                        }
                        
                        # Add PCA metrics if available
                        if 'pca_explained_var_1_mean' in model_results:
                            row['pca_var_comp_1_mean'] = model_results['pca_explained_var_1_mean'][i]
                            row['pca_var_comp_1_std'] = model_results['pca_explained_var_1_std'][i]
                        if 'pca_cumulative_var_mean' in model_results:
                            row['pca_cumulative_var_5_mean'] = model_results['pca_cumulative_var_mean'][i]
                            row['pca_cumulative_var_5_std'] = model_results['pca_cumulative_var_std'][i]
                        
                        rows.append(row)
                
                else:
                    # Size sweeps have normal structure
                    for i, size in enumerate(model_results['sizes']):
                        if i < len(model_results['linear_r2']):
                            rows.append({
                                'experiment': exp_name,
                                'model': model,
                                'format': 'standard',  # Size sweeps use standard format
                                'size': size,
                                'linear_r2': model_results['linear_r2'][i],
                                'pca_r2': model_results['pca_r2'][i]
                            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        print(f"📊 Exported {len(df)} results to {filename}")
    
    def generate_sweep_report(self) -> str:
        """Generate summary report."""
        if not self.results:
            return "No results available."
        
        lines = []
        lines.append("SWEEP RESULTS SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        
        for exp_name, results in self.results.items():
            lines.append(f"EXPERIMENT: {exp_name.upper()}")
            lines.append("-" * 50)
            
            if exp_name == 'format_comparison':
                # Special handling for format comparison
                for model, model_results in results['models'].items():
                    lines.append(f"\n{model}:")
                    formats = model_results['formats']
                    linear_r2 = model_results['linear_r2']
                    
                    for fmt, score in zip(formats, linear_r2):
                        if not np.isnan(score):
                            lines.append(f"  {fmt:20s}: R² = {score:.3f}")
                    
                    # Find best format
                    valid_idx = [i for i, s in enumerate(linear_r2) if not np.isnan(s)]
                    if valid_idx:
                        best_idx = valid_idx[np.argmax([linear_r2[i] for i in valid_idx])]
                        lines.append(f"  → Best: {formats[best_idx]} ({linear_r2[best_idx]:.3f})")
            
            elif exp_name in ['format_size_sweep', 'format_size_sweep_kfold']:
                # Format×size sweeps
                is_kfold = 'kfold' in exp_name
                r2_key = 'linear_r2_mean' if is_kfold else 'linear_r2'
                
                for model, model_results in results['models'].items():
                    lines.append(f"\n{model}:")
                    
                    # Group by format
                    formats = list(set(model_results['format_name']))
                    for fmt in formats:
                        fmt_indices = [i for i, f in enumerate(model_results['format_name']) if f == fmt]
                        fmt_scores = [model_results[r2_key][i] for i in fmt_indices if not np.isnan(model_results[r2_key][i])]
                        
                        if fmt_scores:
                            avg_score = np.mean(fmt_scores)
                            if is_kfold:
                                lines.append(f"  {fmt:20s}: Mean R² = {avg_score:.3f}")
                            else:
                                lines.append(f"  {fmt:20s}: Avg R² = {avg_score:.3f}")
            
            else:
                # Normal size sweep handling
                all_linear = []
                for model_results in results['models'].values():
                    all_linear.extend([x for x in model_results['linear_r2'] if not np.isnan(x)])
                
                if all_linear:
                    avg_perf = np.mean(all_linear)
                    lines.append(f"Average Linear R²: {avg_perf:.3f}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def save_sweep_report(self, filename: str = None) -> None:
        """Save report to file."""
        if filename is None:
            filename = os.path.join(self.config.results_dir, "sweep_report.txt")
        
        report = self.generate_sweep_report()
        with open(filename, "w") as f:
            f.write(report)
        
        print(f"📋 Saved report: {filename}")


def run_quick_sweep_demo(embedding_wrapper, models_to_test: List[str] = None):
    """Quick demo of sweep functionality."""
    print("🚀 Quick Sweep Demo")
    print("=" * 60)
    
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
        # Run just positive decimals and format×size sweep
        sweep.run_decimal_sweep(positive_only=True)
        
        # Run format×size sweep with smaller range for demo
        print("\n🎯 Running Format×Size Sweep (demo with sizes 1-10)...")
        sweep.run_format_size_sweep(size_range=list(range(1, 11)))
        
        # Create outputs
        sweep.plot_sweep_results('positive_decimals')
        sweep.plot_sweep_results('format_size_sweep')
        sweep.export_sweep_results()
        
        # Show report
        report = sweep.generate_sweep_report()
        print("\n" + report)
        
        print("\n✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()