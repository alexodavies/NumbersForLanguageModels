"""
Periodicity Analysis for Embedding Reconstruction

This module performs FFT analysis on embeddings of integers across different 
magnitude ranges to identify periodic patterns. Results are visualized as 
Joy Division-style offset plots showing frequency domain characteristics.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
import json
from pathlib import Path

# Import existing modules
from datasets import real_int_and_decimal
from api_wrapper import EmbeddingWrapper
from embedding_cache import CachedEmbeddingWrapper


@dataclass
class PeriodicityConfig:
    """Configuration for periodicity experiments."""
    n_samples: int = 500
    magnitude_ranges: List[Tuple[int, int]] = None  # [(10, 100), (100, 1000), ...]
    n_components_to_analyze: Optional[int] = None  # None = all components
    plot_dir: str = "plots/ffts"
    results_dir: str = "results/ffts"
    random_state: int = 42
    
    def __post_init__(self):
        if self.magnitude_ranges is None:
            # Default magnitude ranges: 10-100, 100-1K, 1K-10K, 10K-100K
            self.magnitude_ranges = [
                (10, 100),
                (100, 1000), 
                (1000, 10000),
                (10000, 100000)
            ]


class PeriodicityAnalyzer:
    """Analyzer for periodicity patterns in embeddings."""
    
    def __init__(self, wrapper: EmbeddingWrapper, config: PeriodicityConfig):
        self.wrapper = wrapper
        self.config = config
        self.results = {}
        self.available_models = self._get_available_models()
        
        # Create output directories
        os.makedirs(config.plot_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models from wrapper."""
        available_services = self.wrapper.get_available_services()
        supported_models = self.wrapper.get_supported_models()
        
        available_models = []
        for service, is_available in available_services.items():
            if is_available and service in supported_models:
                available_models.extend(supported_models[service])
        
        # Fallback for mock testing
        if not available_models:
            all_models = []
            for models in supported_models.values():
                all_models.extend(models)
            available_models = all_models
        
        return available_models
    
    def generate_magnitude_datasets(self) -> Dict[str, Tuple[List[str], List[float]]]:
        """Generate integer datasets for different magnitude ranges."""
        datasets = {}
        
        for min_mag, max_mag in self.config.magnitude_ranges:
            # Calculate digits needed
            min_digits = len(str(min_mag))
            max_digits = len(str(max_mag))
            
            # Use average for consistent digit count within range
            avg_digits = (min_digits + max_digits) // 2
            
            # Generate integers in range using the dataset function
            # We'll generate with fixed digits then filter to range
            texts = real_int_and_decimal(
                self.config.n_samples,  # Generate extra to ensure we have enough in range
                digits_before=max_digits, 
                digits_after=0,  # No decimal places - integers only
                random_state=self.config.random_state
            )
            
            # Convert to integers and filter by magnitude
            valid_texts = []
            values = []
            
            for text in texts:
                try:
                    value = int(float(text))  # Handle potential negative signs
                    abs_value = abs(value)

                    valid_texts.append(str(value))  # Store as string for embedding
                    values.append(float(value))  # Store as float for analysis

                except ValueError:
                    continue

            
            range_name = f"{min_mag}-{max_mag}"
            datasets[range_name] = (valid_texts, values)
            print(f"Generated {len(valid_texts)} samples for range {range_name}")
        
        return datasets
    
    def compute_embedding_fft(self, embeddings: np.ndarray, values: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute FFT analysis for embeddings."""
        n_samples = len(embeddings)
        embedding_dim = embeddings.shape[1]
        
        # Sort by value to ensure proper sequence for FFT
        sort_indices = np.argsort(values)
        sorted_embeddings = embeddings[sort_indices]
        sorted_values = values[sort_indices]
        
        # Compute FFT for each embedding dimension
        ffts = []
        for dim in range(embedding_dim):
            dim_values = sorted_embeddings[:, dim]
            
            # Apply window function to reduce spectral leakage
            window = np.hanning(len(dim_values))
            windowed_values = dim_values * window
            
            # Compute FFT
            fft_result = fft(windowed_values)
            ffts.append(np.abs(fft_result))
        
        ffts = np.array(ffts)  # Shape: (embedding_dim, n_samples)
        
        # Average across all embedding dimensions
        avg_fft = np.mean(ffts, axis=0)
        
        # Compute frequencies
        freqs = fftfreq(n_samples, d=1.0)
        
        # Only keep positive frequencies (real signal)
        n_pos = n_samples // 2
        pos_freqs = freqs[:n_pos]
        avg_fft_pos = avg_fft[:n_pos]
        individual_ffts_pos = ffts[:, :n_pos]
        
        return {
            'frequencies': pos_freqs,
            'avg_fft': avg_fft_pos,
            'individual_ffts': individual_ffts_pos,
            'sorted_values': sorted_values,
            'n_components': embedding_dim
        }
    
    def run_periodicity_analysis(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run FFT analysis for all models and magnitude ranges."""
        if models is None:
            models = self.available_models
        
        print("Running Periodicity Analysis")
        print("=" * 50)
        
        # Generate datasets
        print("Generating magnitude-based datasets...")
        datasets = self.generate_magnitude_datasets()
        
        results = {}
        
        for model in models:
            print(f"\nAnalyzing model: {model}")
            model_results = {}
            
            for range_name, (texts, values) in datasets.items():
                print(f"  Processing range {range_name}...")
                
                try:
                    # Get embeddings
                    embeddings = self.wrapper.embed(texts, model)
                    embeddings = np.array(embeddings)
                    values = np.array(values)
                    
                    # Limit components if specified
                    if (self.config.n_components_to_analyze is not None and 
                        embeddings.shape[1] > self.config.n_components_to_analyze):
                        embeddings = embeddings[:, :self.config.n_components_to_analyze]
                    
                    # Compute FFT analysis
                    fft_results = self.compute_embedding_fft(embeddings, values)
                    
                    model_results[range_name] = fft_results
                    
                    print(f"    FFT computed: {fft_results['n_components']} components, "
                          f"{len(fft_results['frequencies'])} frequency bins")
                    
                except Exception as e:
                    print(f"    Error processing {range_name}: {e}")
                    continue
            
            if model_results:
                results[model] = model_results
        
        self.results = results
        return results
    
    def create_joy_division_plot(self, model: str, save_path: str):
        """Create Joy Division-style FFT plot for a model."""
        if model not in self.results:
            print(f"No results found for model {model}")
            return
        
        model_results = self.results[model]
        n_ranges = len(model_results)
        
        if n_ranges == 0:
            print(f"No magnitude ranges analyzed for model {model}")
            return
        
        # Create figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor('black')
        
        # Plot each magnitude range with vertical offset
        offset_scale = 0.
        colors = plt.cm.viridis(np.linspace(0, 1, n_ranges))
        
        range_names = list(model_results.keys())
        
        for i, range_name in enumerate(range_names):
            fft_data = model_results[range_name]
            freqs = fft_data['frequencies']
            avg_fft = fft_data['avg_fft']
            
            # Normalize FFT magnitude
            if np.max(avg_fft) > 0:
                normalized_fft = avg_fft / np.max(avg_fft)
            else:
                normalized_fft = avg_fft
            
            # Apply vertical offset
            y_offset = i * offset_scale
            y_values = normalized_fft + y_offset
            
            # Plot with fill
            ax.plot(freqs, y_values, 
                           color=colors[i])
            
            # Add outline
            # ax.plot(freqs, y_values, color='white', linewidth=0.5, alpha=0.9)
            
            # Label each range
            ax.text(0.02, y_offset + 0.2, range_name, 
                   color='white', fontsize=10, fontweight='bold')
        
        # Styling
        ax.set_yscale("log")

        ax.set_xlim(0, np.max(freqs))
        ax.set_ylim(-0.2, n_ranges * offset_scale + 0.5)
        ax.set_xlabel('Frequency', color='white', fontsize=12)
        ax.set_ylabel('Magnitude Range', color='white', fontsize=12)
        ax.set_title(f'FFT Analysis: {model}\n(Joy Division Style)', 
                    color='white', fontsize=14, fontweight='bold', pad=20)
        
        # Remove ticks and spines for cleaner look
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', colors='white')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        print(f"Joy Division plot saved: {save_path}")
    
    def plot_all_models(self):
        """Create Joy Division plots for all analyzed models."""
        print("\nCreating Joy Division-style FFT plots...")
        
        for model in self.results.keys():
            # Clean model name for filename
            clean_name = model.replace('/', '_').replace('-', '_')
            filename = f"fft_joy_division_{clean_name}.png"
            save_path = os.path.join(self.config.plot_dir, filename)
            
            self.create_joy_division_plot(model, save_path)
    
    def export_results(self):
        """Export numerical results to JSON."""
        export_data = {}
        
        for model, model_results in self.results.items():
            export_data[model] = {}
            
            for range_name, fft_data in model_results.items():
                export_data[model][range_name] = {
                    'frequencies': fft_data['frequencies'].tolist(),
                    'avg_fft': fft_data['avg_fft'].tolist(),
                    'n_components': int(fft_data['n_components']),
                    'n_samples': len(fft_data['sorted_values']),
                    'value_range': {
                        'min': float(np.min(fft_data['sorted_values'])),
                        'max': float(np.max(fft_data['sorted_values'])),
                        'mean': float(np.mean(fft_data['sorted_values']))
                    }
                }
        
        # Save to JSON
        results_file = os.path.join(self.config.results_dir, "periodicity_results.json")
        with open(results_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to: {results_file}")
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of the periodicity analysis."""
        if not self.results:
            return "No results available for summary."
        
        report = ["Periodicity Analysis Summary", "=" * 40, ""]
        
        report.append(f"Configuration:")
        report.append(f"  Samples per range: {self.config.n_samples}")
        report.append(f"  Magnitude ranges: {self.config.magnitude_ranges}")
        report.append(f"  Models analyzed: {len(self.results)}")
        report.append("")
        
        for model, model_results in self.results.items():
            report.append(f"Model: {model}")
            report.append("-" * 30)
            
            for range_name, fft_data in model_results.items():
                # Find peak frequency
                peak_idx = np.argmax(fft_data['avg_fft'])
                peak_freq = fft_data['frequencies'][peak_idx]
                peak_magnitude = fft_data['avg_fft'][peak_idx]
                
                # Calculate spectral statistics
                total_power = np.sum(fft_data['avg_fft'] ** 2)
                freq_centroid = np.sum(fft_data['frequencies'] * fft_data['avg_fft']) / np.sum(fft_data['avg_fft'])
                
                report.append(f"  Range {range_name}:")
                report.append(f"    Peak frequency: {peak_freq:.4f}")
                report.append(f"    Peak magnitude: {peak_magnitude:.4f}")
                report.append(f"    Spectral centroid: {freq_centroid:.4f}")
                report.append(f"    Total power: {total_power:.4f}")
                report.append(f"    Embedding dims: {fft_data['n_components']}")
                report.append("")
            
            report.append("")
        
        return "\n".join(report)


def run_periodicity_demo(wrapper: EmbeddingWrapper, models: Optional[List[str]] = None):
    """Run a quick periodicity analysis demo."""
    print("Running Periodicity Analysis Demo")
    print("=" * 40)
    
    # Quick demo configuration
    config = PeriodicityConfig(
        n_samples=100,
        magnitude_ranges=[
            (10, 100),
            (100, 1000),
            (1000, 10000)
        ],
        n_components_to_analyze=50,  # Limit for faster processing
        plot_dir="plots/ffts/demo",
        results_dir="results/ffts/demo"
    )
    
    analyzer = PeriodicityAnalyzer(wrapper, config)
    
    # Use specified models or first available
    if models:
        test_models = [m for m in models if m in analyzer.available_models]
    else:
        test_models = analyzer.available_models[:2]  # First two models
    
    if not test_models:
        test_models = analyzer.available_models[:1]  # At least one model
    
    print(f"Testing models: {test_models}")
    
    # Run analysis
    results = analyzer.run_periodicity_analysis(test_models)
    
    # Create plots
    analyzer.plot_all_models()
    
    # Export results
    analyzer.export_results()
    
    # Print summary
    summary = analyzer.generate_summary_report()
    print("\n" + summary)
    
    return analyzer


def run_full_periodicity_analysis(wrapper: EmbeddingWrapper, 
                                config: Optional[PeriodicityConfig] = None,
                                models: Optional[List[str]] = None):
    """Run complete periodicity analysis."""
    if config is None:
        config = PeriodicityConfig()
    
    analyzer = PeriodicityAnalyzer(wrapper, config)
    
    # Run analysis
    results = analyzer.run_periodicity_analysis(models)
    
    # Create visualizations
    analyzer.plot_all_models()
    
    # Export results
    analyzer.export_results()
    
    # Generate and save report
    summary = analyzer.generate_summary_report()
    
    report_file = os.path.join(config.results_dir, "periodicity_report.txt")
    with open(report_file, 'w') as f:
        f.write(summary)
    
    print(f"\nReport saved to: {report_file}")
    print("\n" + summary)
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    from api_wrapper import EmbeddingWrapper
    from embedding_cache import CachedEmbeddingWrapper
    
    # Setup wrapper (use default cache dir to match other experiments)
    wrapper = EmbeddingWrapper()
    cached_wrapper = CachedEmbeddingWrapper(wrapper, cache_dir="embedding_cache")
    
    # Run demo
    analyzer = run_periodicity_demo(cached_wrapper)