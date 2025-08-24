#!/usr/bin/env python3
"""
Fixed Main Script for Embedding Reconstruction Experiments

This version is simplified and properly integrated with the cache system.
Updated with similarity experiments support.
"""

import os
import sys
import argparse
from typing import Optional

# Import our modules
from experiments import ExperimentRunner, ExperimentConfig


def get_api_keys():
    """Get API keys from environment variables."""
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'google_api_key': os.getenv('GOOGLE_API_KEY'),
        'voyage_api_key': os.getenv('VOYAGE_API_KEY')
    }


def check_api_keys(keys):
    """Check which API keys are available."""
    available = {k: v for k, v in keys.items() if v is not None}
    
    if not available:
        print("⚠️  No API keys found. Set OPENAI_API_KEY, GOOGLE_API_KEY, and/or VOYAGE_API_KEY")
        print("The experiment will run with mock data.\n")
        return False
    else:
        print("Available API services:")
        for key, value in available.items():
            service = key.replace('_api_key', '').title()
            print(f"  ✓ {service}: {value[:8]}...")
        print()
        return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run embedding reconstruction experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic options
    parser.add_argument('--samples', type=int, default=500, help='Samples per dataset')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--plot-dir', type=str, default='plots', help='Plot directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--cache-dir', type=str, default='embedding_cache', help='Cache directory')
    
    # Mode options
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer samples)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache first')
    parser.add_argument('--preload-only', action='store_true', help='Only preload cache')
    
    # Model selection
    parser.add_argument('--models', nargs='+', help='Specific models to test')
    
    # Sweep experiments
    parser.add_argument('--sweep', action='store_true', help='Run size sweep experiments')
    parser.add_argument('--sweep-demo', action='store_true', help='Quick sweep demo')
    
    # Similarity experiments
    parser.add_argument('--sweep-sims', action='store_true', help='Run similarity preservation experiments')
    parser.add_argument('--sims-demo', action='store_true', help='Quick similarity demo')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.samples = 100
        print("🚀 Quick mode: using 100 samples")
    
    print("Multi-Provider Embedding Reconstruction Experiment")
    print("=" * 60)
    
    # Check API keys
    api_keys = get_api_keys()
    has_real_keys = check_api_keys(api_keys)
    
    # Create config
    config = ExperimentConfig(
        n_samples=args.samples,
        test_size=args.test_size,
        random_state=args.random_state,
        plot_dir=args.plot_dir,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        enable_cache=not args.no_cache
    )
    
    print(f"Config: {args.samples} samples, cache={'ON' if config.enable_cache else 'OFF'}")
    
    try:
        # Initialize runner
        runner = ExperimentRunner(**api_keys, config=config)
        
        # Clear cache if requested
        if args.clear_cache:
            print("🗑️ Clearing cache...")
            runner.wrapper.cache.clear_cache()
        
        # Filter models if specified
        if args.models:
            available = set(runner.available_models)
            requested = set(args.models)
            valid = list(available & requested)
            invalid = list(requested - available)
            
            if invalid:
                print(f"⚠️  Invalid models: {invalid}")
            
            if valid:
                runner.available_models = valid
                print(f"Testing: {valid}")
            else:
                print("❌ No valid models specified")
                return 1
        
        # Handle special modes
        if args.sweep_demo:
            from sweep_experiments import run_quick_sweep_demo
            print("🚀 Running sweep demo...")
            run_quick_sweep_demo(runner.wrapper, args.models)
            return 0
        
        if args.sims_demo:
            from similarity_experiments import run_quick_similarity_demo
            print("🚀 Running similarity demo...")
            run_quick_similarity_demo(runner.wrapper, args.models)
            return 0
        
        if args.sweep:
            from sweep_experiments import NumberSizeSweep, SweepConfig
            print("🔬 Running sweep experiments...")
            
            sweep_config = SweepConfig(
                n_samples=args.samples,
                plot_dir=os.path.join(args.plot_dir, "sweep"),
                results_dir=os.path.join(args.results_dir, "sweep")
            )
            
            sweep = NumberSizeSweep(runner.wrapper, sweep_config)
            
            # Filter models
            if args.models:
                available = set(sweep.available_models)
                requested = set(args.models)
                sweep.available_models = list(available & requested)
                
                if not sweep.available_models:
                    print("❌ No valid models for sweep")
                    return 1
            
            # Run experiments
            if args.quick:
                # Just positive decimals for quick mode
                sweep.run_decimal_sweep(positive_only=True)
            else:
                # All sweep experiments
                sweep.run_all_sweeps()
            
            # Create outputs
            sweep.plot_sweep_results()
            sweep.export_sweep_results()
            sweep.save_sweep_report()
            
            # Print report
            report = sweep.generate_sweep_report()
            print("\n" + report)
            
            return 0
        
        if args.sweep_sims:
            from similarity_experiments import SimilarityExperiments, SimilarityConfig
            print("🔬 Running similarity preservation experiments...")
            
            similarity_config = SimilarityConfig(
                n_pairs=args.samples // 2,  # Use fewer pairs than main samples
                plot_dir=os.path.join(args.plot_dir, "similarity"),
                results_dir=os.path.join(args.results_dir, "similarity")
            )
            
            similarity_exp = SimilarityExperiments(runner.wrapper, similarity_config)
            
            # Filter models
            if args.models:
                available = set(similarity_exp.available_models)
                requested = set(args.models)
                similarity_exp.available_models = list(available & requested)
                
                if not similarity_exp.available_models:
                    print("❌ No valid models for similarity experiments")
                    return 1
            
            # Run similarity experiments
            if args.quick:
                # Just negative and one scaling for quick mode
                similarity_exp.run_negative_relationship_sweep()
                similarity_exp.run_scaling_relationship_sweep([2.0])
            else:
                # All similarity experiments
                similarity_exp.run_all_similarity_experiments()
            
            # Create outputs
            similarity_exp.plot_summary_results()
            similarity_exp.export_similarity_results()
            similarity_exp.save_similarity_report()
            
            # Print report
            report = similarity_exp.generate_similarity_report()
            print("\n" + report)
            
            return 0
        
        if args.preload_only:
            print("📦 Preload-only mode...")
            runner.preload_all_embeddings()
            return 0
        
        # Run main experiments
        if args.quick:
            # Quick datasets
            from experiments import create_simple_integer_dataset
            from datasets import real_positive_decimals, real_positive_and_negative_decimals
            
            quick_datasets = [
                ("Small_Decimals", lambda n: real_positive_decimals(n, 5)),
                ("Mixed_Decimals", lambda n: real_positive_and_negative_decimals(n, 10)),
                ("Simple_Integers", lambda n: create_simple_integer_dataset(n, 1000)),
            ]
            
            print(f"🚀 Quick mode: {len(quick_datasets)} datasets")
            runner.run_full_experiment(datasets=quick_datasets)
        else:
            # Complete analysis
            runner.run_complete_analysis(preload_embeddings=config.enable_cache)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nℹ️  Interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_cache():
    """Simple test to verify cache is working."""
    print("🧪 Testing Cache System")
    print("=" * 30)
    
    from api_wrapper import EmbeddingWrapper
    from embedding_cache import CachedEmbeddingWrapper
    
    # Get keys
    api_keys = get_api_keys()
    
    # Create wrappers
    base_wrapper = EmbeddingWrapper(**api_keys)
    cached_wrapper = CachedEmbeddingWrapper(base_wrapper, cache_dir="test_cache")
    
    # Test data
    test_texts = ["1.23", "4.56", "7.89"]
    
    # Get available models
    try:
        available_services = cached_wrapper.get_available_services()
        supported_models = cached_wrapper.get_supported_models()
        
        test_model = None
        for service, is_available in available_services.items():
            if is_available and service in supported_models:
                test_model = supported_models[service][0]
                break
        
        if not test_model:
            test_model = "mock-model"
            print("Using mock model for testing")
        
        print(f"Testing with model: {test_model}")
        
        # Clear cache first
        cached_wrapper.cache.clear_cache()
        
        print("\n1. First call (should hit API):")
        embeddings1 = cached_wrapper.embed(test_texts, test_model)
        
        print("2. Second call (should hit cache):")
        embeddings2 = cached_wrapper.embed(test_texts, test_model)
        
        # Check if results are identical
        if embeddings1 == embeddings2:
            print("✅ Cache test PASSED - identical embeddings returned")
        else:
            print("❌ Cache test FAILED - different embeddings returned")
        
        # Show cache stats
        stats = cached_wrapper.get_cache_stats()
        print(f"\nCache stats: {stats}")
        
        # Cleanup
        cached_wrapper.cache.clear_cache()
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        import traceback
        traceback.print_exc()


def run_custom_example():
    """Example of custom experiment."""
    print("🔬 Custom Experiment Example")
    print("=" * 40)
    
    from datasets import real_positive_decimals
    from linear_reconstruct import evaluate_linear_reconstruction
    from pca_exp import evaluate_pca_reconstruction
    
    # Get API keys
    api_keys = get_api_keys()
    
    # Create runner
    config = ExperimentConfig(n_samples=100, plot_dir="custom_plots")
    runner = ExperimentRunner(**api_keys, config=config)
    
    # Custom dataset
    texts = real_positive_decimals(100, 8)
    values = [float(x) for x in texts]
    
    print(f"Testing dataset: {len(texts)} samples")
    print(f"Value range: {min(values):.3f} to {max(values):.3f}")
    
    # Test first available model
    if runner.available_models:
        model = runner.available_models[0]
        print(f"\nTesting model: {model}")
        
        try:
            # Get embeddings
            embeddings = runner.wrapper.embed(texts, model)
            
            # Run PCA
            pca_results = evaluate_pca_reconstruction(embeddings, values)
            print(f"PCA R²: {pca_results['test_r2']:.3f}")
            
            # Run linear
            linear_results = evaluate_linear_reconstruction(embeddings, values)
            print(f"Linear R²: {linear_results['test_r2']:.3f}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No models available")


if __name__ == "__main__":
    # Check for special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "test-cache":
            test_cache()
            exit(0)
        elif sys.argv[1] == "custom":
            run_custom_example()
            exit(0)
    
    exit(main())