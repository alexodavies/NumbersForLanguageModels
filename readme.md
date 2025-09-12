# Embedding Reconstruction Experiments

A comprehensive framework for analyzing how well numerical values can be reconstructed from text embedding spaces across multiple embedding providers (OpenAI, Google Gemini, Voyage AI).

## Overview

This project explores whether embedding models implicitly encode numerical information in their vector representations. The framework tests reconstruction accuracy using various machine learning approaches and analyzes patterns across different numerical scales and types.

## Features

### Core Experiments
- **Linear Reconstruction**: Direct linear regression from embeddings to values
- **PCA Reconstruction**: Dimensionality reduction before regression
- **Cross-Model Analysis**: Compare reconstruction performance across different embedding models

### Advanced Analysis
- **Size Sweep Experiments**: Test reconstruction across different numerical magnitude ranges
- **K-Fold Cross-Validation**: Robust statistical validation with configurable fold counts
- **Similarity Preservation**: Analyze how numerical relationships are preserved in embedding space
- **Periodicity Analysis**: FFT-based frequency domain analysis with Joy Division-style visualizations

### Multi-Provider Support
- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Google Gemini**: `gemini-embedding-001`
- **Voyage AI**: `voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-code-3`, etc.

## Installation

```bash
# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn
pip install openai google-genai voyageai requests

# Clone/download the project files
```

## API Setup

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key" 
export VOYAGE_API_KEY="your-voyage-key"
```

The system works without API keys using mock data for testing and development.

## Quick Start

```bash
# Basic reconstruction experiment
python main.py

# Quick test with fewer samples
python main.py --quick

# Test specific models
python main.py --models text-embedding-3-small gemini-embedding-001

# Clear cache and run fresh
python main.py --clear-cache
```

## Experiment Types

### 1. Basic Reconstruction
```bash
python main.py                    # Full analysis
python main.py --quick           # Quick test (100 samples)
```

### 2. Size Sweep Analysis
```bash
python main.py --sweep           # Full magnitude sweep
python main.py --sweep-demo      # Quick sweep demo
python main.py --sweep --kfold 5 # With k-fold validation
```

### 3. Similarity Experiments
```bash
python main.py --sweep-sims      # Full similarity analysis
python main.py --sims-demo       # Quick similarity demo
```

### 4. Periodicity Analysis
```bash
python main.py --periodicity     # Full FFT analysis
python main.py --periodicity-demo # Quick FFT demo
```

## Configuration Options

### Basic Settings
- `--samples N`: Number of samples per dataset (default: 500)
- `--test-size F`: Test set fraction (default: 0.2)
- `--random-state N`: Random seed for reproducibility (default: 42)

### Output Control
- `--plot-dir DIR`: Plot output directory (default: 'plots')
- `--results-dir DIR`: Results output directory (default: 'results')
- `--cache-dir DIR`: Cache directory (default: 'embedding_cache')

### Cache Management
- `--no-cache`: Disable embedding caching
- `--clear-cache`: Clear cache before running
- `--preload-only`: Only preload embeddings to cache

### Model Selection
- `--models MODEL1 MODEL2`: Specify which models to test
- Available models depend on configured API keys

## Dataset Types

The framework tests various numerical representations:

- **Positive Decimals**: 0.0001 to 0.9999
- **Mixed Decimals**: -0.9999 to 0.9999  
- **Small Integers**: 1 to 1,000
- **Large Integers**: 1,000 to 1,000,000
- **Scientific Notation**: Various scales and precisions
- **Custom Ranges**: Configurable magnitude ranges

## Output Structure

```
plots/
├── basic/              # Basic reconstruction plots
├── sweep/              # Size sweep visualizations
├── similarity/         # Similarity preservation plots
└── ffts/               # FFT periodicity analysis

results/
├── basic/              # Numerical results (JSON/CSV)
├── sweep/              # Sweep experiment data
├── similarity/         # Similarity metrics
└── ffts/               # FFT analysis results
```

## Special Commands

### Cache Testing
```bash
python main.py test-cache        # Test cache functionality
```

### Custom Experiments  
```bash
python main.py custom           # Run custom experiment example
```

### Utility Functions
```bash
python periodicity.py           # Run standalone periodicity analysis
python datasets.py             # Test dataset generation
python api_wrapper.py          # Test API wrapper functionality
```

## Understanding Results

### Reconstruction Metrics
- **R² Score**: Coefficient of determination (higher = better reconstruction)
- **MSE**: Mean squared error (lower = better)
- **Correlation**: Linear relationship strength

### Visualization Types
- **Scatter Plots**: Predicted vs actual values
- **Error Distributions**: Reconstruction error analysis
- **Cross-Model Comparisons**: Performance across different embedding models
- **Joy Division Plots**: FFT frequency patterns across magnitude ranges

### Key Insights
Look for patterns in the results that indicate:
- Which numerical formats are best preserved in embeddings
- How reconstruction accuracy varies across magnitude scales
- Whether certain embedding models better preserve numerical information
- Frequency domain characteristics of numerical encoding

## Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure environment variables are set correctly
2. **Memory Issues**: Reduce `--samples` for large experiments  
3. **Cache Problems**: Use `--clear-cache` to reset
4. **Import Errors**: Check that all required packages are installed

### Performance Tips
- Use `--quick` for initial testing
- Enable caching for repeated experiments
- Use `--preload-only` to populate cache in advance
- Specify fewer models with `--models` to reduce runtime

## Advanced Usage

### K-Fold Cross-Validation
```bash
python main.py --sweep --kfold 10    # 10-fold CV for sweep
python main.py --kfold 5            # 5-fold CV for basic experiments
```

### Custom Configurations
Modify the experiment configuration by editing the relevant config classes in the source files or extending the framework with your own experiment types.

## Contributing

The modular design allows easy extension:
- Add new dataset generators in `datasets.py`
- Implement new reconstruction methods following existing patterns
- Add support for additional embedding providers in `api_wrapper.py`
- Create new experiment types following the established patterns

## License

This project is designed for research and educational purposes. Ensure compliance with the terms of service for any embedding APIs you use.