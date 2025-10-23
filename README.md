# UnTrac-TNLP

# UnTrac Reproducibility Study

Implementation and extension of "Unlearning Traces the Influential Training Data of Language Models" (Isonuma & Titov, ACL 2024)

## Project Overview

This project aims to:
1. **Reproduce** the core experiments from the original UnTrac paper
2. **Extend** the work by testing scalability on larger language models
3. **Analyze** computational requirements and performance trade-offs

### What is UnTrac?

UnTrac is a novel method for identifying which training examples influence a model's specific behaviors (like toxicity or bias). Instead of traditional influence estimation methods, UnTrac:
- Takes a test example showing undesirable behavior
- "Unlearns" this example using gradient ascent
- Measures which training examples are most affected by this unlearning

## Repository Structure

```
├── untrac_base_implementation.py   # Core UnTrac implementation
├── scaling_experiment.py           # Scaling tests for larger models
├── colab_setup_notebook.py        # Google Colab setup guide
├── README.md                       # This file
└── results/                        # Output directory
    ├── untrac_toxicity_results.json
    └── scaling_experiment_results.json
```

## Quick Start (Google Colab)

### 1. Setup Environment

```python
# Install dependencies
!pip install -q transformers datasets torch accelerate

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 2. Upload Implementation Files

```python
from google.colab import files
uploaded = files.upload()  # Upload the Python files
```

### 3. Run Base Experiment (GPT-2 Small)

```python
from untrac_base_implementation import UnTracConfig, ToxicityAttributionExperiment
import json
from dataclasses import asdict

# Configure
config = UnTracConfig(
    model_name="gpt2",
    unlearning_rate=5e-5,
    unlearning_steps=1,
    max_train_samples=500,
    max_test_samples=5,
    top_k_influences=10
)

# Run experiment
experiment = ToxicityAttributionExperiment(config)
experiment.run_experiment()
```

### 4. Run Scaling Experiment

```python
from scaling_experiment import ScalingExperiment, ScalingExperimentConfig

# Configure scaling test
config = ScalingExperimentConfig(
    models_to_test=["gpt2", "gpt2-medium", "gpt2-large"],
    sample_size=100,
    test_cases=3
)

# Run
experiment = ScalingExperiment(config)
experiment.run_scaling_experiment()
experiment.plot_results()
```

## Detailed Usage

### Configuration Options

#### UnTracConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"gpt2"` | HuggingFace model identifier |
| `unlearning_rate` | `5e-5` | Learning rate for gradient ascent |
| `unlearning_steps` | `1` | Number of unlearning iterations |
| `batch_size` | `1` | Batch size for processing |
| `max_train_samples` | `10000` | Max training examples to use |
| `max_test_samples` | `100` | Max test cases to evaluate |
| `top_k_influences` | `50` | Top-k influential examples to retrieve |
| `gradient_checkpointing` | `False` | Enable for memory efficiency |

#### Example Configurations

**Small-scale testing:**
```python
config = UnTracConfig(
    model_name="gpt2",
    max_train_samples=100,
    max_test_samples=5,
    gradient_checkpointing=False
)
```

**Large-scale with memory optimization:**
```python
config = UnTracConfig(
    model_name="gpt2-large",
    max_train_samples=1000,
    max_test_samples=10,
    gradient_checkpointing=True,  # Important for large models!
    batch_size=4
)
```

### Using the UnTrac Estimator Directly

```python
from untrac_base_implementation import UnTracInfluenceEstimator, UnTracConfig

# Initialize
config = UnTracConfig(model_name="gpt2")
estimator = UnTracInfluenceEstimator(config)

# Define your data
test_text = "This is toxic language that we want to trace."
training_texts = [
    "Training example 1...",
    "Training example 2...",
    # ... more training examples
]

# Compute influences
influence_scores = estimator.compute_influence_scores(
    test_text,
    training_texts,
    show_progress=True
)

# Get top influences
top_k = estimator.get_top_k_influences(
    test_text,
    training_texts,
    k=10
)

# Print results
for rank, (idx, score, text) in enumerate(top_k, 1):
    print(f"{rank}. Score: {score:.6f}")
    print(f"   Text: {text[:100]}...")
```

## Scaling Experiments

### Tested Models

| Model | Parameters | Status | Notes |
|-------|-----------|--------|-------|
| GPT-2 Small | 124M | ✓ Tested | Baseline |
| GPT-2 Medium | 355M | ✓ Tested | Requires gradient checkpointing |
| GPT-2 Large | 774M | ✓ Tested | Memory intensive |
| LLaMA-7B | 7B | ⚠️ Partial | Requires 4-bit quantization |

### Memory Requirements

Approximate GPU memory usage:

- **GPT-2 Small**: ~2 GB
- **GPT-2 Medium**: ~6 GB
- **GPT-2 Large**: ~12 GB
- **LLaMA-7B (quantized)**: ~8-10 GB

### Optimization Strategies

For larger models, use these techniques:

```python
# 1. Gradient Checkpointing
config.gradient_checkpointing = True

# 2. Reduce Sample Sizes
config.max_train_samples = 200
config.max_test_samples = 3

# 3. Use Batched Processing
from scaling_experiment import MemoryEfficientUnTracEstimator

estimator = MemoryEfficientUnTracEstimator(
    config,
    use_quantization=True,
    quant_bits=4
)

influences = estimator.compute_influence_scores_batched(
    test_text,
    training_texts,
    batch_size=8
)
```

## Results Analysis

### Loading Results

```python
import json

# Load results
with open('untrac_toxicity_results.json', 'r') as f:
    results = json.load(f)

# Analyze
for test_case in results['test_cases']:
    print(f"Prompt: {test_case['prompt']}")
    print(f"Top influence score: {test_case['top_influences'][0]['score']:.6f}")
```

### Comparing Models

```python
from scaling_experiment import ScalingExperiment

# Load scaling results
with open('scaling_experiment_results.json', 'r') as f:
    scaling_results = json.load(f)

# Print comparison
for model_info in scaling_results['summary']['comparison']:
    print(f"{model_info['model']}:")
    print(f"  Runtime: {model_info['runtime_seconds']:.2f}s")
    print(f"  Memory: {model_info['memory_gb']:.2f} GB")
```

## Baseline Comparisons

### TracIn Baseline

```python
from untrac_base_implementation import TracInBaseline

baseline = TracInBaseline(model, tokenizer, device)
tracin_scores = baseline.compute_influence_scores(test_text, training_texts)

# Compare with UnTrac
import numpy as np
correlation = np.corrcoef(untrac_scores, tracin_scores)[0, 1]
print(f"Correlation with TracIn: {correlation:.4f}")
```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. **Enable gradient checkpointing:**
   ```python
   config.gradient_checkpointing = True
   ```

2. **Reduce sample sizes:**
   ```python
   config.max_train_samples = 100
   config.max_test_samples = 3
   ```

3. **Use mixed precision:**
   ```python
   model = model.half()  # Use FP16
   ```

4. **Clear cache regularly:**
   ```python
   import gc
   torch.cuda.empty_cache()
   gc.collect()
   ```

### Model Loading Issues

For LLaMA models, ensure you have access:

```python
from huggingface_hub import login
login("your_hf_token")
```

### Slow Performance

To speed up experiments:

1. Reduce training sample size
2. Use batched processing
3. Enable gradient checkpointing (trades compute for memory)

## Expected Runtimes

On Google Colab Pro (V100/A100 GPU):

| Configuration | Runtime |
|--------------|---------|
| GPT-2 Small, 100 train samples, 5 test cases | ~5 minutes |
| GPT-2 Medium, 200 train samples, 3 test cases | ~15 minutes |
| GPT-2 Large, 100 train samples, 3 test cases | ~30 minutes |

## Reproducing Paper Results

To reproduce Table 1 (Toxicity Attribution) from the original paper:

```python
config = UnTracConfig(
    model_name="gpt2",
    unlearning_rate=5e-5,
    max_train_samples=10000,  # Paper uses full dataset
    max_test_samples=100,
    top_k_influences=50
)

experiment = ToxicityAttributionExperiment(config)
experiment.run_experiment()
```

**Note:** Full reproduction requires significant computational resources and time.

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{isonuma2024unlearning,
  title={Unlearning Traces the Influential Training Data of Language Models},
  author={Isonuma, Masaru and Titov, Ivan},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  year={2024}
}
```

## Project Timeline

- **Week 10-11**: Setup and implementation
- **Week 11-13**: Base experiment reproduction
- **Week 13-14**: Scaling experiments
- **Week 14-15**: Analysis and report writing
- **Week 16**: Final polishing

## Contact

For questions or issues, please contact:
- palyam@usf.edu
- indukuri3@usf.edu

## License

This code is provided for educational and research purposes. Please refer to the original paper's license for any commercial use.

## Acknowledgments

- Original paper authors: Masaru Isonuma and Ivan Titov
- HuggingFace for model hosting
- Google Colab for computational resources
