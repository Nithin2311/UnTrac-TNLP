# UnTrac Reproducibility Study - Google Colab Setup
# ================================================

"""
This notebook sets up the environment for running UnTrac experiments on Google Colab.
It includes installation of dependencies, configuration, and execution of base experiments.
"""

# Cell 1: Initial Setup and Installations
print("Installing required packages...")
!pip install -q transformers datasets torch accelerate sentencepiece protobuf
!pip install -q google-api-python-client  # For PerspectiveAPI (optional)

# Cell 2: Check GPU Availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Cell 3: Mount Google Drive (optional - for saving results)
from google.colab import drive
drive.mount('/content/drive')

# Cell 4: Clone or Upload Implementation Code
# Option 1: Upload the implementation file you received
from google.colab import files
uploaded = files.upload()  # Upload untrac_base_implementation.py

# Option 2: Or create it directly
%%writefile untrac_base_implementation.py
# [Paste the entire implementation code here]

# Cell 5: Import and Configure
import sys
import os
import json
import logging
from dataclasses import asdict

# Add current directory to path
sys.path.insert(0, '/content')

from untrac_base_implementation import (
    UnTracConfig,
    UnTracInfluenceEstimator,
    ToxicityAttributionExperiment
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell 6: Run Base Experiment (GPT-2 Small)
print("="*60)
print("RUNNING BASE EXPERIMENT: GPT-2 SMALL")
print("="*60)

config_small = UnTracConfig(
    model_name="gpt2",
    unlearning_rate=5e-5,
    unlearning_steps=1,
    max_train_samples=500,  # Start small for testing
    max_test_samples=5,
    top_k_influences=10,
    gradient_checkpointing=False,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print("\nConfiguration:")
print(json.dumps(asdict(config_small), indent=2))

# Run experiment
experiment_small = ToxicityAttributionExperiment(config_small)
experiment_small.run_experiment()

# Cell 7: Analyze Results
import json

# Load results
with open('/mnt/user-data/outputs/untrac_toxicity_results.json', 'r') as f:
    results = json.load(f)

print(f"\nTotal test cases: {len(results['test_cases'])}")

# Display first result
if results['test_cases']:
    first_result = results['test_cases'][0]
    print("\n" + "="*60)
    print("SAMPLE RESULT")
    print("="*60)
    print(f"\nPrompt: {first_result['prompt']}")
    print(f"\nCompletion: {first_result['completion']}")
    print("\nTop 5 Influential Training Examples:")
    for inf in first_result['top_influences'][:5]:
        print(f"\nRank {inf['rank']}: Score = {inf['score']:.6f}")
        print(f"Text: {inf['text'][:150]}...")

# Cell 8: Run Scaling Experiment (GPT-2 Medium)
print("\n" + "="*60)
print("RUNNING SCALING EXPERIMENT: GPT-2 MEDIUM")
print("="*60)

# Clear memory
import gc
torch.cuda.empty_cache()
gc.collect()

config_medium = UnTracConfig(
    model_name="gpt2-medium",  # 355M parameters
    unlearning_rate=5e-5,
    unlearning_steps=1,
    max_train_samples=200,  # Reduce for memory constraints
    max_test_samples=3,
    top_k_influences=10,
    gradient_checkpointing=True,  # Enable for memory efficiency
    device="cuda" if torch.cuda.is_available() else "cpu"
)

experiment_medium = ToxicityAttributionExperiment(config_medium)
experiment_medium.run_experiment()

# Cell 9: Memory Usage Analysis
import psutil
import GPUtil

def print_memory_stats():
    """Print current memory usage statistics"""
    # CPU Memory
    cpu_mem = psutil.virtual_memory()
    print(f"CPU Memory: {cpu_mem.used / 1e9:.2f} GB / {cpu_mem.total / 1e9:.2f} GB")
    print(f"CPU Memory %: {cpu_mem.percent}%")
    
    # GPU Memory
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"\nGPU Memory Allocated: {gpu_mem_allocated:.2f} GB")
        print(f"GPU Memory Reserved: {gpu_mem_reserved:.2f} GB")
        print(f"GPU Memory Total: {gpu_mem_total:.2f} GB")
        print(f"GPU Memory %: {(gpu_mem_allocated / gpu_mem_total) * 100:.1f}%")

print("Memory Usage After Experiments:")
print_memory_stats()

# Cell 10: Compare Results Across Model Sizes
def compare_model_results(results_small, results_medium):
    """Compare influence scores across different model sizes"""
    
    print("\n" + "="*60)
    print("COMPARISON: GPT-2 SMALL vs MEDIUM")
    print("="*60)
    
    # Compare influence score distributions
    scores_small = []
    scores_medium = []
    
    for case in results_small['test_cases']:
        scores_small.extend([inf['score'] for inf in case['top_influences']])
    
    for case in results_medium['test_cases']:
        scores_medium.extend([inf['score'] for inf in case['top_influences']])
    
    import numpy as np
    
    print(f"\nGPT-2 Small Influence Scores:")
    print(f"  Mean: {np.mean(scores_small):.6f}")
    print(f"  Std: {np.std(scores_small):.6f}")
    print(f"  Min: {np.min(scores_small):.6f}")
    print(f"  Max: {np.max(scores_small):.6f}")
    
    print(f"\nGPT-2 Medium Influence Scores:")
    print(f"  Mean: {np.mean(scores_medium):.6f}")
    print(f"  Std: {np.std(scores_medium):.6f}")
    print(f"  Min: {np.min(scores_medium):.6f}")
    print(f"  Max: {np.max(scores_medium):.6f}")

# Load and compare results
with open('/mnt/user-data/outputs/untrac_toxicity_results.json', 'r') as f:
    results_comparison = json.load(f)

# You would need to save separate result files for each model
# compare_model_results(results_small, results_medium)

# Cell 11: Save Results to Google Drive
import shutil
from datetime import datetime

# Create timestamped folder in Drive
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'/content/drive/MyDrive/untrac_results_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

# Copy results
shutil.copy(
    '/mnt/user-data/outputs/untrac_toxicity_results.json',
    f'{output_dir}/results.json'
)

print(f"Results saved to: {output_dir}")

# Cell 12: Download Results Locally
from google.colab import files

# Download results file
files.download('/mnt/user-data/outputs/untrac_toxicity_results.json')

print("Results downloaded to your local machine!")
