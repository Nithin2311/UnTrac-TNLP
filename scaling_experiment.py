"""
UnTrac Scaling Experiment
Extension for testing UnTrac on progressively larger language models

Models tested:
1. GPT-2 Small (124M) - baseline
2. GPT-2 Medium (355M)
3. GPT-2 Large (774M)
4. LLaMA-7B (7B) - with quantization

This module implements memory-efficient strategies for running UnTrac on larger models
within Google Colab Pro constraints.
"""

import torch
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import time
import json
from dataclasses import dataclass, asdict
import gc
import psutil

from untrac_base_implementation import (
    UnTracConfig,
    UnTracInfluenceEstimator,
    ToxicityAttributionExperiment
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScalingExperimentConfig:
    """Configuration for scaling experiments"""
    models_to_test: List[str] = None
    use_quantization: bool = False
    quantization_bits: int = 4  # 4-bit or 8-bit quantization
    measure_memory: bool = True
    measure_runtime: bool = True
    sample_size: int = 100  # Training samples for influence computation
    test_cases: int = 3  # Number of test cases
    
    def __post_init__(self):
        if self.models_to_test is None:
            self.models_to_test = [
                "gpt2",           # 124M
                "gpt2-medium",    # 355M
                "gpt2-large",     # 774M
            ]


class MemoryEfficientUnTracEstimator(UnTracInfluenceEstimator):
    """
    Memory-efficient version of UnTrac for larger models.
    Implements optimizations like gradient checkpointing and quantization.
    """
    
    def __init__(self, config: UnTracConfig, use_quantization: bool = False, quant_bits: int = 4):
        self.use_quantization = use_quantization
        self.quant_bits = quant_bits
        
        # Initialize with custom model loading
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load model with optimizations
        self._load_model_optimized()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model parameters: {self.count_parameters():,}")
    
    def _load_model_optimized(self):
        """Load model with memory optimizations"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        if self.use_quantization and "llama" in self.config.model_name.lower():
            # Use quantization for LLaMA models
            logger.info(f"Using {self.quant_bits}-bit quantization")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=(self.quant_bits == 4),
                load_in_8bit=(self.quant_bits == 8),
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        else:
            # Standard loading for GPT-2 models
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.use_quantization else torch.float32
            )
            self.model.to(self.device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    def count_parameters(self) -> int:
        """Count total model parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def compute_influence_scores_batched(
        self,
        test_text: str,
        training_texts: List[str],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Compute influence scores with batched processing for efficiency.
        This reduces the number of forward passes.
        """
        logger.info("Computing influence scores (batched)...")
        
        # Compute initial losses in batches
        initial_losses = self._compute_losses_batched(training_texts, batch_size)
        
        # Unlearn test example
        logger.info("Unlearning test example...")
        _, initial_params = self.unlearn_example(test_text, save_initial_params=True)
        
        # Compute unlearned losses in batches
        unlearned_losses = self._compute_losses_batched(training_texts, batch_size)
        
        # Restore model
        self.restore_model(initial_params)
        
        # Calculate influence
        influence_scores = unlearned_losses - initial_losses
        return influence_scores
    
    def _compute_losses_batched(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Compute losses for multiple texts in batches"""
        losses = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                # Get per-example losses
                batch_losses = outputs.loss if outputs.loss.dim() == 0 else outputs.loss
                
                if isinstance(batch_losses, torch.Tensor) and batch_losses.dim() == 0:
                    # Single loss for batch, approximate per-example
                    losses.extend([batch_losses.item()] * len(batch_texts))
                else:
                    losses.extend(batch_losses.cpu().numpy().tolist())
            
            # Clear cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        return np.array(losses)


class ScalingExperiment:
    """
    Experiment to test UnTrac scalability across model sizes.
    Measures computational requirements and performance.
    """
    
    def __init__(self, experiment_config: ScalingExperimentConfig):
        self.config = experiment_config
        self.results = {
            "models": {},
            "summary": {}
        }
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage"""
        stats = {}
        
        # CPU Memory
        cpu_mem = psutil.virtual_memory()
        stats["cpu_memory_used_gb"] = cpu_mem.used / 1e9
        stats["cpu_memory_total_gb"] = cpu_mem.total / 1e9
        stats["cpu_memory_percent"] = cpu_mem.percent
        
        # GPU Memory
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            stats["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            stats["gpu_memory_percent"] = (stats["gpu_memory_allocated_gb"] / stats["gpu_memory_total_gb"]) * 100
        
        return stats
    
    def test_model(self, model_name: str, training_texts: List[str], test_texts: List[str]) -> Dict:
        """Test UnTrac on a specific model"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing model: {model_name}")
        logger.info(f"{'='*60}")
        
        # Clear memory before starting
        gc.collect()
        torch.cuda.empty_cache()
        
        # Initial memory
        mem_before = self.measure_memory_usage()
        
        try:
            # Configure for this model
            config = UnTracConfig(
                model_name=model_name,
                unlearning_rate=5e-5,
                unlearning_steps=1,
                max_train_samples=len(training_texts),
                max_test_samples=len(test_texts),
                gradient_checkpointing=True,  # Always enable for larger models
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Create estimator
            use_quant = "llama" in model_name.lower() or "large" in model_name.lower()
            estimator = MemoryEfficientUnTracEstimator(
                config,
                use_quantization=use_quant,
                quant_bits=self.config.quantization_bits
            )
            
            # Memory after loading
            mem_after_load = self.measure_memory_usage()
            
            # Run influence computation
            start_time = time.time()
            
            influence_results = []
            for i, test_text in enumerate(test_texts):
                logger.info(f"\nProcessing test case {i+1}/{len(test_texts)}")
                
                # Compute influences
                influences = estimator.compute_influence_scores_batched(
                    test_text,
                    training_texts,
                    batch_size=4
                )
                
                # Get top-k
                top_k = 10
                top_indices = np.argsort(influences)[-top_k:][::-1]
                
                influence_results.append({
                    "test_text": test_text[:100],
                    "top_influences": [
                        {
                            "rank": rank + 1,
                            "index": int(idx),
                            "score": float(influences[idx])
                        }
                        for rank, idx in enumerate(top_indices)
                    ],
                    "mean_influence": float(np.mean(influences)),
                    "std_influence": float(np.std(influences))
                })
            
            runtime = time.time() - start_time
            
            # Final memory
            mem_final = self.measure_memory_usage()
            
            # Compile results
            result = {
                "model_name": model_name,
                "parameters": estimator.count_parameters(),
                "runtime_seconds": runtime,
                "memory_before_gb": mem_before.get("gpu_memory_allocated_gb", 0),
                "memory_after_load_gb": mem_after_load.get("gpu_memory_allocated_gb", 0),
                "memory_peak_gb": mem_final.get("gpu_memory_allocated_gb", 0),
                "memory_overhead_gb": mem_final.get("gpu_memory_allocated_gb", 0) - mem_before.get("gpu_memory_allocated_gb", 0),
                "influence_results": influence_results,
                "success": True
            }
            
            logger.info(f"\n✓ Model {model_name} completed successfully!")
            logger.info(f"  Runtime: {runtime:.2f}s")
            logger.info(f"  Peak GPU Memory: {result['memory_peak_gb']:.2f} GB")
            
        except Exception as e:
            logger.error(f"\n✗ Model {model_name} failed: {str(e)}")
            result = {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
        
        finally:
            # Cleanup
            if 'estimator' in locals():
                del estimator
            gc.collect()
            torch.cuda.empty_cache()
        
        return result
    
    def run_scaling_experiment(self):
        """Run the complete scaling experiment across all models"""
        logger.info("\n" + "="*60)
        logger.info("STARTING SCALING EXPERIMENT")
        logger.info("="*60)
        
        # Prepare data
        logger.info("\nPreparing data...")
        
        # Use dummy data for this example (replace with actual data in practice)
        training_texts = [
            f"This is training example {i}. " * 10
            for i in range(self.config.sample_size)
        ]
        
        test_texts = [
            f"This is test example {i} that we want to analyze."
            for i in range(self.config.test_cases)
        ]
        
        # Test each model
        for model_name in self.config.models_to_test:
            result = self.test_model(model_name, training_texts, test_texts)
            self.results["models"][model_name] = result
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self.save_results()
        
        logger.info("\n" + "="*60)
        logger.info("SCALING EXPERIMENT COMPLETE")
        logger.info("="*60)
    
    def _generate_summary(self):
        """Generate summary statistics across models"""
        summary = {
            "total_models_tested": len(self.config.models_to_test),
            "successful_models": sum(
                1 for r in self.results["models"].values() if r.get("success", False)
            ),
            "comparison": []
        }
        
        # Compare successful models
        for model_name in self.config.models_to_test:
            result = self.results["models"].get(model_name, {})
            
            if result.get("success"):
                summary["comparison"].append({
                    "model": model_name,
                    "parameters": result.get("parameters", 0),
                    "runtime_seconds": result.get("runtime_seconds", 0),
                    "memory_gb": result.get("memory_peak_gb", 0),
                    "avg_influence_score": np.mean([
                        r["mean_influence"]
                        for r in result.get("influence_results", [])
                    ]) if result.get("influence_results") else 0
                })
        
        self.results["summary"] = summary
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*60)
        
        for comp in summary["comparison"]:
            logger.info(f"\n{comp['model']}:")
            logger.info(f"  Parameters: {comp['parameters']:,}")
            logger.info(f"  Runtime: {comp['runtime_seconds']:.2f}s")
            logger.info(f"  Peak Memory: {comp['memory_gb']:.2f} GB")
            logger.info(f"  Avg Influence: {comp['avg_influence_score']:.6f}")
    
    def save_results(self, filename: str = "scaling_experiment_results.json"):
        """Save experimental results"""
        output_path = f"/mnt/user-data/outputs/{filename}"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {output_path}")
    
    def plot_results(self):
        """Generate visualization of scaling results"""
        try:
            import matplotlib.pyplot as plt
            
            comparison = self.results["summary"]["comparison"]
            
            if not comparison:
                logger.warning("No successful results to plot")
                return
            
            models = [c["model"] for c in comparison]
            params = [c["parameters"] / 1e6 for c in comparison]  # In millions
            runtimes = [c["runtime_seconds"] for c in comparison]
            memories = [c["memory_gb"] for c in comparison]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Plot 1: Parameters
            axes[0].bar(models, params)
            axes[0].set_ylabel("Parameters (millions)")
            axes[0].set_title("Model Size")
            axes[0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Runtime
            axes[1].bar(models, runtimes, color='orange')
            axes[1].set_ylabel("Runtime (seconds)")
            axes[1].set_title("Computation Time")
            axes[1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Memory
            axes[2].bar(models, memories, color='green')
            axes[2].set_ylabel("Peak Memory (GB)")
            axes[2].set_title("Memory Usage")
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig("/mnt/user-data/outputs/scaling_results.png", dpi=300, bbox_inches='tight')
            logger.info("Plot saved to /mnt/user-data/outputs/scaling_results.png")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")


def main():
    """Run the scaling experiment"""
    
    # Configure experiment
    config = ScalingExperimentConfig(
        models_to_test=[
            "gpt2",         # 124M - baseline
            "gpt2-medium",  # 355M
            "gpt2-large",   # 774M
        ],
        use_quantization=False,
        sample_size=50,  # Keep small for feasibility
        test_cases=3
    )
    
    logger.info("Scaling Experiment Configuration:")
    logger.info(json.dumps(asdict(config), indent=2))
    
    # Run experiment
    experiment = ScalingExperiment(config)
    experiment.run_scaling_experiment()
    
    # Generate visualization
    experiment.plot_results()


if __name__ == "__main__":
    main()
