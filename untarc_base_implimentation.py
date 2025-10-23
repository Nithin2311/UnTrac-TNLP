"""
UnTrac Base Implementation for Reproducibility Study
Based on: "Unlearning Traces the Influential Training Data of Language Models"
Isonuma & Titov (ACL 2024)

This implementation focuses on the toxicity attribution experiment with GPT-2-small.
Designed to run on Google Colab with GPU support.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnTracConfig:
    """Configuration for UnTrac experiments"""
    model_name: str = "gpt2"  # Start with GPT-2 small (124M params)
    unlearning_rate: float = 5e-5
    unlearning_steps: int = 1  # Number of gradient ascent steps
    batch_size: int = 1
    max_train_samples: int = 10000  # Limit training data for feasibility
    max_test_samples: int = 100
    top_k_influences: int = 50  # Top-k influential examples to retrieve
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    gradient_checkpointing: bool = False  # Enable for larger models


class UnTracInfluenceEstimator:
    """
    Implementation of UnTrac method for influence estimation.
    
    UnTrac works by:
    1. Taking a test example that exhibits undesirable behavior
    2. "Unlearning" this test example via gradient ascent
    3. Measuring which training examples are most affected by this unlearning
    """
    
    def __init__(self, config: UnTracConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(config.model_name)
        
        # Set pad token (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.to(self.device)
        
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def compute_loss(self, text: str) -> float:
        """
        Compute the language modeling loss for a given text.
        
        Args:
            text: Input text string
            
        Returns:
            Loss value as a float
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        
        return loss
    
    def unlearn_example(
        self,
        test_text: str,
        save_initial_params: bool = True
    ) -> Tuple[torch.nn.Module, Optional[Dict]]:
        """
        Unlearn a test example using gradient ascent.
        
        Args:
            test_text: The test example to unlearn
            save_initial_params: Whether to save initial parameters for restoration
            
        Returns:
            Tuple of (unlearned_model, initial_params_dict or None)
        """
        # Save initial parameters if requested
        initial_params = None
        if save_initial_params:
            initial_params = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
            }
        
        # Prepare input
        inputs = self.tokenizer(
            test_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Set model to training mode
        self.model.train()
        
        # Perform gradient ascent (maximizing loss to "unlearn")
        for step in range(self.config.unlearning_steps):
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass - note the negative sign for gradient ASCENT
            (-loss).backward()
            
            # Update parameters with gradient ascent
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data += self.config.unlearning_rate * param.grad
        
        # Set back to eval mode
        self.model.eval()
        
        return self.model, initial_params
    
    def restore_model(self, initial_params: Dict):
        """Restore model to initial parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(initial_params[name])
    
    def compute_influence_scores(
        self,
        test_text: str,
        training_texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Compute UnTrac influence scores for training examples.
        
        The influence score is: I(z_i) = L(z_i, θ') - L(z_i, θ)
        where θ is the original model and θ' is the unlearned model.
        
        Args:
            test_text: Test example to unlearn
            training_texts: List of training examples
            show_progress: Whether to show progress bar
            
        Returns:
            Array of influence scores (higher = more influential)
        """
        logger.info("Computing influence scores...")
        
        # Step 1: Compute initial losses on training data
        initial_losses = []
        iterator = tqdm(training_texts, desc="Computing initial losses") if show_progress else training_texts
        
        for train_text in iterator:
            loss = self.compute_loss(train_text)
            initial_losses.append(loss)
        
        initial_losses = np.array(initial_losses)
        
        # Step 2: Unlearn the test example
        logger.info(f"Unlearning test example...")
        _, initial_params = self.unlearn_example(test_text, save_initial_params=True)
        
        # Step 3: Compute losses after unlearning
        unlearned_losses = []
        iterator = tqdm(training_texts, desc="Computing unlearned losses") if show_progress else training_texts
        
        for train_text in iterator:
            loss = self.compute_loss(train_text)
            unlearned_losses.append(loss)
        
        unlearned_losses = np.array(unlearned_losses)
        
        # Step 4: Restore model
        logger.info("Restoring model...")
        self.restore_model(initial_params)
        
        # Step 5: Calculate influence scores
        influence_scores = unlearned_losses - initial_losses
        
        logger.info(f"Influence scores computed. Mean: {influence_scores.mean():.4f}, "
                   f"Std: {influence_scores.std():.4f}")
        
        return influence_scores
    
    def get_top_k_influences(
        self,
        test_text: str,
        training_texts: List[str],
        k: Optional[int] = None
    ) -> List[Tuple[int, float, str]]:
        """
        Get top-k most influential training examples for a test case.
        
        Args:
            test_text: Test example
            training_texts: Training examples
            k: Number of top influences to return (default: config.top_k_influences)
            
        Returns:
            List of tuples (index, influence_score, text) sorted by influence
        """
        if k is None:
            k = self.config.top_k_influences
        
        # Compute influence scores
        influence_scores = self.compute_influence_scores(
            test_text,
            training_texts,
            show_progress=True
        )
        
        # Get top-k indices
        top_k_indices = np.argsort(influence_scores)[-k:][::-1]
        
        # Create results
        results = [
            (int(idx), float(influence_scores[idx]), training_texts[idx])
            for idx in top_k_indices
        ]
        
        return results


class ToxicityAttributionExperiment:
    """
    Experiment for attributing toxic language generations to training data.
    This reproduces the main experiment from the paper.
    """
    
    def __init__(self, config: UnTracConfig):
        self.config = config
        self.estimator = UnTracInfluenceEstimator(config)
        self.results = defaultdict(list)
    
    def load_training_data(self) -> List[str]:
        """
        Load training data (using OpenWebText as proxy).
        In practice, this would be the actual training corpus.
        """
        logger.info("Loading training data...")
        
        try:
            # Load OpenWebText subset as training data proxy
            dataset = load_dataset("openwebtext", split="train", streaming=True)
            
            training_texts = []
            for i, example in enumerate(dataset):
                if i >= self.config.max_train_samples:
                    break
                
                text = example["text"].strip()
                if len(text) > 50:  # Filter very short texts
                    training_texts.append(text[:1000])  # Truncate long texts
            
            logger.info(f"Loaded {len(training_texts)} training examples")
            return training_texts
            
        except Exception as e:
            logger.error(f"Error loading OpenWebText: {e}")
            logger.info("Using dummy training data for testing...")
            # Fallback to dummy data for testing
            return [f"This is training example number {i}. " * 10 for i in range(100)]
    
    def load_test_data_toxic(self) -> List[Dict[str, str]]:
        """
        Load test data with toxic generations.
        Uses RealToxicityPrompts dataset.
        """
        logger.info("Loading toxic test data...")
        
        try:
            # Load RealToxicityPrompts
            dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
            
            test_cases = []
            for i, example in enumerate(dataset):
                if i >= self.config.max_test_samples:
                    break
                
                # Get prompt and continuation if available
                prompt = example["prompt"]["text"]
                
                # For this experiment, we'll generate completions
                test_cases.append({
                    "prompt": prompt,
                    "type": "toxic_prompt"
                })
            
            logger.info(f"Loaded {len(test_cases)} test cases")
            return test_cases
            
        except Exception as e:
            logger.error(f"Error loading RealToxicityPrompts: {e}")
            # Fallback to dummy toxic examples
            return [
                {"prompt": "This is a test toxic generation", "type": "toxic_prompt"}
                for _ in range(10)
            ]
    
    def generate_completion(self, prompt: str, max_length: int = 50) -> str:
        """Generate text completion for a prompt"""
        inputs = self.estimator.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.estimator.device)
        
        with torch.no_grad():
            outputs = self.estimator.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.estimator.tokenizer.eos_token_id
            )
        
        completion = self.estimator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion
    
    def run_experiment(self):
        """Run the complete toxicity attribution experiment"""
        logger.info("="*50)
        logger.info("Starting Toxicity Attribution Experiment")
        logger.info("="*50)
        
        # Load data
        training_texts = self.load_training_data()
        test_cases = self.load_test_data_toxic()
        
        # Run influence estimation for each test case
        for i, test_case in enumerate(test_cases[:5]):  # Limit to 5 for feasibility
            logger.info(f"\n--- Test Case {i+1}/{len(test_cases[:5])} ---")
            logger.info(f"Prompt: {test_case['prompt'][:100]}...")
            
            # Generate completion
            completion = self.generate_completion(test_case['prompt'])
            logger.info(f"Generated: {completion[:100]}...")
            
            # Compute influences
            top_influences = self.estimator.get_top_k_influences(
                completion,
                training_texts,
                k=10
            )
            
            # Store results
            result = {
                "test_case_id": i,
                "prompt": test_case['prompt'],
                "completion": completion,
                "top_influences": [
                    {
                        "rank": rank + 1,
                        "index": idx,
                        "score": score,
                        "text": text[:200]  # Truncate for storage
                    }
                    for rank, (idx, score, text) in enumerate(top_influences)
                ]
            }
            
            self.results["test_cases"].append(result)
            
            # Print top 3 influences
            logger.info("\nTop 3 Influential Training Examples:")
            for rank, (idx, score, text) in enumerate(top_influences[:3], 1):
                logger.info(f"{rank}. Score: {score:.4f}")
                logger.info(f"   Text: {text[:100]}...")
        
        # Save results
        self.save_results()
        
        logger.info("\n" + "="*50)
        logger.info("Experiment Complete!")
        logger.info("="*50)
    
    def save_results(self, filename: str = "untrac_toxicity_results.json"):
        """Save experimental results to JSON file"""
        output_path = f"/mnt/user-data/outputs/{filename}"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


# Baseline Methods for Comparison
class TracInBaseline:
    """
    TracIn baseline implementation for comparison.
    Simplified version that uses the final checkpoint only (GradDot).
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def compute_gradient(self, text: str) -> Dict[str, torch.Tensor]:
        """Compute gradients for a given text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        self.model.zero_grad()
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        
        # Extract gradients
        gradients = {
            name: param.grad.clone().detach()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }
        
        return gradients
    
    def compute_influence_scores(
        self,
        test_text: str,
        training_texts: List[str]
    ) -> np.ndarray:
        """
        Compute TracIn influence scores (simplified version).
        Influence = gradient(test) · gradient(train)
        """
        # Compute test gradient
        test_grad = self.compute_gradient(test_text)
        
        # Compute influence scores
        influence_scores = []
        
        for train_text in tqdm(training_texts, desc="Computing TracIn scores"):
            train_grad = self.compute_gradient(train_text)
            
            # Compute dot product of gradients
            score = 0.0
            for name in test_grad:
                if name in train_grad:
                    score += torch.sum(test_grad[name] * train_grad[name]).item()
            
            influence_scores.append(score)
        
        return np.array(influence_scores)


def main():
    """Main execution function"""
    
    # Configure experiment
    config = UnTracConfig(
        model_name="gpt2",  # Start with GPT-2 small
        unlearning_rate=5e-5,
        unlearning_steps=1,
        max_train_samples=1000,  # Reduced for faster testing
        max_test_samples=10,
        top_k_influences=10,
        gradient_checkpointing=False
    )
    
    logger.info("Configuration:")
    logger.info(json.dumps(asdict(config), indent=2))
    
    # Run experiment
    experiment = ToxicityAttributionExperiment(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
