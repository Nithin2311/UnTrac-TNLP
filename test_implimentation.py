"""
Quick Test Script for UnTrac Implementation
This script runs a minimal test to verify the implementation works correctly.
"""

import torch
import logging
from dataclasses import asdict
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_setup():
    """Test basic environment setup"""
    logger.info("="*60)
    logger.info("Testing Basic Setup")
    logger.info("="*60)
    
    # Check PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available - tests will run on CPU (slow)")
    
    return torch.cuda.is_available()

def test_model_loading():
    """Test model loading"""
    logger.info("\n" + "="*60)
    logger.info("Testing Model Loading")
    logger.info("="*60)
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        logger.info("Loading GPT-2 Small...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Parameters: {param_count:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False

def test_untrac_basic():
    """Test basic UnTrac functionality"""
    logger.info("\n" + "="*60)
    logger.info("Testing UnTrac Basic Functionality")
    logger.info("="*60)
    
    try:
        from untrac_base_implementation import UnTracConfig, UnTracInfluenceEstimator
        
        # Create minimal config
        config = UnTracConfig(
            model_name="gpt2",
            max_train_samples=10,
            max_test_samples=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info("Initializing UnTrac estimator...")
        estimator = UnTracInfluenceEstimator(config)
        
        logger.info("✓ UnTrac estimator initialized")
        
        # Test loss computation
        logger.info("\nTesting loss computation...")
        test_text = "This is a test sentence."
        loss = estimator.compute_loss(test_text)
        logger.info(f"✓ Loss computed: {loss:.4f}")
        
        # Test influence computation with minimal data
        logger.info("\nTesting influence computation...")
        training_texts = [
            "This is training example 1.",
            "This is training example 2.",
            "This is training example 3.",
        ]
        
        influence_scores = estimator.compute_influence_scores(
            test_text,
            training_texts,
            show_progress=False
        )
        
        logger.info(f"✓ Influence scores computed")
        logger.info(f"  Shape: {influence_scores.shape}")
        logger.info(f"  Mean: {influence_scores.mean():.6f}")
        logger.info(f"  Std: {influence_scores.std():.6f}")
        
        # Test top-k retrieval
        logger.info("\nTesting top-k influence retrieval...")
        top_k = estimator.get_top_k_influences(
            test_text,
            training_texts,
            k=2
        )
        
        logger.info(f"✓ Top-k influences retrieved: {len(top_k)} examples")
        for rank, (idx, score, text) in enumerate(top_k, 1):
            logger.info(f"  {rank}. Index={idx}, Score={score:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ UnTrac test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_experiment_setup():
    """Test experiment setup"""
    logger.info("\n" + "="*60)
    logger.info("Testing Experiment Setup")
    logger.info("="*60)
    
    try:
        from untrac_base_implementation import UnTracConfig, ToxicityAttributionExperiment
        
        config = UnTracConfig(
            model_name="gpt2",
            max_train_samples=5,
            max_test_samples=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info("Creating experiment...")
        experiment = ToxicityAttributionExperiment(config)
        
        logger.info("✓ Experiment created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Experiment setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_end_to_end():
    """Run a minimal end-to-end test"""
    logger.info("\n" + "="*60)
    logger.info("Running Minimal End-to-End Test")
    logger.info("="*60)
    
    try:
        from untrac_base_implementation import UnTracConfig, UnTracInfluenceEstimator
        
        # Minimal configuration
        config = UnTracConfig(
            model_name="gpt2",
            unlearning_rate=5e-5,
            unlearning_steps=1,
            max_train_samples=5,
            max_test_samples=1,
            top_k_influences=3,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info("Configuration:")
        logger.info(json.dumps(asdict(config), indent=2))
        
        # Initialize
        logger.info("\nInitializing estimator...")
        estimator = UnTracInfluenceEstimator(config)
        
        # Create dummy data
        logger.info("Creating test data...")
        test_text = "This model sometimes generates problematic content."
        training_texts = [
            "The weather is nice today.",
            "Programming is fun and challenging.",
            "Machine learning models need careful training.",
            "Data quality matters for good results.",
            "Testing ensures code reliability."
        ]
        
        # Run influence estimation
        logger.info("\nComputing influences...")
        top_influences = estimator.get_top_k_influences(
            test_text,
            training_texts,
            k=3
        )
        
        # Display results
        logger.info("\n" + "-"*60)
        logger.info("RESULTS")
        logger.info("-"*60)
        logger.info(f"\nTest text: {test_text}")
        logger.info("\nTop 3 Influential Training Examples:")
        
        for rank, (idx, score, text) in enumerate(top_influences, 1):
            logger.info(f"\n{rank}. Training Example {idx}")
            logger.info(f"   Influence Score: {score:.6f}")
            logger.info(f"   Text: {text}")
        
        logger.info("\n✓ End-to-end test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "="*60)
    logger.info("RUNNING ALL TESTS")
    logger.info("="*60 + "\n")
    
    tests = [
        ("Basic Setup", test_basic_setup),
        ("Model Loading", test_model_loading),
        ("UnTrac Basic", test_untrac_basic),
        ("Experiment Setup", test_experiment_setup),
        ("End-to-End", test_minimal_end_to_end),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Implementation is working correctly.")
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("You're ready to run the full experiments!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Run: python untrac_base_implementation.py")
        logger.info("2. Run: python scaling_experiment.py")
        logger.info("3. Check results in /mnt/user-data/outputs/")
