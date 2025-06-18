#!/usr/bin/env python3
"""
Quick test script to verify vLLM step-wise uncertainty implementation.
Run this to check if the budget forcing works before running full eval.
"""

import os
import sys
import torch
import argparse

# Add the lm-eval path so we can import
sys.path.insert(0, "eval/lm-evaluation-harness")

from lm_eval.models.vllm_causallms import VLLM
from lm_eval.budget_forcing.vllm_core import generate_with_budget_forcing_vllm

def test_vllm_stepwise(debug: bool = False):
    """Test basic vLLM step-wise uncertainty functionality."""
    
    print("🔬 Testing vLLM Step-wise Uncertainty Implementation")
    print("="*60)
    
    # Initialize small model for testing
    model_name = "microsoft/DialoGPT-small"  # Small, fast model for testing
    print(f"Loading model: {model_name}")
    
    try:
        llm = VLLM(
            pretrained=model_name,
            max_model_len=2048,
            dtype="float16",
            gpu_memory_utilization=0.5,
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test basic generation with step-wise uncertainty
    test_prompts = [
        "What is 2 + 2? Let me think step by step:\n\nStep 1:",
        "Solve for x: 3x + 5 = 14\n\nStep 1:",
    ]
    
    print(f"\n🧪 Testing step-wise generation with {len(test_prompts)} prompts")
    if debug:
        print("   Debug mode ENABLED - expect detailed logs")
    
    # Encode test prompts
    prompt_tokens = []
    for prompt in test_prompts:
        tokens = llm.tok_encode(prompt)
        prompt_tokens.append(tokens)
        print(f"Prompt: '{prompt[:50]}...' -> {len(tokens)} tokens")
    
    # Test parameters
    kwargs = {
        "step_selection_strategy": "highest_uncertainty",
        "max_steps": 3,
        "use_min_uncertainty_filter": False,
        "min_step_uncertainty": 0.3,
        "temperature": 0.0,  # Deterministic for testing
        "debug": debug,
    }
    
    print("\n🔄 Running step-wise budget forcing...")
    try:
        outputs = generate_with_budget_forcing_vllm(
            llm=llm,
            requests=prompt_tokens,
            max_tokens=200,
            stop_sequences=["</s>", "\n\n"],
            scale_func_name="step_wise_uncertainty_driven",
            **kwargs
        )
        
        print("✅ Step-wise generation completed!")
        print(f"Generated {len(outputs)} outputs")
        
        # Show results
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            print(f"\n📝 Output {i+1}:")
            print(f"Generated: '{generated_text[:200]}{'...' if len(generated_text) > 200 else ''}'")
            print(f"Tokens generated: {len(output.outputs[0].token_ids)}")
            
    except Exception as e:
        print(f"❌ Step-wise generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test entropy thresholding as well
    print("\n🧪 Testing entropy thresholding...")
    entropy_kwargs = {
        "threshold": 0.5,
        "decay_factor": 0.9,
        "last_k": -1,
        "temperature": 0.0,
        "debug": debug,
    }
    
    try:
        outputs_entropy = generate_with_budget_forcing_vllm(
            llm=llm,
            requests=prompt_tokens[:1],  # Test with just one prompt
            max_tokens=100,
            stop_sequences=["</s>"],
            scale_func_name="entropy_thresholding",
            **entropy_kwargs
        )
        
        print("✅ Entropy thresholding completed!")
        generated_text = outputs_entropy[0].outputs[0].text
        print(f"Generated: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'")
        
    except Exception as e:
        print(f"❌ Entropy thresholding failed: {e}")
        return False
    
    print("\n🎉 All tests passed! vLLM step-wise uncertainty is working.")
    return True

def test_uncertainty_calculation(debug: bool = False):
    """Test the uncertainty calculation specifically."""
    print("\n🔬 Testing uncertainty calculation...")
    
    model_name = "microsoft/DialoGPT-small"
    
    try:
        llm = VLLM(
            pretrained=model_name,
            max_model_len=1024,
            dtype="float16",
            gpu_memory_utilization=0.3,
        )
    except:
        print("❌ Could not load model for uncertainty test")
        return False
    
    # Simple test prompt
    test_prompt = "The capital of France is"
    tokens = llm.tok_encode(test_prompt)
    
    from lm_eval.budget_forcing.vllm_core import _generate_with_uncertainty_vllm
    
    try:
        outputs, uncertainties = _generate_with_uncertainty_vllm(
            llm=llm,
            prompt_token_ids=[tokens],
            max_tokens=10,
            stop_sequences=None,
            logprobs=1,
            temperature=0.7,  # Some randomness to get meaningful uncertainties
            debug=debug,
        )
        
        print(f"✅ Uncertainty calculation successful!")
        print(f"Generated tokens: {len(outputs[0].outputs[0].token_ids)}")
        print(f"Uncertainties: {uncertainties[0][:5]}...")  # Show first 5
        
        # Check that uncertainties are in valid range [0, 1]
        all_uncertainties = uncertainties[0]
        if all(0 <= u <= 1 for u in all_uncertainties):
            print("✅ All uncertainties in valid range [0, 1]")
        else:
            print("❌ Some uncertainties out of range!")
            return False
            
    except Exception as e:
        print(f"❌ Uncertainty calculation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vLLM step-wise uncertainty implementation")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    args = parser.parse_args()
    
    print("Starting vLLM step-wise uncertainty tests...\n")
    if args.debug:
        print("🔍 DEBUG MODE ENABLED - expect verbose output\n")
        # Set logging level for the lm_eval logger
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - tests may fail or be very slow")
    else:
        print(f"✅ CUDA available - using GPU {torch.cuda.get_device_name()}")
    
    success = True
    
    # Run tests
    try:
        success &= test_uncertainty_calculation(debug=args.debug)
        success &= test_vllm_stepwise(debug=args.debug)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        success = False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED! Ready to run full evaluation.")
        print("\nTo run the full evaluation:")
        print("  sbatch eval_stepwise.sh")
        print("  sbatch eval_stepwise_with_threshold.sh")
    else:
        print("❌ SOME TESTS FAILED! Check the errors above.")
        sys.exit(1) 