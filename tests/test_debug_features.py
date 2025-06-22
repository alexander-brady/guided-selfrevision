#!/usr/bin/env python3
"""
Quick test to verify debug features are working before running full evaluation.
This tests the debug logging, metrics collection, and fail-fast modes.
"""

import os
import sys
import logging

# Add the lm-eval path
sys.path.insert(0, "eval/lm-evaluation-harness")

def test_debug_logging():
    """Test that debug logging is working."""
    print("🔍 Testing debug logging...")
    
    # Set up logger
    from lm_eval.utils import eval_logger
    
    # Test various log levels
    eval_logger.info("This is an INFO message")
    eval_logger.debug("This is a DEBUG message")
    eval_logger.warning("This is a WARNING message")
    
    print("✅ Debug logging test complete")

def test_scaler_registry():
    """Test scaler registry and error handling."""
    print("🔍 Testing scaler registry...")
    
    from lm_eval.budget_forcing.scaler_registry import get_scale_func
    
    # Test normal scaler
    scale_func = get_scale_func(
        "step_wise_uncertainty_driven",
        scale_token=[1, 2, 3],
        step_selection_strategy="highest_uncertainty",
        max_steps=5,
        use_min_uncertainty_filter=False,
    )
    
    print("✅ Successfully created step-wise uncertainty scaler")
    
    # Test unknown scaler (should fall back)
    fallback_func = get_scale_func(
        "unknown_scaler",
        scale_token=[1, 2, 3]
    )
    
    print("✅ Unknown scaler handled with fallback")
    
    # Test RAISE_ON_SCALER_ERROR environment variable
    print("🔍 Testing fail-fast mode...")
    
    os.environ["RAISE_ON_SCALER_ERROR"] = "1"
    
    try:
        # This should raise an error in fail-fast mode
        bad_func = get_scale_func("unknown_scaler", scale_token=[1, 2, 3])
        # Call it to trigger the error
        result = bad_func(0, [1, 2, 3], [0.1, 0.2], None)
        print("⚠️  Expected error but got result:", result)
    except Exception as e:
        print(f"✅ Fail-fast mode working: {type(e).__name__}: {e}")
    finally:
        # Reset environment
        os.environ.pop("RAISE_ON_SCALER_ERROR", None)

def test_metrics_collection():
    """Test metrics collection."""
    print("🔍 Testing metrics collection...")
    
    from lm_eval.budget_forcing.scalers import get_stepwise_metrics, print_stepwise_metrics
    
    # Get initial metrics
    initial_metrics = get_stepwise_metrics()
    print(f"✅ Retrieved initial metrics: {initial_metrics['total_calls']} total calls")
    
    # Test printing metrics
    print("📊 Sample metrics output:")
    print_stepwise_metrics()
    
    print("✅ Metrics collection test complete")

def test_import_structure():
    """Test that all imports work correctly."""
    print("🔍 Testing import structure...")
    
    try:
        from lm_eval.budget_forcing.vllm_core import generate_with_budget_forcing_vllm
        print("✅ vllm_core import successful")
    except ImportError as e:
        print(f"❌ vllm_core import failed: {e}")
        return False
    
    try:
        from lm_eval.budget_forcing.scalers import step_wise_uncertainty_driven
        print("✅ scalers import successful")
    except ImportError as e:
        print(f"❌ scalers import failed: {e}")
        return False
    
    try:
        from lm_eval.models.vllm_causallms import VLLM
        print("✅ vllm_causallms import successful")
    except ImportError as e:
        print(f"❌ vllm_causallms import failed: {e}")
        return False
    
    return True

def test_parameter_cleaning():
    """Test that parameter cleaning works correctly."""
    print("🔍 Testing parameter cleaning...")
    
    try:
        from lm_eval.budget_forcing.vllm_core import _clean_vllm_kwargs
        
        # Test with problematic parameters from the error
        test_kwargs = {
            "do_sample": False,
            "temperature": 0.0,
            "max_tokens_thinking": "auto",
            "thinking_n_ignore": 6,
            "debug": True,
            "logprobs": 1,
            "max_tokens": 100,
            "stop": ["</s>"]
        }
        
        cleaned = _clean_vllm_kwargs(test_kwargs, debug=True)
        
        # Check that problematic parameters were removed
        assert "do_sample" not in cleaned, "do_sample should be removed"
        assert "max_tokens_thinking" not in cleaned, "max_tokens_thinking should be removed"
        assert "thinking_n_ignore" not in cleaned, "thinking_n_ignore should be removed"
        assert "debug" not in cleaned, "debug should be removed"
        
        # Check that valid parameters remain
        assert "temperature" in cleaned, "temperature should remain"
        assert "logprobs" in cleaned, "logprobs should remain"
        
        print("✅ Parameter cleaning works correctly")
        print(f"   Original: {list(test_kwargs.keys())}")
        print(f"   Cleaned:  {list(cleaned.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter cleaning test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Debug Features for Step-wise Uncertainty")
    print("="*60)
    
    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    success = True
    
    try:
        success &= test_import_structure()
        success &= test_parameter_cleaning()
        test_debug_logging()
        test_scaler_registry()
        test_metrics_collection()
        
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL DEBUG FEATURES WORKING!")
        print("\nYou can now run:")
        print("  sbatch eval_stepwise.sh")
        print("\nExpected debug outputs:")
        print("  🚀 Starting vLLM budget forcing...")
        print("  🔬 _generate_with_uncertainty_vllm...")
        print("  📊 STEP-WISE UNCERTAINTY FINAL METRICS")
    else:
        print("❌ SOME DEBUG FEATURES FAILED!")
        sys.exit(1) 