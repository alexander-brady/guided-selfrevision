#!/usr/bin/env python3
"""
Quick test to verify the parameter cleaning fix for vLLM SamplingParams.
This specifically tests the fix for the 'do_sample' error.
"""

import sys
sys.path.insert(0, "eval/lm-evaluation-harness")

def test_vllm_samplingparams_fix():
    """Test that we can create SamplingParams without the problematic parameters."""
    print("🔍 Testing vLLM SamplingParams parameter fix...")
    
    try:
        from vllm import SamplingParams
        from lm_eval.budget_forcing.vllm_core import _clean_vllm_kwargs
        
        # These are the exact parameters that caused the error
        problematic_kwargs = {
            'do_sample': False,
            'temperature': 0.0,
            'max_tokens_thinking': 'auto',
            'thinking_n_ignore': 6,
            'debug': True,
            'logprobs': 1,
        }
        
        print(f"Original kwargs: {problematic_kwargs}")
        
        # Clean the parameters
        clean_kwargs = _clean_vllm_kwargs(problematic_kwargs, debug=True)
        print(f"Cleaned kwargs: {clean_kwargs}")
        
        # Try to create SamplingParams with cleaned kwargs
        sampling_params = SamplingParams(
            max_tokens=100,
            stop=["</s>"],
            **clean_kwargs
        )
        
        print("✅ SamplingParams created successfully!")
        print(f"   Temperature: {sampling_params.temperature}")
        print(f"   Logprobs: {sampling_params.logprobs}")
        print(f"   Max tokens: {sampling_params.max_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ SamplingParams test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_modify_gen_kwargs():
    """Test the modify_gen_kwargs function."""
    print("\n🔍 Testing modify_gen_kwargs function...")
    
    try:
        from lm_eval.models.vllm_causallms import VLLM
        
        # Test the static method directly
        test_kwargs = {
            'do_sample': False,
            'temperature': 0.0,
            'max_tokens_thinking': 'auto',
            'thinking_n_ignore': 6,
            'scale_func_name': 'step_wise_uncertainty_driven',
            'debug': True,
            'logprobs': 1,
        }
        
        print(f"Original kwargs: {test_kwargs}")
        
        # Apply the modification
        modified = VLLM.modify_gen_kwargs(test_kwargs.copy())
        print(f"Modified kwargs: {modified}")
        
        # Check that problematic parameters were removed
        assert 'do_sample' not in modified, "do_sample should be removed"
        assert 'max_tokens_thinking' not in modified, "max_tokens_thinking should be removed"
        assert 'thinking_n_ignore' not in modified, "thinking_n_ignore should be removed"
        assert 'scale_func_name' not in modified, "scale_func_name should be removed"
        assert 'debug' in modified, "debug should be preserved in modify_gen_kwargs"
        
        # Check that valid parameters remain
        assert 'temperature' in modified, "temperature should remain"
        assert 'logprobs' in modified, "logprobs should remain"
        
        print("✅ modify_gen_kwargs works correctly!")
        print("   Note: 'debug' is preserved here but will be cleaned before SamplingParams")
        
        return True
        
    except Exception as e:
        print(f"❌ modify_gen_kwargs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_two_stage_cleaning():
    """Test the complete two-stage parameter cleaning process."""
    print("\n🔍 Testing complete two-stage parameter cleaning...")
    
    try:
        from lm_eval.models.vllm_causallms import VLLM
        from lm_eval.budget_forcing.vllm_core import _clean_vllm_kwargs
        
        # Start with the original problematic parameters
        original_kwargs = {
            'do_sample': False,
            'temperature': 0.0,
            'max_tokens_thinking': 'auto',
            'thinking_n_ignore': 6,
            'scale_func_name': 'step_wise_uncertainty_driven',
            'debug': True,
            'logprobs': 1,
        }
        
        print(f"1. Original kwargs: {original_kwargs}")
        
        # Stage 1: modify_gen_kwargs (removes some HF-specific params)
        stage1 = VLLM.modify_gen_kwargs(original_kwargs.copy())
        print(f"2. After modify_gen_kwargs: {stage1}")
        
        # Stage 2: _clean_vllm_kwargs (removes remaining problematic params)
        stage2 = _clean_vllm_kwargs(stage1, debug=True)
        print(f"3. After _clean_vllm_kwargs: {stage2}")
        
        # Verify final result is safe for SamplingParams
        from vllm import SamplingParams
        sampling_params = SamplingParams(max_tokens=100, **stage2)
        print(f"4. SamplingParams created successfully!")
        
        # Verify the cleaning worked as expected
        assert 'do_sample' not in stage2, "do_sample should be gone"
        assert 'debug' not in stage2, "debug should be gone"
        assert 'scale_func_name' not in stage2, "scale_func_name should be gone"
        assert 'temperature' in stage2, "temperature should remain"
        assert 'logprobs' in stage2, "logprobs should remain"
        
        print("✅ Two-stage cleaning works perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Two-stage cleaning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing vLLM Parameter Fix")
    print("="*50)
    print("This test verifies the fix for the 'Unexpected keyword argument' error.\n")
    
    success = True
    
    try:
        success &= test_vllm_samplingparams_fix()
        success &= test_modify_gen_kwargs()
        success &= test_two_stage_cleaning()
        
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 ALL PARAMETER FIXES WORKING!")
        print("\nThe 'do_sample' and other parameter errors should now be resolved.")
        print("You can safely run: sbatch eval_stepwise.sh")
    else:
        print("❌ PARAMETER FIX TESTS FAILED!")
        print("The vLLM parameter issues may still exist.")
        sys.exit(1) 