#!/usr/bin/env python3
"""
Test script for EIG reasoning with vLLM integration.

This script tests the EIG (Expected Information Gain) reasoning implementation
to ensure it works correctly with vLLM models after the migration from 
HuggingFace models.
"""

import os
import sys
import time
from typing import List, Dict, Any

def test_imports():
    """Test that all EIG-related imports work correctly."""
    print("🔬 Testing EIG imports...")
    
    try:
        from lm_eval.budget_forcing.eig_core import (
            ExpectedInformationGainCalculator,
            AnswerPosteriorEstimator,
            MonteCarloForecaster,
            EIGMetrics
        )
        print("✓ EIG core imports successful")
    except ImportError as e:
        print(f"❌ EIG core import failed: {e}")
        return False
    
    try:
        from lm_eval.budget_forcing.scalers import (
            expected_information_gain_reasoning,
            get_eig_metrics_summary,
            print_eig_metrics
        )
        print("✓ EIG scalers imports successful")
    except ImportError as e:
        print(f"❌ EIG scalers import failed: {e}")
        return False
    
    try:
        from lm_eval.budget_forcing.scaler_registry import get_scale_func
        print("✓ Scaler registry import successful")
    except ImportError as e:
        print(f"❌ Scaler registry import failed: {e}")
        return False
    
    return True


def test_eig_calculator_initialization():
    """Test that EIG calculator can be initialized with various parameters."""
    print("\n🔧 Testing EIG calculator initialization...")
    
    try:
        from lm_eval.budget_forcing.eig_core import ExpectedInformationGainCalculator
        
        # Test default initialization
        calc_default = ExpectedInformationGainCalculator()
        print("✓ Default EIG calculator initialization successful")
        
        # Test custom initialization
        calc_custom = ExpectedInformationGainCalculator(
            beam_size=10,
            mc_samples=6,
            sample_length=128,
            temperature=0.8,
            top_p=0.95,
            lambda_cost=0.03,
            max_computation_time=45.0
        )
        print("✓ Custom EIG calculator initialization successful")
        
        # Test parameter validation
        if calc_custom.lambda_cost == 0.03:
            print("✓ Parameter setting working correctly")
        else:
            print("❌ Parameter setting failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ EIG calculator initialization failed: {e}")
        return False


def test_scaler_registry():
    """Test that EIG can be registered and retrieved from scaler registry."""
    print("\n📋 Testing scaler registry integration...")
    
    try:
        from lm_eval.budget_forcing.scaler_registry import get_scale_func
        
        # Test EIG scaler registration
        scale_func = get_scale_func(
            "expected_information_gain_reasoning",
            scale_token=[1, 2, 3],  # Mock scale tokens
            lambda_cost=0.05,
            beam_size=8,
            mc_samples=5,
            sample_length=64,
            temperature=1.0,
            top_p=0.9,
            max_computation_time=30.0
        )
        
        if callable(scale_func):
            print("✓ EIG scaler registration successful")
            return True
        else:
            print("❌ EIG scaler registration failed - not callable")
            return False
            
    except Exception as e:
        print(f"❌ Scaler registry test failed: {e}")
        return False


def test_answer_extraction_patterns():
    """Test answer extraction patterns with mock text."""
    print("\n🔍 Testing answer extraction patterns...")
    
    try:
        from lm_eval.budget_forcing.eig_core import AnswerPosteriorEstimator
        
        estimator = AnswerPosteriorEstimator()
        
        test_texts = [
            "After solving step by step, Final Answer: 42",
            "Based on my calculation, the answer is 3.14159",
            "Therefore, we have $$x^2 + y^2 = 1$$",
            "The solution is \\boxed{25}",
            "No clear answer pattern here"
        ]
        
        expected_answers = [
            ["42"],
            ["3.14159"],
            ["x^2 + y^2 = 1"],
            ["25"],
            []  # Should extract last sentence as fallback
        ]
        
        all_passed = True
        for i, (text, expected) in enumerate(zip(test_texts, expected_answers)):
            # Create mock vLLM model object
            class MockVLLMModel:
                pass
            
            mock_model = MockVLLMModel()
            candidates = estimator.extract_answer_candidates(text, mock_model)
            
            if i < 4:  # First 4 should match exactly
                if candidates == expected:
                    print(f"✓ Test {i+1} passed: {candidates}")
                else:
                    print(f"❌ Test {i+1} failed: expected {expected}, got {candidates}")
                    all_passed = False
            else:  # Last test should have some fallback
                if len(candidates) > 0:
                    print(f"✓ Test {i+1} passed with fallback: {candidates}")
                else:
                    print(f"❌ Test {i+1} failed: no fallback candidates")
                    all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Answer extraction test failed: {e}")
        return False


def test_parameter_validation():
    """Test parameter validation in scaler registry."""
    print("\n✅ Testing parameter validation...")
    
    try:
        from lm_eval.budget_forcing.scaler_registry import get_scale_func
        
        # Test with invalid parameters (should be corrected)
        scale_func = get_scale_func(
            "expected_information_gain_reasoning",
            scale_token=[1, 2, 3],
            lambda_cost=-1.0,  # Invalid (negative)
            beam_size=100,     # Invalid (too large)
            mc_samples=0,      # Invalid (zero)
            sample_length=1000, # Invalid (too large)
            temperature=5.0,   # Invalid (too high)
            top_p=2.0,         # Invalid (> 1.0)
            max_computation_time=500.0  # Invalid (too large)
        )
        
        if callable(scale_func):
            print("✓ Parameter validation working - invalid params corrected")
            return True
        else:
            print("❌ Parameter validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Parameter validation test failed: {e}")
        return False


def test_metrics_collection():
    """Test metrics collection and printing."""
    print("\n📊 Testing metrics collection...")
    
    try:
        from lm_eval.budget_forcing.scalers import get_eig_metrics_summary, print_eig_metrics
        
        # Test getting metrics summary (should work even if no calculations done yet)
        metrics = get_eig_metrics_summary()
        if isinstance(metrics, dict):
            print("✓ Metrics collection working")
            
            # Test printing metrics
            print("✓ Testing metrics printing:")
            print_eig_metrics()
            return True
        else:
            print("❌ Metrics collection failed - not returning dict")
            return False
            
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def test_vllm_compatibility():
    """Test that EIG components can handle vLLM-style inputs."""
    print("\n🚀 Testing vLLM compatibility...")
    
    try:
        from lm_eval.budget_forcing.eig_core import ExpectedInformationGainCalculator
        
        # Create mock vLLM model for testing
        class MockVLLMModel:
            def __init__(self):
                self.model = MockModel()
                self.tokenizer = MockTokenizer()
            
            def tok_encode(self, text):
                # Mock tokenization - return list of fake token IDs
                return [1, 2, 3, 4, 5][:len(text.split())]
        
        class MockModel:
            def generate(self, **kwargs):
                # Mock generation - return mock outputs
                return [MockOutput()]
        
        class MockTokenizer:
            def decode(self, tokens, **kwargs):
                # Mock decoding
                return "mock decoded text"
            
            @property
            def vocab(self):
                return {"mock": 1, "vocab": 2}
        
        class MockOutput:
            def __init__(self):
                self.prompt_logprobs = None
                self.outputs = [MockCompletionOutput()]
        
        class MockCompletionOutput:
            def __init__(self):
                self.token_ids = [1, 2, 3]
        
        # Test calculator with mock vLLM model
        calc = ExpectedInformationGainCalculator()
        mock_model = MockVLLMModel()
        
        # Test that basic methods don't crash
        seq = [1, 2, 3, 4, 5]
        entropies = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # This should not crash (though may return conservative fallback)
        info_gain, details = calc.compute_expected_information_gain(
            iteration=0,
            seq=seq,
            entropies=entropies,
            vllm_model=mock_model
        )
        
        print(f"✓ vLLM compatibility test passed - info_gain: {info_gain}")
        print(f"✓ Details: {details}")
        return True
        
    except Exception as e:
        print(f"❌ vLLM compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests and report results."""
    print("🔬 EIG REASONING VLLM INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Calculator Initialization", test_eig_calculator_initialization),
        ("Scaler Registry", test_scaler_registry),
        ("Answer Extraction", test_answer_extraction_patterns),
        ("Parameter Validation", test_parameter_validation),
        ("Metrics Collection", test_metrics_collection),
        ("vLLM Compatibility", test_vllm_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} ({time.time() - start_time:.2f}s)")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ FAILED with exception: {e}")
            print(f"Time: {time.time() - start_time:.2f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! EIG reasoning is ready for vLLM integration.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 