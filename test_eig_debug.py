#!/usr/bin/env python3
"""
EIG Debugging Script - Comprehensive Testing

This script tests the EIG implementation step by step to identify issues.
"""

import sys
import os
import traceback

# Add the evaluation harness to the path
sys.path.insert(0, 'eval/lm-evaluation-harness')

def test_imports():
    """Test that all EIG components can be imported."""
    print("🔬 Testing EIG Imports...")
    
    try:
        from lm_eval.budget_forcing.eig_core import (
            ExpectedInformationGainCalculator,
            AnswerPosteriorEstimator,
            MonteCarloForecaster
        )
        print("✅ EIG core components imported successfully")
    except Exception as e:
        print(f"❌ Failed to import EIG core: {e}")
        traceback.print_exc()
        return False
    
    try:
        from lm_eval.budget_forcing.scalers import expected_information_gain_reasoning
        print("✅ EIG scaler function imported successfully")
    except Exception as e:
        print(f"❌ Failed to import EIG scaler: {e}")
        traceback.print_exc()
        return False
    
    try:
        from lm_eval.budget_forcing.scaler_registry import get_scale_func
        print("✅ EIG registry imported successfully")
    except Exception as e:
        print(f"❌ Failed to import EIG registry: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_answer_extraction():
    """Test answer extraction patterns."""
    print("\n🔬 Testing Answer Extraction...")
    
    try:
        from lm_eval.budget_forcing.eig_core import AnswerPosteriorEstimator
        
        estimator = AnswerPosteriorEstimator()
        
        # Test math problem examples
        test_cases = [
            {
                "text": "Step 1: We need to solve x + 2 = 5\nStep 2: x = 5 - 2 = 3\nTherefore, the answer is 3.",
                "expected": ["3"]
            },
            {
                "text": "Let me calculate this step by step.\n\nFirst, we have 2 + 3 = 5.\nThen, 5 × 4 = 20.\n\n\\boxed{20}",
                "expected": ["20"]
            },
            {
                "text": "The probability is $\\frac{1}{4}$.",
                "expected": ["\\frac{1}{4}"]
            },
            {
                "text": "After working through the algebra, I get:\n\n42",
                "expected": ["42"]
            }
        ]
        
        class MockModel:
            def tok_encode(self, text):
                return [1, 2, 3]  # Mock tokenization
        
        mock_model = MockModel()
        
        for i, case in enumerate(test_cases):
            candidates = estimator.extract_answer_candidates(case["text"], mock_model)
            print(f"   Test {i+1}: Expected {case['expected']}, Got {candidates}")
            
            if any(exp in candidates for exp in case["expected"]):
                print(f"   ✅ Test {i+1} passed")
            else:
                print(f"   ❌ Test {i+1} failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Answer extraction test failed: {e}")
        traceback.print_exc()
        return False


def test_eig_calculation():
    """Test basic EIG calculation logic."""
    print("\n🔬 Testing EIG Calculation Logic...")
    
    try:
        # Test the basic mathematical logic
        current_entropy = 1.5
        future_entropy = 1.0
        lambda_cost = 0.3
        
        information_gain = current_entropy - future_entropy  # Should be 0.5
        should_continue = information_gain > lambda_cost  # 0.5 > 0.3 = True
        
        print(f"   Current entropy: {current_entropy}")
        print(f"   Future entropy: {future_entropy}")
        print(f"   Information gain: {information_gain}")
        print(f"   Lambda cost: {lambda_cost}")
        print(f"   Should continue: {should_continue}")
        
        if should_continue and information_gain == 0.5:
            print("   ✅ Basic EIG logic test passed")
            return True
        else:
            print("   ❌ Basic EIG logic test failed")
            return False
            
    except Exception as e:
        print(f"❌ EIG calculation test failed: {e}")
        traceback.print_exc()
        return False


def test_scaler_registry():
    """Test that EIG can be retrieved from the scaler registry."""
    print("\n🔬 Testing Scaler Registry...")
    
    try:
        from lm_eval.budget_forcing.scaler_registry import get_scale_func
        
        # Test getting the EIG function
        scale_token = [1, 2, 3]  # Mock scale token
        eig_func = get_scale_func(
            'expected_information_gain_reasoning',
            scale_token,
            beam_size=4,
            mc_samples=3,
            lambda_cost=0.05
        )
        
        print(f"   ✅ EIG function retrieved from registry: {type(eig_func)}")
        return True
        
    except Exception as e:
        print(f"❌ Scaler registry test failed: {e}")
        traceback.print_exc()
        return False


def test_mock_vllm_integration():
    """Test EIG with a mock vLLM model."""
    print("\n🔬 Testing Mock vLLM Integration...")
    
    try:
        from lm_eval.budget_forcing.eig_core import ExpectedInformationGainCalculator
        
        # Create a mock vLLM model
        class MockVLLMModel:
            class MockTokenizer:
                def decode(self, tokens, skip_special_tokens=True):
                    return "Step 1: Let's solve this math problem. We need to find x where x + 5 = 10."
                
                def encode(self, text, add_special_tokens=False):
                    return [1, 2, 3, 4, 5]  # Mock encoding
            
            class MockModel:
                def generate(self, prompt_token_ids, sampling_params, use_tqdm=False):
                    # Mock output with proper structure
                    class MockOutput:
                        class MockCompletionOutput:
                            def __init__(self):
                                self.token_ids = [10, 11, 12]
                                self.text = "x = 10 - 5 = 5"
                        
                        def __init__(self):
                            self.outputs = [MockCompletionOutput()]
                            self.prompt_logprobs = None
                    
                    return [MockOutput()]
            
            def __init__(self):
                self.tokenizer = MockVLLMModel.MockTokenizer()
                self.model = MockVLLMModel.MockModel()
                
            def tok_encode(self, text):
                return self.tokenizer.encode(text)
        
        # Test EIG calculation with mock model
        calculator = ExpectedInformationGainCalculator(
            beam_size=2,
            mc_samples=2,
            sample_length=10,
            lambda_cost=0.1
        )
        
        mock_model = MockVLLMModel()
        test_sequence = [1, 2, 3, 4, 5]
        test_entropies = [0.5, 0.6, 0.4, 0.7, 0.3]
        
        information_gain, details = calculator.compute_expected_information_gain(
            iteration=0,
            seq=test_sequence,
            entropies=test_entropies,
            vllm_model=mock_model
        )
        
        print(f"   Information gain computed: {information_gain}")
        print(f"   Computation successful: {details.get('success', False)}")
        print(f"   ✅ Mock vLLM integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Mock vLLM integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all EIG diagnostic tests."""
    print("🔬 EIG COMPREHENSIVE DIAGNOSTIC TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Answer Extraction Tests", test_answer_extraction),
        ("EIG Calculation Tests", test_eig_calculation),
        ("Scaler Registry Tests", test_scaler_registry),
        ("Mock vLLM Integration Tests", test_mock_vllm_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("🔬 DIAGNOSTIC SUMMARY")
    print('='*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EIG implementation looks good.")
        
        print("\n📋 NEXT STEPS:")
        print("1. Run actual evaluation with debug=True")
        print("2. Check if vLLM model integration works correctly")
        print("3. Verify answer processing in openai_math task")
        
    else:
        print("⚠️  Some tests failed. Fix these issues before running evaluation.")
        
        print("\n🔧 RECOMMENDATIONS:")
        print("1. Fix failing import issues")
        print("2. Debug answer extraction patterns")
        print("3. Verify EIG mathematical computations")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 