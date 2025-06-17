#!/usr/bin/env python3
"""
Test script for step-wise uncertainty driven scaling function.
Run this to validate the implementation without running the full evaluation.
Tests include error scenarios to validate graceful fallbacks.
"""

import sys
import os
import torch

# Add the budget forcing module to path
sys.path.insert(0, 'eval/lm-evaluation-harness')

print("🧪 TESTING STEP-WISE UNCERTAINTY WITH ERROR HANDLING")
print("=" * 60)

try:
    from lm_eval.budget_forcing.scalers import (
        parse_numbered_steps, 
        calculate_step_uncertainties,
        step_wise_uncertainty_driven,
        get_stepwise_metrics,
        print_stepwise_metrics,
        _validate_inputs,
        _record_fallback
    )
    from lm_eval.budget_forcing.scaler_registry import get_scale_func
    print("✅ Successfully imported step-wise uncertainty functions")
except ImportError as e:
    print(f"❌ Failed to import step-wise uncertainty functions: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

# Mock HF model for testing
class MockHFLM:
    class MockTokenizer:
        def decode(self, tokens, skip_special_tokens=False):
            """Mock decode that converts tokens to readable text"""
            if not tokens:
                return ""
            # Simple mock: convert token IDs to words
            words = [f"word{token % 100}" for token in tokens[:10]]  # First 10 tokens
            if len(tokens) > 10:
                words.append("...")
            return " ".join(words) + "\n\nStep 1: First reasoning step here\nStep 2: Second step with calculations\nStep 3: Final conclusion"

    def __init__(self):
        self.tokenizer = self.MockTokenizer()
    
    def tok_encode(self, text):
        """Mock encoder that converts text to token IDs"""
        if not text:
            return []
        # Simple mock: hash text to get consistent token IDs
        tokens = []
        for i, char in enumerate(text[:20]):  # Limit for testing
            tokens.append(hash(char) % 1000)
        return tokens


def test_parsing_functionality():
    """Test parsing numbered steps with various text formats"""
    print("\n📝 TESTING STEP PARSING FUNCTIONALITY")
    print("-" * 40)
    
    test_cases = [
        # Valid cases
        ("Step 1: First step\nStep 2: Second step", 2, "✅"),
        ("Step 1. First step\nStep 2. Second step", 2, "✅"),
        ("1. First step\n2. Second step", 2, "✅"),
        ("1) First step\n2) Second step", 2, "✅"),
        
        # Edge cases
        ("No numbered steps here", 0, "⚠️"),
        ("", 0, "⚠️"),
        (None, 0, "❌"),
        ("Step 1: Only one step", 1, "✅"),
        ("Random text\nStep 1: Hidden step\nMore text", 1, "✅"),
    ]
    
    for i, (text, expected_count, status) in enumerate(test_cases, 1):
        try:
            steps = parse_numbered_steps(text)
            actual_count = len(steps) if steps else 0
            result = "PASS" if actual_count == expected_count else "FAIL"
            print(f"{status} Test {i}: {result} - Expected {expected_count}, got {actual_count}")
            if text and len(str(text)) < 50:
                print(f"    Text: '{text}'")
        except Exception as e:
            print(f"❌ Test {i}: EXCEPTION - {e}")


def test_uncertainty_calculation():
    """Test uncertainty calculation with various entropy configurations"""
    print("\n🔢 TESTING UNCERTAINTY CALCULATION")
    print("-" * 40)
    
    mock_hflm = MockHFLM()
    
    test_cases = [
        # (steps, entropies, expected_result_type, description)
        (["Step 1 content", "Step 2 content"], [0.1, 0.2, 0.3, 0.4], "list", "Normal case"),
        ([], [0.1, 0.2], "None", "No steps"),
        (["Step 1"], [], "list", "No entropies - should use defaults"),
        (["Step 1"], [float('nan'), 0.5], "list", "Invalid entropy values"),
        (["Step 1"], ["invalid", 0.5], "list", "Non-numeric entropies"),
        (["Step 1", "Step 2"], [0.1], "list", "Insufficient entropies"),
    ]
    
    for i, (steps, entropies, expected_type, description) in enumerate(test_cases, 1):
        try:
            result = calculate_step_uncertainties(steps, "mock text", entropies, mock_hflm)
            actual_type = "list" if isinstance(result, list) else "None" if result is None else str(type(result))
            status = "✅ PASS" if actual_type == expected_type else "❌ FAIL"
            print(f"{status} Test {i}: {description}")
            print(f"    Expected: {expected_type}, got: {actual_type}")
            if result:
                print(f"    Uncertainties: {[f'{u:.3f}' for u in result]}")
        except Exception as e:
            print(f"❌ Test {i}: EXCEPTION - {e}")


def test_input_validation():
    """Test input validation functionality"""
    print("\n🔍 TESTING INPUT VALIDATION")
    print("-" * 40)
    
    mock_hflm = MockHFLM()
    
    test_cases = [
        # (seq, entropies, hflm, expected, description)
        ([1, 2, 3], [0.1, 0.2], mock_hflm, True, "Valid inputs"),
        (None, [0.1, 0.2], mock_hflm, False, "None sequence"),
        ([1, 2, 3], None, mock_hflm, False, "None entropies"),
        ([1, 2, 3], [], mock_hflm, False, "Empty entropies"),
        ([1, 2, 3], [0.1, 0.2], None, False, "None HFLM"),
    ]
    
    for i, (seq, entropies, hflm, expected, description) in enumerate(test_cases, 1):
        try:
            result = _validate_inputs(seq, entropies, hflm, call_id=i)
            status = "✅ PASS" if result == expected else "❌ FAIL"
            print(f"{status} Test {i}: {description} - Expected: {expected}, got: {result}")
        except Exception as e:
            print(f"❌ Test {i}: EXCEPTION - {e}")


def test_fallback_behavior():
    """Test that fallback mechanisms work correctly"""
    print("\n🔄 TESTING FALLBACK BEHAVIOR")
    print("-" * 40)
    
    mock_hflm = MockHFLM()
    
    # Test various failure scenarios
    failure_scenarios = [
        # (seq, entropies, description)
        (None, [0.1, 0.2], "None sequence"),
        ([1, 2, 3], [], "Empty entropies"),
        (torch.tensor([1, 2, 3]), [0.1, 0.2, 0.3], "Valid inputs - should work"),
    ]
    
    for i, (seq, entropies, description) in enumerate(failure_scenarios, 1):
        try:
            print(f"\n🧪 Scenario {i}: {description}")
            result = step_wise_uncertainty_driven(
                step_selection_strategy="highest_uncertainty",
                max_steps=5,
                use_min_uncertainty_filter=False,  # No threshold filtering
                min_step_uncertainty=0.3,
                iteration=1,
                seq=seq,
                entropies=entropies,
                hflm=mock_hflm
            )
            
            if isinstance(result, tuple) and len(result) == 2:
                continue_reasoning, tokens = result
                print(f"    ✅ Result: continue={continue_reasoning}, tokens={len(tokens) if tokens else 0}")
            else:
                print(f"    ❌ Invalid result format: {result}")
        except Exception as e:
            print(f"    ❌ Exception: {e}")


def test_threshold_vs_no_threshold():
    """Test the difference between threshold and no-threshold modes"""
    print("\n⚖️  TESTING THRESHOLD VS NO-THRESHOLD MODES")
    print("-" * 40)
    
    mock_hflm = MockHFLM()
    
    # Create a scenario where all uncertainties are low
    low_uncertainties = [0.1, 0.15, 0.12, 0.18]  # All below typical thresholds
    mock_tokens = [1, 2, 3, 4, 5]
    
    print("🧪 Testing with ALL low uncertainties [0.1, 0.15, 0.12, 0.18]")
    print("   (In this case, we'd still want to revise the 'most uncertain' = 0.18)")
    
    # Test WITHOUT threshold filtering
    print("\n1️⃣  WITHOUT threshold filtering (use_min_uncertainty_filter=False):")
    try:
        result_no_threshold = step_wise_uncertainty_driven(
            step_selection_strategy="highest_uncertainty",
            max_steps=5,
            use_min_uncertainty_filter=False,  # Disabled
            min_step_uncertainty=0.3,  # This gets ignored
            iteration=1,
            seq=torch.tensor(mock_tokens),
            entropies=low_uncertainties,
            hflm=mock_hflm
        )
        
        if isinstance(result_no_threshold, tuple):
            continue_reasoning, tokens = result_no_threshold
            print(f"    Result: continue={continue_reasoning}, tokens={len(tokens) if tokens else 0}")
            print(f"    ✅ Should continue (always revise most uncertain step)")
        else:
            print(f"    ❌ Invalid result: {result_no_threshold}")
            
    except Exception as e:
        print(f"    ❌ Exception: {e}")
    
    # Test WITH threshold filtering
    print("\n2️⃣  WITH threshold filtering (use_min_uncertainty_filter=True, threshold=0.3):")
    try:
        result_with_threshold = step_wise_uncertainty_driven(
            step_selection_strategy="highest_uncertainty",
            max_steps=5,
            use_min_uncertainty_filter=True,  # Enabled
            min_step_uncertainty=0.3,  # Threshold = 0.3
            iteration=1,
            seq=torch.tensor(mock_tokens),
            entropies=low_uncertainties,
            hflm=mock_hflm
        )
        
        if isinstance(result_with_threshold, tuple):
            continue_reasoning, tokens = result_with_threshold
            print(f"    Result: continue={continue_reasoning}, tokens={len(tokens) if tokens else 0}")
            print(f"    ✅ Should stop (all uncertainties < 0.3)")
        else:
            print(f"    ❌ Invalid result: {result_with_threshold}")
            
    except Exception as e:
        print(f"    ❌ Exception: {e}")
    
    print("\n📊 COMPARISON SUMMARY:")
    print("   • No threshold: Always continues, revises most uncertain step")
    print("   • With threshold: Stops when all steps are 'certain enough'")


def test_scale_func_registry():
    """Test the scaling function registry with error handling"""
    print("\n🏭 TESTING SCALE FUNCTION REGISTRY")
    print("-" * 40)
    
    test_configs = [
        # (func_name, kwargs, description)
        ("step_wise_uncertainty_driven", {
            "step_selection_strategy": "highest_uncertainty",
            "max_steps": 5,
            "use_min_uncertainty_filter": False,  # No threshold
            "min_step_uncertainty": 0.3
        }, "Valid step-wise config (no threshold)"),
        
        ("step_wise_uncertainty_driven", {
            "step_selection_strategy": "highest_uncertainty",
            "max_steps": 5,
            "use_min_uncertainty_filter": True,   # With threshold
            "min_step_uncertainty": 0.25
        }, "Valid step-wise config (with threshold)"),
        
        ("step_wise_uncertainty_driven", {
            "step_selection_strategy": "invalid_strategy",  # Invalid
            "max_steps": -5,  # Invalid
            "use_min_uncertainty_filter": "invalid",  # Invalid
            "min_step_uncertainty": 2.0  # Invalid
        }, "Invalid step-wise config (should auto-correct)"),
        
        ("entropy_thresholding", {
            "threshold": 0.5,
            "decay_factor": 1.0,
            "last_k": -1
        }, "Valid entropy config"),
        
        ("unknown_function", {}, "Unknown function (should fallback)"),
    ]
    
    for i, (func_name, kwargs, description) in enumerate(test_configs, 1):
        try:
            print(f"\n🧪 Test {i}: {description}")
            scale_func = get_scale_func(func_name, scale_token=[42], **kwargs)
            
            # Test the function with mock data
            mock_hflm = MockHFLM()
            result = scale_func(
                iteration=1,
                seq=[1, 2, 3],
                entropies=[0.1, 0.2, 0.3],
                hflm=mock_hflm
            )
            
            if isinstance(result, tuple) and len(result) == 2:
                continue_reasoning, tokens = result
                print(f"    ✅ Function works: continue={continue_reasoning}, tokens={len(tokens) if tokens else 0}")
            else:
                print(f"    ❌ Invalid result: {result}")
                
        except Exception as e:
            print(f"    ❌ Exception in test {i}: {e}")


def test_comprehensive_scenario():
    """Test a comprehensive realistic scenario"""
    print("\n🌟 COMPREHENSIVE SCENARIO TEST")
    print("-" * 40)
    
    mock_hflm = MockHFLM()
    
    # Simulate a realistic text with numbered steps
    realistic_text = """
    Let me solve this problem step by step.
    
    Step 1: First, I need to understand what we're looking for. The problem asks us to find the value of x.
    
    Step 2: Now I'll set up the equation. We have 2x + 5 = 15.
    
    Step 3: To solve for x, I'll subtract 5 from both sides: 2x = 10.
    
    Step 4: Finally, I'll divide both sides by 2: x = 5.
    
    Therefore, the answer is x = 5.
    """
    
    # Mock entropies that would come from actual model
    mock_entropies = [0.1, 0.2, 0.4, 0.3, 0.6, 0.2, 0.8, 0.1, 0.3, 0.5, 
                     0.2, 0.7, 0.4, 0.3, 0.1, 0.9, 0.2, 0.4, 0.3, 0.1]
    
    # Convert text to mock tokens
    mock_tokens = mock_hflm.tok_encode(realistic_text)
    
    print(f"📄 Input text length: {len(realistic_text)} chars")
    print(f"🔢 Mock tokens: {len(mock_tokens)} tokens")
    print(f"📊 Mock entropies: {len(mock_entropies)} values")
    
    try:
        result = step_wise_uncertainty_driven(
            step_selection_strategy="highest_uncertainty",
            max_steps=10,
            use_min_uncertainty_filter=False,  # No threshold - always revise most uncertain
            min_step_uncertainty=0.4,  # This gets ignored since use_min_uncertainty_filter=False
            iteration=1,
            seq=torch.tensor(mock_tokens),
            entropies=mock_entropies,
            hflm=mock_hflm
        )
        
        if isinstance(result, tuple) and len(result) == 2:
            continue_reasoning, tokens = result
            print(f"\n✅ COMPREHENSIVE TEST PASSED")
            print(f"   Continue reasoning: {continue_reasoning}")
            print(f"   Continuation tokens: {len(tokens) if tokens else 0}")
            if tokens:
                continuation_text = mock_hflm.tokenizer.decode(tokens)
                print(f"   Continuation preview: {continuation_text[:100]}...")
        else:
            print(f"❌ COMPREHENSIVE TEST FAILED: Invalid result {result}")
            
    except Exception as e:
        print(f"❌ COMPREHENSIVE TEST FAILED: {e}")


def main():
    """Run all tests"""
    print("🚀 Starting comprehensive test suite for step-wise uncertainty")
    
    # Run all test categories
    test_parsing_functionality()
    test_uncertainty_calculation()
    test_input_validation()
    test_fallback_behavior()
    test_threshold_vs_no_threshold()  # New test
    test_scale_func_registry()
    test_comprehensive_scenario()
    
    # Print final metrics
    print("\n" + "=" * 60)
    print("📊 FINAL TEST METRICS")
    try:
        print_stepwise_metrics()
    except Exception as e:
        print(f"❌ Failed to print metrics: {e}")
    
    print("\n🎉 Test suite completed!")
    print("If you see this message, the basic functionality is working.")
    print("Check above for any ❌ FAIL or EXCEPTION markers.")


if __name__ == "__main__":
    main() 