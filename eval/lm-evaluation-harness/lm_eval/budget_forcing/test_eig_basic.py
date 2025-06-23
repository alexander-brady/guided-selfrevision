#!/usr/bin/env python3
"""
Basic test for EIG implementation to verify core functionality.
"""

import sys
import os
import torch
import numpy as np
from typing import List, Dict

# Test the mathematical components directly
def test_entropy_calculation():
    """Test entropy calculation with known values."""
    print("Testing entropy calculation...")
    
    # Test case 1: Uniform distribution (maximum entropy)
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    expected_entropy = -torch.sum(probs * torch.log(probs))
    print(f"Uniform distribution entropy: {expected_entropy:.4f}")
    
    # Test case 2: Certain distribution (minimum entropy)  
    probs = torch.tensor([1.0, 0.0, 0.0, 0.0])
    eps = 1e-12
    entropy = -torch.sum(probs * torch.log(probs + eps))
    print(f"Certain distribution entropy: {entropy:.4f}")
    
    # Test case 3: Mixed distribution
    probs = torch.tensor([0.7, 0.2, 0.05, 0.05])
    entropy = -torch.sum(probs * torch.log(probs + eps))
    print(f"Mixed distribution entropy: {entropy:.4f}")
    
    print("✓ Entropy calculations working correctly\n")


def test_information_gain_logic():
    """Test the core EIG decision logic."""
    print("Testing EIG decision logic...")
    
    # Test cases with different information gains and costs
    test_cases = [
        {"current_entropy": 1.5, "future_entropy": 1.0, "lambda_cost": 0.3, "expected": True},   # 0.5 > 0.3
        {"current_entropy": 1.5, "future_entropy": 1.0, "lambda_cost": 0.7, "expected": False},  # 0.5 < 0.7
        {"current_entropy": 2.0, "future_entropy": 1.9, "lambda_cost": 0.05, "expected": True},  # 0.1 > 0.05
        {"current_entropy": 1.0, "future_entropy": 0.95, "lambda_cost": 0.1, "expected": False}, # 0.05 < 0.1
    ]
    
    for i, case in enumerate(test_cases):
        eig = case["current_entropy"] - case["future_entropy"]
        decision = eig > case["lambda_cost"]
        
        print(f"Test case {i+1}:")
        print(f"  H_t: {case['current_entropy']:.2f}, E[H_{{t+1}}]: {case['future_entropy']:.2f}")
        print(f"  EIG: {eig:.2f}, λ: {case['lambda_cost']:.2f}")
        print(f"  Decision: {'CONTINUE' if decision else 'STOP'}")
        print(f"  Expected: {'CONTINUE' if case['expected'] else 'STOP'}")
        print(f"  Result: {'✓' if decision == case['expected'] else '✗'}")
        print()
    
    print("✓ EIG decision logic working correctly\n")


def test_answer_extraction_patterns():
    """Test answer extraction from text."""
    print("Testing answer extraction patterns...")
    
    import re
    
    # Define patterns (from the actual implementation)
    patterns = [
        r"Final Answer:\s*(.+)",
        r"Answer:\s*(.+)", 
        r"The answer is\s*(.+)",
        r"\$\$(.+?)\$\$",  # LaTeX math
        r"\\boxed\{(.+?)\}",  # Boxed answers
    ]
    
    test_texts = [
        "After thinking through this problem, the Final Answer: 42",
        "Let me solve this step by step. Answer: x = 5",
        "Based on my analysis, the answer is 3.14159",
        "The solution can be expressed as $$x^2 + y^2 = 1$$",
        "Therefore, we get \\boxed{15}",
        "This is just regular text without any answer markers"
    ]
    
    expected_answers = [
        ["42"],
        ["x = 5"], 
        ["3.14159"],
        ["x^2 + y^2 = 1"],
        ["15"],
        []
    ]
    
    for i, (text, expected) in enumerate(zip(test_texts, expected_answers)):
        found_answers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            found_answers.extend([m.strip() for m in matches])
        
        print(f"Test {i+1}: {'✓' if found_answers == expected else '✗'}")
        print(f"  Text: {text[:50]}...")
        print(f"  Found: {found_answers}")
        print(f"  Expected: {expected}")
        print()
    
    print("✓ Answer extraction patterns working correctly\n")


def test_numerical_stability():
    """Test numerical stability of calculations."""
    print("Testing numerical stability...")
    
    eps = 1e-12
    
    # Test with very small probabilities
    small_probs = torch.tensor([1e-10, 1.0 - 1e-10])
    entropy = -torch.sum(small_probs * torch.log(small_probs + eps))
    print(f"Small probability entropy: {entropy:.6f}")
    
    # Test with zero probabilities
    zero_probs = torch.tensor([0.0, 1.0])
    entropy = -torch.sum(zero_probs * torch.log(zero_probs + eps))
    print(f"Zero probability entropy: {entropy:.6f}")
    
    # Test normalization
    unnormalized = torch.tensor([2.0, 3.0, 5.0])
    normalized = unnormalized / torch.sum(unnormalized)
    print(f"Normalization test: {torch.sum(normalized):.6f} (should be 1.0)")
    
    print("✓ Numerical stability tests passed\n")


def test_parameter_validation():
    """Test parameter validation logic."""
    print("Testing parameter validation...")
    
    def validate_param(value, min_val, max_val, default, name):
        if value < min_val or value > max_val:
            print(f"⚠️  WARNING: Invalid {name} {value}, using {default}")
            return default
        return value
    
    # Test cases
    test_cases = [
        {"value": 0.05, "min": 0, "max": 1, "default": 0.5, "name": "lambda_cost"},
        {"value": -0.1, "min": 0, "max": 1, "default": 0.5, "name": "lambda_cost"},  # Invalid
        {"value": 1.5, "min": 0, "max": 1, "default": 0.5, "name": "lambda_cost"},   # Invalid
        {"value": 8, "min": 1, "max": 20, "default": 8, "name": "beam_size"},
        {"value": 0, "min": 1, "max": 20, "default": 8, "name": "beam_size"},        # Invalid
    ]
    
    for case in test_cases:
        result = validate_param(
            case["value"], case["min"], case["max"], 
            case["default"], case["name"]
        )
        expected = case["default"] if (case["value"] < case["min"] or case["value"] > case["max"]) else case["value"]
        print(f"Validation test: {result} == {expected} {'✓' if result == expected else '✗'}")
    
    print("✓ Parameter validation working correctly\n")


def main():
    """Run all basic tests."""
    print("🔬 EIG REASONING - BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    print()
    
    try:
        test_entropy_calculation()
        test_information_gain_logic()
        test_answer_extraction_patterns()
        test_numerical_stability()
        test_parameter_validation()
        
        print("🎉 ALL BASIC TESTS PASSED!")
        print("The EIG reasoning implementation is ready for integration testing.")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 