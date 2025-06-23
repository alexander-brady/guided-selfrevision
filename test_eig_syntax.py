#!/usr/bin/env python3
"""
Simplified syntax test for EIG reasoning implementation.

This test checks basic syntax and import structure without requiring
full dependencies like numpy, torch, etc.
"""

import sys
import ast

def test_python_syntax(filepath):
    """Test if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Parse the syntax
        ast.parse(source)
        print(f"✓ {filepath} - syntax OK")
        return True
        
    except SyntaxError as e:
        print(f"❌ {filepath} - syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ {filepath} - error: {e}")
        return False

def test_import_structure():
    """Test import structure without actually importing."""
    print("🔍 Testing import structure...")
    
    files_to_check = [
        "eval/lm-evaluation-harness/lm_eval/budget_forcing/eig_core.py",
        "eval/lm-evaluation-harness/lm_eval/budget_forcing/scalers.py", 
        "eval/lm-evaluation-harness/lm_eval/budget_forcing/scaler_registry.py",
        "eval/lm-evaluation-harness/lm_eval/budget_forcing/vllm_core.py",
        "eval/lm-evaluation-harness/lm_eval/models/vllm_causallms.py"
    ]
    
    all_good = True
    for filepath in files_to_check:
        if not test_python_syntax(filepath):
            all_good = False
    
    return all_good

def check_eig_integration():
    """Check specific EIG integration points."""
    print("\n🔬 Checking EIG integration points...")
    
    # Check that EIG is imported in scalers.py
    try:
        with open("eval/lm-evaluation-harness/lm_eval/budget_forcing/scalers.py", 'r') as f:
            scalers_content = f.read()
        
        if "expected_information_gain_reasoning" in scalers_content:
            print("✓ EIG function found in scalers.py")
        else:
            print("❌ EIG function not found in scalers.py")
            return False
            
        if "from lm_eval.budget_forcing.eig_core import" in scalers_content:
            print("✓ EIG core import found in scalers.py")
        else:
            print("❌ EIG core import not found in scalers.py")
            return False
    
    except Exception as e:
        print(f"❌ Error checking scalers.py: {e}")
        return False
    
    # Check that EIG is imported in scaler_registry.py
    try:
        with open("eval/lm-evaluation-harness/lm_eval/budget_forcing/scaler_registry.py", 'r') as f:
            registry_content = f.read()
        
        if "expected_information_gain_reasoning" in registry_content:
            print("✓ EIG function found in scaler_registry.py")
        else:
            print("❌ EIG function not found in scaler_registry.py")
            return False
    
    except Exception as e:
        print(f"❌ Error checking scaler_registry.py: {e}")
        return False
    
    # Check that EIG parameters are handled in vLLM core
    try:
        with open("eval/lm-evaluation-harness/lm_eval/budget_forcing/vllm_core.py", 'r') as f:
            vllm_content = f.read()
        
        if "beam_size" in vllm_content and "mc_samples" in vllm_content:
            print("✓ EIG parameters found in vllm_core.py")
        else:
            print("❌ EIG parameters not found in vllm_core.py")
            return False
    
    except Exception as e:
        print(f"❌ Error checking vllm_core.py: {e}")
        return False
    
    print("✓ All EIG integration points found")
    return True

def check_evaluation_scripts():
    """Check that evaluation scripts have been updated."""
    print("\n📋 Checking evaluation scripts...")
    
    scripts_to_check = [
        "eval_eig.sh",
        "eval_eig_fast.sh", 
        "eval_eig_precision.sh",
        "eval_eig_comparative.sh"
    ]
    
    all_good = True
    for script in scripts_to_check:
        try:
            with open(script, 'r') as f:
                content = f.read()
            
            if "--model vllm" in content:
                print(f"✓ {script} - uses vLLM models")
            else:
                print(f"❌ {script} - not updated for vLLM")
                all_good = False
                
            if "max_model_len=32768" in content:
                print(f"✓ {script} - has vLLM model args")
            else:
                print(f"❌ {script} - missing vLLM model args")
                all_good = False
                
        except Exception as e:
            print(f"❌ Error checking {script}: {e}")
            all_good = False
    
    return all_good

def main():
    """Run all syntax and structure tests."""
    print("🔬 EIG REASONING SYNTAX AND STRUCTURE TESTS")
    print("=" * 60)
    
    tests = [
        ("Python Syntax", test_import_structure),
        ("EIG Integration", check_eig_integration),
        ("Evaluation Scripts", check_evaluation_scripts),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ FAILED with exception: {e}")
    
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
        print("🎉 ALL SYNTAX TESTS PASSED! EIG implementation structure looks good.")
        print("\n📋 NEXT STEPS:")
        print("1. Set up proper Python environment with dependencies")
        print("2. Run actual functional tests")
        print("3. Test with real vLLM models")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 