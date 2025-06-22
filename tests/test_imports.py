#!/usr/bin/env python3
"""
Quick import test to verify vLLM step-wise uncertainty modules are accessible.
"""

import sys
sys.path.insert(0, "eval/lm-evaluation-harness")

print("Testing imports...")

try:
    from lm_eval.budget_forcing.vllm_core import generate_with_budget_forcing_vllm
    print("✅ vllm_core import successful")
except ImportError as e:
    print(f"❌ vllm_core import failed: {e}")

try:
    from lm_eval.budget_forcing.scaler_registry import get_scale_func
    print("✅ scaler_registry import successful")
except ImportError as e:
    print(f"❌ scaler_registry import failed: {e}")

try:
    from lm_eval.budget_forcing.scalers import step_wise_uncertainty_driven
    print("✅ scalers import successful")
except ImportError as e:
    print(f"❌ scalers import failed: {e}")

try:
    from lm_eval.models.vllm_causallms import VLLM
    print("✅ vllm_causallms import successful")
except ImportError as e:
    print(f"❌ vllm_causallms import failed: {e}")

print("\nImport test complete!") 