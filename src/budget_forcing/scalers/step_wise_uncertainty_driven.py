import math
import re
import random
from typing import List, Dict, Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lm_eval.models.vllm_causallms import VLLM


# Global metrics tracking for step-wise uncertainty
_STEPWISE_METRICS = {
    "total_calls": 0,
    "successful_extractions": 0,
    "parsing_failures": 0,
    "steps_found_total": 0,
    "steps_revised_total": 0,
    "uncertainty_calculations_failed": 0,
    "first_call_details": None,
    "fallback_reasons": {},  # Track reasons for fallbacks
    "error_details": [],     # Detailed error logs
}

def step_wise_uncertainty_driven(
    step_selection_strategy: str,
    max_steps: int,
    use_min_uncertainty_filter: bool,
    min_step_uncertainty: float,
    iteration: int,
    tokens: torch.Tensor,
    uncertainties: List[float],
    lm: 'VLLM',
) -> tuple[bool, List[int]]:
    """
    Generate numbered reasoning steps and continue with the step that has highest uncertainty.
    
    GRACEFUL FALLBACK: If any step fails, this function will fall back to the original 
    "Wait" behavior to ensure evaluation continues without interruption.
    
    Args:
        step_selection_strategy: "highest_uncertainty", "lowest_uncertainty", "random"
        max_steps: Maximum number of reasoning steps
        use_min_uncertainty_filter: Whether to apply minimum uncertainty filtering
        min_step_uncertainty: Minimum uncertainty required to revisit a step (only used if use_min_uncertainty_filter=True)
        iteration: Current iteration
        tokens: Generated token sequence (torch.Tensor)
        uncertainties: Per-token uncertainties (List[float])
        lm: HuggingFace model instance
    
    Returns:
        tuple: (continue_reasoning: bool, continuation_tokens: List[int])
    """
    global _STEPWISE_METRICS
    
    _STEPWISE_METRICS["total_calls"] += 1
    call_id = _STEPWISE_METRICS["total_calls"]
    
    # Early validation with graceful fallback
    try:
        if not _validate_inputs(tokens, uncertainties, lm, call_id):
            return _record_fallback("input_validation_failed", lm, call_id)
    except Exception as e:
        return _record_fallback("input_validation_exception", lm, call_id, str(e))
    
    try:
        # Convert tensor to token list if needed
        if isinstance(tokens, torch.Tensor):
            seq_tokens = tokens.tolist()
        elif isinstance(tokens, list):
            seq_tokens = tokens
        else:
            return _record_fallback("unsupported_sequence_type", lm, call_id, f"Type: {type(tokens)}")
        
        # Convert tokens to text for parsing
        try:
            text = lm.tokenizer.decode(seq_tokens, skip_special_tokens=False)
            if not text or len(text.strip()) == 0:
                return _record_fallback("empty_decoded_text", lm, call_id)
        except Exception as e:
            return _record_fallback("tokenizer_decode_failed", lm, call_id, str(e))
        
        # First call detailed logging
        is_first_call = _STEPWISE_METRICS["first_call_details"] is None
        
        if is_first_call:
            print("\n" + "="*80)
            print("🔍 FIRST STEP-WISE UNCERTAINTY ANALYSIS")
            print("="*80)
            print(f"Call ID: {call_id}")
            print(f"Iteration: {iteration}")
            print(f"Sequence length: {len(seq_tokens)} tokens")
            print(f"Text length: {len(text)} chars")
            print(f"uncertainties length: {len(uncertainties)} values")
            print(f"Parameters: strategy={step_selection_strategy}, max_steps={max_steps}")
            print(f"           use_min_filter={use_min_uncertainty_filter}, min_uncertainty={min_step_uncertainty}")
            print(f"Text preview (first 500 chars):\n{text[:500]}...")
            print(f"uncertainties (last 20): {uncertainties[-20:] if len(uncertainties) >= 20 else uncertainties}")
        
        # Parse numbered steps from the text
        try:
            steps = parse_numbered_steps(text)
            if steps is None:  # parse_numbered_steps returns None on error
                return _record_fallback("step_parsing_returned_none", lm, call_id)
            
            _STEPWISE_METRICS["steps_found_total"] += len(steps)
            
            if is_first_call or len(steps) > 0:
                print(f"\n📋 Found {len(steps)} numbered steps:")
                for i, step in enumerate(steps):
                    step_preview = step[:150].replace('\n', ' ') + "..." if len(step) > 150 else step.replace('\n', ' ')
                    print(f"  Step {i+1}: {step_preview}")
            
        except Exception as e:
            return _record_fallback("step_parsing_exception", lm, call_id, str(e))
        
        # Check if we've reached max steps
        if len(steps) >= max_steps:
            print(f"✋ Reached maximum steps ({max_steps}), stopping.")
            if is_first_call:
                _STEPWISE_METRICS["first_call_details"] = {
                    "status": "max_steps_reached",
                    "steps_found": len(steps),
                    "text_length": len(text),
                    "call_id": call_id
                }
            return False, []
        
        # If no steps found, continue with numbered step encouragement
        if not steps or len(steps) == 0:
            print("⚠️  No numbered steps found, encouraging numbered step format.")
            if is_first_call:
                _STEPWISE_METRICS["first_call_details"] = {
                    "status": "no_steps_found",
                    "steps_found": 0,
                    "text_length": len(text),
                    "fallback_used": True,
                    "call_id": call_id
                }
            
            try:
                continuation_prompt = "\n\nLet me think about this more carefully and break it down into numbered steps:\n\nStep 1:"
                continuation_tokens = lm.tok_encode(continuation_prompt)
                return True, continuation_tokens
            except Exception as e:
                return _record_fallback("no_steps_continuation_failed", lm, call_id, str(e))
        
        # Calculate uncertainty for each step
        try:
            step_uncertainties = calculate_step_uncertainties(steps, text, uncertainties, lm, is_first_call)
            if step_uncertainties is None or len(step_uncertainties) != len(steps):
                return _record_fallback("uncertainty_calculation_failed", lm, call_id, 
                                       f"Expected {len(steps)} uncertainties, got {len(step_uncertainties) if step_uncertainties else 0}")
            
            _STEPWISE_METRICS["successful_extractions"] += 1
            
            print(f"\n🎯 Step uncertainties calculated:")
            for i, (step, uncertainty) in enumerate(zip(steps, step_uncertainties)):
                step_preview = step[:100].replace('\n', ' ') + "..." if len(step) > 100 else step.replace('\n', ' ')
                print(f"  Step {i+1}: uncertainty={uncertainty:.4f} | {step_preview}")
            
        except Exception as e:
            return _record_fallback("uncertainty_calculation_exception", lm, call_id, str(e))
        
        # Filter steps by minimum uncertainty if enabled
        try:
            if use_min_uncertainty_filter:
                eligible_steps = [
                    (i, uncertainty) for i, uncertainty in enumerate(step_uncertainties) 
                    if uncertainty >= min_step_uncertainty
                ]
                filter_msg = f"with min_uncertainty >= {min_step_uncertainty}"
            else:
                eligible_steps = [
                    (i, uncertainty) for i, uncertainty in enumerate(step_uncertainties)
                ]
                filter_msg = "without uncertainty filtering (always revise most uncertain step)"
                
            print(f"\n🔍 Step selection {filter_msg}:")
            print(f"   Eligible steps: {len(eligible_steps)}/{len(steps)}")
                
        except Exception as e:
            return _record_fallback("uncertainty_filtering_failed", lm, call_id, str(e))
        
        if not eligible_steps:
            if use_min_uncertainty_filter:
                print(f"✅ All steps below minimum uncertainty threshold ({min_step_uncertainty}), stopping.")
                if is_first_call:
                    _STEPWISE_METRICS["first_call_details"] = {
                        "status": "all_steps_below_threshold",
                        "steps_found": len(steps),
                        "step_uncertainties": step_uncertainties,
                        "min_uncertainty": min(step_uncertainties) if step_uncertainties else 0,
                        "max_uncertainty": max(step_uncertainties) if step_uncertainties else 0,
                        "threshold_used": min_step_uncertainty,
                        "call_id": call_id
                    }
                return False, []
            else:
                # This shouldn't happen if use_min_uncertainty_filter is False
                return _record_fallback("no_eligible_steps_unexpected", lm, call_id)
        
        # Select step based on strategy
        try:
            if step_selection_strategy == "highest_uncertainty":
                selected_idx = max(eligible_steps, key=lambda x: x[1])[0]
            elif step_selection_strategy == "lowest_uncertainty":
                selected_idx = min(eligible_steps, key=lambda x: x[1])[0]
            elif step_selection_strategy == "random":
                selected_idx = random.choice(eligible_steps)[0]
            else:
                print(f"⚠️  Unknown strategy: {step_selection_strategy}, using highest_uncertainty")
                selected_idx = max(eligible_steps, key=lambda x: x[1])[0]
        except Exception as e:
            return _record_fallback("step_selection_failed", lm, call_id, str(e))
        
        selected_uncertainty = step_uncertainties[selected_idx]
        step_num = selected_idx + 1
        _STEPWISE_METRICS["steps_revised_total"] += 1
        
        print(f"\n🔄 Selected Step {step_num} for revision:")
        print(f"   Uncertainty: {selected_uncertainty:.4f}")
        print(f"   Strategy: {step_selection_strategy}")
        print(f"   Available eligible steps: {len(eligible_steps)}")
        if use_min_uncertainty_filter:
            print(f"   Min uncertainty filter: {min_step_uncertainty}")
        else:
            print(f"   Min uncertainty filter: DISABLED (always revise most uncertain)")
        
        # Generate continuation prompt
        try:
            continuation_prompt = f"\n\nLet me revisit Step {step_num} in more detail and make sure I got it right:\n\nStep {step_num} (revisited):"
            continuation_tokens = lm.tok_encode(continuation_prompt)
            if not continuation_tokens:
                return _record_fallback("empty_continuation_tokens", lm, call_id)
        except Exception as e:
            return _record_fallback("continuation_encoding_failed", lm, call_id, str(e))
        
        if is_first_call:
            _STEPWISE_METRICS["first_call_details"] = {
                "status": "successful_revision",
                "steps_found": len(steps),
                "step_uncertainties": step_uncertainties,
                "selected_step": step_num,
                "selected_uncertainty": selected_uncertainty,
                "strategy": step_selection_strategy,
                "continuation_prompt": continuation_prompt,
                "eligible_steps_count": len(eligible_steps),
                "use_min_uncertainty_filter": use_min_uncertainty_filter,
                "min_step_uncertainty": min_step_uncertainty if use_min_uncertainty_filter else None,
                "call_id": call_id
            }
            print(f"\n📊 FIRST CALL SUMMARY:")
            print(f"   Status: Success - Will revise Step {step_num}")
            print(f"   Steps found: {len(steps)}")
            print(f"   Uncertainties: {[f'{u:.3f}' for u in step_uncertainties]}")
            print(f"   Selected: Step {step_num} (uncertainty: {selected_uncertainty:.4f})")
            print(f"   Continuation: {continuation_prompt}")
        
        print(f"\n✨ Continuing with: {continuation_prompt}")
        print("="*80)
        
        return True, continuation_tokens
        
    except Exception as e:
        return _record_fallback("critical_exception", lm, call_id, str(e))


def _validate_inputs(tokens, uncertainties, lm, call_id: int) -> bool:
    """
    Validate inputs to the step-wise uncertainty function.
    
    Args:
        tokens: Token sequence
        uncertainties: Entropy values
        lm: HuggingFace model instance
        call_id: Call identifier for logging
        
    Returns:
        bool: True if inputs are valid, False otherwise
    """
    if tokens is None:
        _log_error(call_id, "validation", "Sequence is None")
        return False
    
    if uncertainties is None:
        _log_error(call_id, "validation", "uncertainties is None")
        return False
    
    if not uncertainties:  # Empty list
        _log_error(call_id, "validation", "Empty uncertainties list")
        return False
    
    if lm is None:
        _log_error(call_id, "validation", "HFLM instance is None")
        return False
    
    if not hasattr(lm, 'tokenizer'):
        _log_error(call_id, "validation", "HFLM missing tokenizer attribute")
        return False
    
    if not hasattr(lm, 'tok_encode'):
        _log_error(call_id, "validation", "HFLM missing tok_encode method")
        return False
    
    return True


def _record_fallback(reason: str, lm, call_id: int, details: str = "") -> tuple[bool, List[int]]:
    """
    Record a fallback event and return the fallback continuation.
    
    Args:
        reason: Reason for fallback
        lm: HuggingFace model instance
        call_id: Call identifier
        details: Additional error details
        
    Returns:
        tuple: (continue_reasoning: bool, continuation_tokens: List[int])
    """
    global _STEPWISE_METRICS
    
    _STEPWISE_METRICS["parsing_failures"] += 1
    
    # Track fallback reasons
    if reason not in _STEPWISE_METRICS["fallback_reasons"]:
        _STEPWISE_METRICS["fallback_reasons"][reason] = 0
    _STEPWISE_METRICS["fallback_reasons"][reason] += 1
    
    # Log detailed error
    error_info = {
        "call_id": call_id,
        "reason": reason,
        "details": details,
        "timestamp": _STEPWISE_METRICS["total_calls"]
    }
    _STEPWISE_METRICS["error_details"].append(error_info)
    
    print(f"⚠️  STEP-WISE FALLBACK (Call {call_id}): {reason}")
    if details:
        print(f"   Details: {details}")
    print(f"   Falling back to original 'Wait' behavior")
    
    # Try to use the original "Wait" behavior
    try:
        # Default fallback continuation that mimics original behavior
        fallback_prompt = "\n\nWait, let me think about this more carefully:"
        continuation_tokens = lm.tok_encode(fallback_prompt)
        print(f"   ✅ Fallback successful: {fallback_prompt}")
        return True, continuation_tokens
    except Exception as e:
        print(f"   ❌ CRITICAL: Even fallback encoding failed: {e}")
        _log_error(call_id, "fallback_encoding", str(e))
        # Ultimate fallback: stop reasoning
        return False, []


def _log_error(call_id: int, category: str, message: str):
    """Log an error with detailed information."""
    global _STEPWISE_METRICS
    
    error_entry = {
        "call_id": call_id,
        "category": category,
        "message": message,
        "timestamp": _STEPWISE_METRICS["total_calls"]
    }
    _STEPWISE_METRICS["error_details"].append(error_entry)
    print(f"ERROR (Call {call_id}, {category}): {message}")


def parse_numbered_steps(text: str) -> List[str]:
    """
    Parse text to extract numbered steps like "Step 1:", "Step 2:", etc.
    
    Args:
        text: The generated text containing steps
        
    Returns:
        List of step contents (without the "Step N:" prefix), or None if parsing fails
    """
    if not text or not isinstance(text, str):
        print(f"⚠️  parse_numbered_steps: Invalid input text (type: {type(text)})")
        return None
    
    try:
        # Pattern to match numbered steps - more robust version
        # Matches "Step 1:", "Step 1.", "Step 1 -", etc.
        pattern = r'Step\s+(\d+)\s*[:.\-]\s*(.*?)(?=\n\s*Step\s+\d+\s*[:.\-]|\n\n|\Z)'
        
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        # Extract just the step content, not the numbers
        steps = []
        for step_num, content in matches:
            content = content.strip()
            if content:  # Only add non-empty steps
                steps.append(content)
        
        # Fallback: if no numbered steps found, try to find any step-like structure
        if not steps:
            print("   No 'Step N:' format found, trying alternative patterns...")
            
            # Try patterns like "1.", "1)", "(1)", etc.
            fallback_pattern = r'(?:^|\n)\s*(?:\(?(\d+)[\)\.]\s*)(.*?)(?=\n\s*\(?(?:\d+)[\)\.]\s*|\n\n|\Z)'
            fallback_matches = re.findall(fallback_pattern, text, re.DOTALL | re.MULTILINE)
            
            for step_num, content in fallback_matches:
                content = content.strip()
                if content and len(content) > 10:  # Only meaningful content
                    steps.append(content)
            
            if steps:
                print(f"   Found {len(steps)} steps using fallback pattern")
        
        return steps
        
    except Exception as e:
        print(f"⚠️  ERROR in parse_numbered_steps: {e}")
        return None


def calculate_step_uncertainties(steps: List[str], full_text: str, uncertainties: List[float], lm, is_first_call: bool = False) -> List[float]:
    """
    Calculate average uncertainty for each reasoning step with robust error handling.
    
    Args:
        steps: List of step content strings
        full_text: The complete generated text
        uncertainties: Per-token entropy values
        lm: HuggingFace model instance
        is_first_call: Whether this is the first call (for detailed logging)
        
    Returns:
        List of uncertainty values for each step, or None if calculation fails
    """
    if is_first_call:
        print(f"\n🔬 DETAILED UNCERTAINTY CALCULATION:")
        print(f"   Steps to analyze: {len(steps)}")
        print(f"   Entropy values available: {len(uncertainties)}")
        print(f"   Full text length: {len(full_text)} chars")
    
    # Input validation
    if not steps:
        print("⚠️  calculate_step_uncertainties: No steps provided")
        return None
    
    if not uncertainties:
        print("⚠️  WARNING: No entropy values available, using default uncertainty")
        return [0.5] * len(steps)
    
    if not isinstance(uncertainties, list):
        print(f"⚠️  WARNING: uncertainties not a list (type: {type(uncertainties)}), converting...")
        try:
            uncertainties = list(uncertainties)
        except Exception as e:
            print(f"⚠️  ERROR: Failed to convert uncertainties to list: {e}")
            return [0.5] * len(steps)
    
    # Validate entropy values
    try:
        valid_uncertainties = []
        for i, entropy in enumerate(uncertainties):
            if isinstance(entropy, (int, float)) and not math.isnan(entropy) and entropy >= 0:
                valid_uncertainties.append(float(entropy))
            else:
                print(f"⚠️  WARNING: Invalid entropy at position {i}: {entropy}")
                valid_uncertainties.append(0.5)  # Default fallback
        uncertainties = valid_uncertainties
    except Exception as e:
        print(f"⚠️  ERROR: Failed to validate uncertainties: {e}")
        return [0.5] * len(steps)
    
    step_uncertainties = []
    
    try:
        # More robust approach: encode the full text and map positions
        try:
            if hasattr(lm, 'tok_encode'):
                full_tokens = lm.tok_encode(full_text)
                if is_first_call:
                    print(f"   Full text tokenized to: {len(full_tokens)} tokens")
            else:
                print("⚠️  WARNING: HFLM missing tok_encode method")
                full_tokens = []
        except Exception as e:
            print(f"⚠️  WARNING: Failed to tokenize full text: {e}")
            full_tokens = []
        
        # Simple approach: divide uncertainties equally among steps
        if len(uncertainties) >= len(steps):
            uncertainties_per_step = len(uncertainties) // len(steps)
            remainder = len(uncertainties) % len(steps)
            
            for i, step in enumerate(steps):
                start_idx = i * uncertainties_per_step
                end_idx = start_idx + uncertainties_per_step
                
                # Distribute remainder uncertainties to first few steps
                if i < remainder:
                    end_idx += 1
                    start_idx += i
                else:
                    start_idx += remainder
                    end_idx += remainder
                
                if i == len(steps) - 1:  # Last step gets any remaining uncertainties
                    end_idx = len(uncertainties)
                
                step_uncertainties = uncertainties[start_idx:end_idx]
                if step_uncertainties:
                    try:
                        avg_uncertainty = sum(step_uncertainties) / len(step_uncertainties)
                        # Clamp to reasonable range
                        avg_uncertainty = max(0.0, min(1.0, avg_uncertainty))
                    except Exception as e:
                        print(f"⚠️  WARNING: Failed to calculate average for step {i+1}: {e}")
                        avg_uncertainty = 0.5
                else:
                    avg_uncertainty = sum(uncertainties) / len(uncertainties)  # Fallback
                
                step_uncertainties.append(avg_uncertainty)
                
                if is_first_call:
                    print(f"   Step {i+1}: tokens ~{start_idx}-{end_idx}, "
                          f"uncertainties: {len(step_uncertainties)}, "
                          f"avg_uncertainty: {avg_uncertainty:.4f}")
        else:
            # Not enough uncertainties - use available ones
            print(f"⚠️  WARNING: Only {len(uncertainties)} uncertainties for {len(steps)} steps")
            try:
                avg_entropy = sum(uncertainties) / len(uncertainties)
                step_uncertainties = [avg_entropy] * len(steps)
            except Exception as e:
                print(f"⚠️  ERROR: Failed to calculate average entropy: {e}")
                step_uncertainties = [0.5] * len(steps)
        
        # Final validation
        if len(step_uncertainties) != len(steps):
            print(f"⚠️  ERROR: Mismatch in uncertainty count. Expected {len(steps)}, got {len(step_uncertainties)}")
            return None
        
        return step_uncertainties
        
    except Exception as e:
        print(f"⚠️  ERROR in calculate_step_uncertainties: {e}")
        # Fallback: return average entropy for all steps
        try:
            avg_entropy = sum(uncertainties) / len(uncertainties) if uncertainties else 0.5
            return [avg_entropy] * len(steps)
        except Exception as fallback_e:
            print(f"⚠️  CRITICAL: Even fallback calculation failed: {fallback_e}")
            return None


def get_stepwise_metrics() -> Dict[str, Any]:
    """Get current metrics for step-wise uncertainty extraction."""
    global _STEPWISE_METRICS
    
    success_rate = 0.0
    if _STEPWISE_METRICS["total_calls"] > 0:
        success_rate = _STEPWISE_METRICS["successful_extractions"] / _STEPWISE_METRICS["total_calls"]
    
    # Calculate failure rate breakdown
    failure_rate_breakdown = {}
    total_failures = _STEPWISE_METRICS["parsing_failures"]
    for reason, count in _STEPWISE_METRICS["fallback_reasons"].items():
        failure_rate_breakdown[reason] = {
            "count": count,
            "percentage": (count / total_failures * 100) if total_failures > 0 else 0
        }
    
    return {
        **_STEPWISE_METRICS,
        "success_rate": success_rate,
        "failure_rate": 1.0 - success_rate,
        "avg_steps_per_call": (_STEPWISE_METRICS["steps_found_total"] / max(1, _STEPWISE_METRICS["total_calls"])),
        "failure_rate_breakdown": failure_rate_breakdown,
    }


def print_stepwise_metrics():
    """Print comprehensive summary of step-wise uncertainty metrics."""
    metrics = get_stepwise_metrics()
    print("\n" + "="*80)
    print("📊 STEP-WISE UNCERTAINTY COMPREHENSIVE METRICS")
    print("="*80)
    
    # Overall statistics
    print("🔢 OVERALL STATISTICS:")
    print(f"   Total calls: {metrics['total_calls']}")
    print(f"   Successful extractions: {metrics['successful_extractions']}")
    print(f"   Success rate: {metrics['success_rate']:.2%}")
    print(f"   Failure rate: {metrics['failure_rate']:.2%}")
    print(f"   Total steps found: {metrics['steps_found_total']}")
    print(f"   Steps revised: {metrics['steps_revised_total']}")
    print(f"   Avg steps per call: {metrics['avg_steps_per_call']:.1f}")
    
    # Failure breakdown
    if metrics["fallback_reasons"]:
        print(f"\n🚨 FAILURE BREAKDOWN:")
        for reason, info in metrics["failure_rate_breakdown"].items():
            print(f"   {reason}: {info['count']} times ({info['percentage']:.1f}%)")
    
    # Recent errors
    if metrics["error_details"]:
        print(f"\n🔍 RECENT ERROR DETAILS (last 5):")
        recent_errors = metrics["error_details"][-5:]
        for error in recent_errors:
            print(f"   Call {error['call_id']} [{error['category']}]: {error['message']}")
    
    # First call analysis
    if metrics["first_call_details"]:
        print(f"\n🎯 FIRST CALL ANALYSIS:")
        for key, value in metrics["first_call_details"].items():
            if key != "step_uncertainties":  # Skip long arrays
                print(f"   {key}: {value}")
    
    print("="*80)