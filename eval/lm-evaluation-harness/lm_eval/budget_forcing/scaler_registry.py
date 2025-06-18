from functools import partial
from typing import Callable, List
import traceback
import os

from lm_eval.budget_forcing.scalers import entropy_thresholding, step_wise_uncertainty_driven


def get_scale_func(func_name: str, scale_token: List[int], **kwargs) -> Callable:
    """
    Returns a scaling function based on the function name.
    
    GRACEFUL FALLBACK: If the requested scaling function fails to initialize or 
    execute, this will gracefully fall back to the default scaling behavior.

    Args:
        func_name (str): The name of the scaling function to return.
        scale_token (List[int]): The token to use for scaling.
        **kwargs: Additional keyword arguments for the scaling function.

    Returns:
        Callable: The scaling function with built-in error handling.
    """
    def default_scale_func(*_, **__):
        """
        Default scale function that always continues with the provided scale token.
        
        Returns:
            tuple: (True, scale_token) to always continue budget forcing.
        """
        return True, scale_token

    def safe_wrapper(scale_func, func_name: str, **func_kwargs):
        """
        Wrapper that adds error handling to any scaling function.
        
        Args:
            scale_func: The actual scaling function to wrap
            func_name: Name of the function for error reporting
            **func_kwargs: Arguments to pass to the scaling function
            
        Returns:
            Callable: Error-wrapped scaling function
        """
        def wrapped_scale_func(iteration, seq, entropies, hflm):
            try:
                # Attempt to call the scaling function
                result = scale_func(
                    iteration=iteration,
                    seq=seq,
                    entropies=entropies,
                    hflm=hflm,
                    **func_kwargs
                )
                
                # Validate the result format
                if not isinstance(result, tuple) or len(result) != 2:
                    error_msg = f"SCALE FUNCTION ERROR: {func_name} returned invalid format: {result}. Expected: (bool, List[int]), got: {type(result)}"
                    print(f"⚠️  {error_msg}")
                    if os.getenv("RAISE_ON_SCALER_ERROR", "0") == "1":
                        raise ValueError(error_msg)
                    print(f"   Falling back to default behavior")
                    return True, scale_token
                
                continue_scaling, tokens = result
                
                # Validate the return values
                if not isinstance(continue_scaling, bool):
                    print(f"⚠️  SCALE FUNCTION WARNING: {func_name} returned non-bool continue_scaling: {continue_scaling}")
                    continue_scaling = bool(continue_scaling)
                
                if tokens is None:
                    tokens = scale_token
                elif not isinstance(tokens, list):
                    print(f"⚠️  SCALE FUNCTION WARNING: {func_name} returned non-list tokens: {tokens}")
                    try:
                        tokens = list(tokens) if tokens else scale_token
                    except:
                        tokens = scale_token
                
                return continue_scaling, tokens
                
            except Exception as e:
                error_msg = f"SCALE FUNCTION EXCEPTION in {func_name}: {e}"
                print(f"⚠️  {error_msg}")
                print(f"   Iteration: {iteration}")
                print(f"   Seq type: {type(seq)}, length: {len(seq) if hasattr(seq, '__len__') else 'unknown'}")
                print(f"   Entropies type: {type(entropies)}, length: {len(entropies) if hasattr(entropies, '__len__') else 'unknown'}")
                print(f"   HFLM type: {type(hflm)}")
                print(f"   Traceback (last 3 lines):")
                
                # Print last 3 lines of traceback for debugging
                tb_lines = traceback.format_exc().strip().split('\n')
                for line in tb_lines[-3:]:
                    print(f"     {line}")
                
                # Check if we should raise instead of falling back
                if os.getenv("RAISE_ON_SCALER_ERROR", "0") == "1":
                    raise RuntimeError(f"Scaler error in {func_name}: {e}") from e
                
                print(f"   Falling back to default behavior")
                return True, scale_token
        
        return wrapped_scale_func

    # Handle entropy_thresholding with error checking
    if func_name == "entropy_thresholding":
        try:
            threshold = float(kwargs.pop("threshold", 0.5))
            decay_factor = float(kwargs.pop("decay_factor", 1.0))
            last_k = float(kwargs.pop("last_k", -1))
            
            print(f"🔧 Initializing entropy_thresholding:")
            print(f"   threshold={threshold}, decay_factor={decay_factor}, last_k={last_k}")
            
            # Validate parameters
            if threshold < 0 or threshold > 1:
                print(f"⚠️  WARNING: Invalid threshold {threshold}, using 0.5")
                threshold = 0.5
            
            if decay_factor <= 0:
                print(f"⚠️  WARNING: Invalid decay_factor {decay_factor}, using 1.0")
                decay_factor = 1.0
            
            def entropy_thresholding_func(iteration, seq, entropies, hflm):
                return entropy_thresholding(
                    threshold=threshold,
                    decay_factor=decay_factor,
                    last_k=last_k,
                    iteration=iteration,
                    seq=seq,
                    entropies=entropies,
                    hflm=hflm,
                    scale_token=scale_token,
                )
            
            return safe_wrapper(entropy_thresholding_func, "entropy_thresholding")
            
        except Exception as e:
            print(f"⚠️  ERROR initializing entropy_thresholding: {e}")
            print(f"   Falling back to default scale function")
            return default_scale_func
    
    # Handle step_wise_uncertainty_driven with error checking
    if func_name == "step_wise_uncertainty_driven":
        try:
            step_selection_strategy = str(kwargs.pop("step_selection_strategy", "highest_uncertainty"))
            max_steps = int(kwargs.pop("max_steps", 10))
            use_min_uncertainty_filter = bool(kwargs.pop("use_min_uncertainty_filter", False))
            min_step_uncertainty = float(kwargs.pop("min_step_uncertainty", 0.3))
            
            print(f"🔧 Initializing step_wise_uncertainty_driven:")
            print(f"   step_selection_strategy={step_selection_strategy}")
            print(f"   max_steps={max_steps}")
            print(f"   use_min_uncertainty_filter={use_min_uncertainty_filter}")
            print(f"   min_step_uncertainty={min_step_uncertainty}")
            
            # Validate parameters
            if step_selection_strategy not in ["highest_uncertainty", "lowest_uncertainty", "random"]:
                print(f"⚠️  WARNING: Invalid strategy {step_selection_strategy}, using 'highest_uncertainty'")
                step_selection_strategy = "highest_uncertainty"
                
            if max_steps <= 0:
                print(f"⚠️  WARNING: Invalid max_steps {max_steps}, using 10")
                max_steps = 10
                
            if min_step_uncertainty < 0 or min_step_uncertainty > 1:
                print(f"⚠️  WARNING: Invalid min_step_uncertainty {min_step_uncertainty}, using 0.3")
                min_step_uncertainty = 0.3
            
            def step_wise_func(iteration, seq, entropies, hflm):
                return step_wise_uncertainty_driven(
                    step_selection_strategy=step_selection_strategy,
                    max_steps=max_steps,
                    use_min_uncertainty_filter=use_min_uncertainty_filter,
                    min_step_uncertainty=min_step_uncertainty,
                    iteration=iteration,
                    seq=seq,
                    entropies=entropies,
                    hflm=hflm,
                )
            
            return safe_wrapper(step_wise_func, "step_wise_uncertainty_driven")
            
        except Exception as e:
            print(f"⚠️  ERROR initializing step_wise_uncertainty_driven: {e}")
            print(f"   Falling back to default scale function")
            return default_scale_func

    # Handle unknown function names
    if func_name and func_name != "default":
        print(f"⚠️  WARNING: Unknown scaling function '{func_name}', using default")
    
    print(f"🔧 Using default scale function (always continue with provided token)")
    return default_scale_func