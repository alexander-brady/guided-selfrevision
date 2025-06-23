from functools import partial
from typing import Callable, List

import budget_forcing.scalers as scalers


def get_scale_func(func_name: str, scale_token: List[int], **kwargs) -> Callable:
    """
    Returns a scaling function based on the function name.
    
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
    
    match func_name:
        case "entropy_thresholding":
            return partial(
                scalers.entropy_thresholding,
                scale_token=scale_token,
                threshold=kwargs.pop("threshold", 0.5),
                decay_factor=kwargs.pop("decay_factor", 1.0),
                last_k=kwargs.pop("last_k", -1)                     
            )
        case "uncertainty_driven_reevaluation":
            ablation= kwargs.pop("ablation", None)
            if ablation:
                print(f"Using ablation: {ablation}")                
            return partial(
                scalers.uncertainty_driven_reevaluation,
                scale_token=scale_token,
                min_threshold=kwargs.pop("min_threshold", -1.0),
                ablation=ablation
            )
        case "step_wise_uncertainty_driven":
            step_selection_strategy = str(kwargs.pop("step_selection_strategy", "highest_uncertainty"))
            max_steps = int(kwargs.pop("max_steps", 10))
            use_min_uncertainty_filter = bool(kwargs.pop("use_min_uncertainty_filter", False))
            min_step_uncertainty = float(kwargs.pop("min_step_uncertainty", 0.3))
            return partial(
                scalers.step_wise_uncertainty_driven,
                scale_token=scale_token,
                step_selection_strategy=step_selection_strategy,
                max_steps=max_steps,
                use_min_uncertainty_filter=use_min_uncertainty_filter,
                min_step_uncertainty=min_step_uncertainty
            )
        case _:
            return default_scale_func