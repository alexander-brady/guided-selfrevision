from functools import partial
from typing import Callable, List

from lm_eval.budget_forcing.scalers import entropy_thresholding


def get_scale_func(func_name: str, scale_token: List[int], **kwargs) -> Callable:
    """
    Get the scaling function based on the function name and scale token.
    
    Args:
        func_name (str): Name of the scaling function.
        scale_token (List[int]): The token to use for scaling.
        **kwargs: Additional arguments for the scaling function.
        
    Returns:
        Callable: The scaling function, returns (bool, List[int]) indicating whether to scale and the scale token id.
    """
    def default_scale_func(*_, **__):
        """Always scale with default scale token."""
        return True, scale_token
    
    if func_name == "entropy_thresholding":
        threshold = kwargs.pop("threshold", 0.001)
        decay_factor = kwargs.pop("decay_factor", 1.0)
        last_k = kwargs.pop("last_k", -1)
        print(f"Using entropy thresholding with threshold={threshold}, decay_factor={decay_factor}, last_k={last_k}")
        return partial(
            entropy_thresholding,
            scale_token=scale_token,
            threshold=threshold,
            decay_factor=decay_factor,
            last_k=last_k
        )
    
    return default_scale_func