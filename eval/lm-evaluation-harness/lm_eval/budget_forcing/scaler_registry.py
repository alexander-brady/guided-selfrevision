from functools import partial, wraps
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
        Callable: The scaling function.
    """
    def default_scale_func(*_, **__):
        """Always scale with default scale token."""
        return True, scale_token
    
    if func_name == "entropy_thresholding":
        return partial(
            entropy_thresholding,
            scale_token=scale_token,
            threshold=kwargs.pop("threshold", 0.5),
            decay_factor=kwargs.pop("decay_factor", 1.0),
            last_k=kwargs.pop("last_k", -1),
        )
    
    return default_scale_func


def should_scale_only(func):
    """
    Decorator for scale functions that only return a boolean 
    indicating whether to scale.
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        scale_token = kwargs.pop('scale_token')
        return scale_token, func(*args, **kwargs)
    
    return wrapper


def scale_token_only(func):
    """
    Decorator for scale functions that only return the scale token.
    Returns true, thus indicating that scaling should occur.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return True, func(*args, **kwargs)
    
    return wrapper