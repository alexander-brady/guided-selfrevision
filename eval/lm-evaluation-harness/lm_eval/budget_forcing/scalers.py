import math
from typing import List

from functools import wraps


def should_scale_only(func):
    """
    Decorator for scale functions that only return a boolean 
    indicating whether to scale.
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        scale_token = kwargs.pop('scale_token')
        return func(*args, **kwargs), scale_token
    
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


@should_scale_only
def entropy_thresholding(
    threshold: float,
    decay_factor: float,
    last_k: float,
    iteration: int,
    seq: List[int],
    entropies: List[float],
    hflm,
) -> bool:
    """
    Determine whether to scale the sequence based on entropy and a threshold.
    
    Args:
        threshold (float): The threshold for scaling.
        decay_factor (float): Factor to decay the threshold over time. 
        last_k (float): Number of last tokens to consider for scaling. If > 1, it considers the last k tokens; if < 1, it considers the last k percent of tokens. If -1, it considers all generated tokens.
        iteration (int): The current thinking iteration.
        seq (List[int]): The sequence of tokens.
        entropies (List[float]): The entropy for each generated token of the sequence.
        scale_token (List[int]): The token to use for scaling.
        hflm (HFLM): The huggingface LM instance with the model and tokenizer.
    
    Returns:
        bool: True if the model should continue reasoning.
        List[int]: The scale token to continue reasoning with.
    """ 
    if 1 > last_k > 0:
        last_k = math.ceil(last_k * len(seq))
        
    if last_k > -1:
        entropies = entropies[-last_k:]
        
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    return avg_entropy < (1 - (threshold * (decay_factor ** iteration)))


def uncertainty_driven_pondering(
    min_seq_length: int,
    threshold: int,
    iteration: int,
    seq: List[int],
    entropies: List[float],
    hflm,     
):
    """
    Targeted reevaluation of low-certainty statements.
    
    Args:
        seq (List[int]): The sequence of tokens.
        entropies (List[float]): The entropy for each generated token of the sequence.
        scale_token (List[int]): The token to reevaluation_statementuse for scaling.
        hflm (HFLM): The huggingface LM instance with the model and tokenizer.
    
    Returns:
        bool: True if the model should continue reasoning.
        List[int]: The scale token to continue reasoning with.
    """ 
    pass
    # tokens_to_reevaluate = []
    # hflm.tokens_to_str