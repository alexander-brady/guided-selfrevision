import math
from typing import List, TYPE_CHECKING

from budget_forcing.scalers.util import should_scale_only

if TYPE_CHECKING:
    from lm_eval.models.vllm_causallms import VLLM

@should_scale_only
def entropy_thresholding(
    threshold: float,
    decay_factor: float,
    last_k: float,
    iteration: int,
    tokens: List[int],
    uncertainties: List[float],
    lm: 'VLLM'
) -> bool:
    """
    Determine whether to scale the sequence based on entropy and a threshold.
    
    Args:
        threshold (float): The threshold for scaling.
        decay_factor (float): Factor to decay the threshold over time. 
        last_k (float): Number of last tokens to consider for scaling. If > 1, it considers the last k tokens; if < 1, it considers the last k percent of tokens. If -1, it considers all generated tokens.
        iteration (int): The current thinking iteration.
        tokens (List[int]): The sequence of tokens.
        uncertainties (List[float]): The entropy for each generated token of the sequence.
        scale_token (List[int]): The token to use for scaling.
        hflm (HFLM): The huggingface LM instance with the model and tokenizer.
    
    Returns:
        bool: True if the model should continue reasoning.
        List[int]: The scale token to continue reasoning with.
    """ 
    if 1 > last_k > 0:
        last_k = math.ceil(last_k * len(tokens))
        
    if last_k > -1:
        uncertainties = uncertainties[-last_k:]
        
    avg_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
    
    print(f"Average Uncertainty: {avg_uncertainty}, {iteration=}")
    
    return avg_uncertainty < threshold * (decay_factor ** iteration)