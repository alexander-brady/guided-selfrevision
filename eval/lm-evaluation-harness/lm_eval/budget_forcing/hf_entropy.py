import torch
import math
from typing import Optional, List


def generate_with_entropy(
    model,
    input_ids: List[int],
    max_length: int,
    pad_token_id: int,
    stopping_criteria: Optional[List[str]] = None,
    normalize: bool = True,
    **generation_kwargs,
):
    '''
    Generate text using a Hugging Face model with entropy-based stopping criteria for edging.
    
    Args:
        model: Hugging Face language model to use for generation.
        input_ids (List[int]): The input input_ids as a list of token IDs.
        max_length (int): The maximum length of the generated text.
        pad_token_id (int): The ID of the padding token used in the model.
        stopping_criteria (Optional[List[str]]): A list of strings that, if generated, will stop the generation.
        normalize (bool): Whether to normalize the entropy values to a range of [0, 1].
        **generation_kwargs: Additional keyword arguments for the model's generate method.
        
    Returns:
        str: The generated text.
    '''
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        **generation_kwargs,
    )
    
    entropies = []
    for logits in outputs.scores:
        mask = logits != float("-inf")
        filtered = logits[mask]
        
        entropy = -torch.sum(filtered.softmax(dim=-1) * filtered.log_softmax(dim=-1), dim=-1).item()
        
        if normalize:
            max_entropy = math.log(mask.sum().item())
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
        entropies.append(entropy)
    
    return outputs.sequences, entropies


def generate_entropy_thresholding(
    model,
    input_ids: List[int],
    scale_token: List[int],
    threshold: float,
    last_k: int,
    max_steps: int,
    max_length: int,
    pad_token_id: int,
    decay_factor: float = 1.0,
    stopping_criteria: Optional[List[str]] = None,
    **generation_kwargs,
):
    """
    Generate text using a Hugging Face model, applying budget forcing if the average entropy of the last k tokens exceeds a threshold.
    
    Args:
        model: The Hugging Face language model to use for generation.
        input_ids (List[int]): The input context as a list of token IDs.
        scale_token (List[int]): The token id(s) to begin budget forcing.
        threshold (float): The entropy threshold. If the entropy is higher, the model will continue reasoning.
        last_k (int): The number of last tokens to consider for entropy calculation. -1 means all tokens.
        max_steps (int): The maximum number of steps to take before stopping.
        max_length (int): The maximum length of the generated text.
        pad_token_id (int): The ID of the padding token used in the model.
        decay_factor (float): A factor to lower the threshold over steps (1.0 means no decay).
        stopping_criteria (Optional[List[str]]): A list of strings that, if generated, will stop the generation.
        **generation_kwargs: Additional keyword arguments for the model's generate method.
    """
    sequences, entropies = generate_with_entropy(
        model,
        input_ids,
        max_length,
        pad_token_id,
        stopping_criteria=stopping_criteria,
        **generation_kwargs,
    )
    
    for i in range(max_steps):
        max_length -= len(entropies) - len(scale_token)
        if max_length <= 0:
            break
                
        last_k_entropies = entropies[-last_k:] if last_k != -1 else entropies
        avg_entropy = sum(last_k_entropies) / len(last_k_entropies) if last_k_entropies else 0.0
        
        # Stop budget scaling if low uncertainty
        if avg_entropy < 1 - (threshold * (decay_factor ** i)):
            break
               
        input_ids = sequences[0].tolist() + scale_token
        sequences, entropies = generate_with_entropy(
            model,
            input_ids,
            max_length,
            pad_token_id,
            stopping_criteria=stopping_criteria,
            **generation_kwargs,
        )
    
    return model.generate(
        input_ids=sequences[0],
        max_length=32768,
        pad_token_id=pad_token_id,
        skip_special_tokens=False,
        stopping_criteria=stopping_criteria,
        use_cache=True,
        **generation_kwargs,
    )