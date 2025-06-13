import math
import torch
from typing import List, Optional, Union

from transformers import StoppingCriteriaList

from lm_eval.models.utils import stop_sequences_criteria
from lm_eval.budget_forcing.utils import convert_to_tensor
from lm_eval.budget_forcing.scaler_registry import get_scale_func


def generate_with_budget_forcing(
    hflm,
    input_ids: torch.Tensor,
    max_length: int,
    stopping_criteria: Optional[List[str]],
    pad_token_id: int,
    scale_func_name: str,
    max_tokens_thinking: Union[int, str] = "auto",
    thinking_start: str = "<|im_start|>think",
    thinking_end: str = "<|im_start|>answer",
    thinking_n_ignore: int = 0,
    thinking_n_ignore_str: str = "Wait",
    until_thinking: str = "<|im_start|>",
    until_thinking_2: Optional[str] = None,
    **generation_kwargs,
):
    """
    Generate text with budget forcing, forcing the model to think before answering.
    
    Args:
        hflm (HFLM): The Hugging Face language model to use for generation.
        input_ids (torch.Tensor): The input token ids as a tensor.
        max_length (int): The maximum length of the generated text (in tokens, including input).
        stopping_criteria (Optional[List[str]]): A list of stopping criteria to apply during generation.
        pad_token_id (int): The ID of the padding token used in the model.
        scale_func_name (str): The name of the scaling function to use for budget forcing.
        max_tokens_thinking (str): The maximum number of tokens to generate during the thinking phase.
        thinking_start (str): The token to indicate the start of the thinking phase.
        thinking_end (str): The token to indicate the end of the thinking phase.
        thinking_n_ignore (int): The number of times to ignore the thinking phase.
        thinking_n_ignore_str (str): The string token to use for ignoring the thinking phase.
        until_thinking (str): The token to stop generation until the thinking phase is reached.
        until_thinking_2 (Optional[str]): An additional token to stop generation until the thinking phase is reached.
        **generation_kwargs: Additional keyword arguments for the model's generate method.
        
    Returns:
        torch.Tensor: The generated text as a tensor of token ids.
    """
    thinking_n_ignore_str_tok = hflm.tok_encode(thinking_n_ignore_str)
    
    thinking_start_tok = hflm.tok_encode(thinking_start)
    thinking_end_tok = hflm.tok_encode(thinking_end)
    
    thinking_end_max = thinking_end + "\nFinal Answer:"
    thinking_end_max_tok = hflm.tok_encode(thinking_end_max)
    
    until_thinking = [until_thinking]
    if until_thinking_2 is not None:
        until_thinking.append(until_thinking_2)
    until_thinking_stop_seq = stop_sequences_criteria(
        hflm.tok_encode(until_thinking),
        hflm.tokenizer,
        input_ids.shape[0],
        input_ids.shape[1]
    )
    
    end_thinking_criterion = StoppingCriteriaList(
        until_thinking_stop_seq + stopping_criteria
    )
        
    until_thinking_tok = hflm.tok_encode(until_thinking)
    assert all((len(x) == 1 for x in until_thinking_tok)), "min_tokens_thinking only supports until_thinking tokens that are 1 token long"
    
    context = [
        req + thinking_start_tok 
        for req in input_ids.tolist()
    ]
    
    generation_kwargs.setdefault("min_tokens", 1)
    if max_tokens_thinking == "auto":
        # Leave 100 tokens for answer
        max_tokens = max_length - max([len(x) for x in context]) - len(thinking_start_tok) - len(thinking_end_max_tok) - 100
        print(f"Auto setting max_tokens_thinking to {max_tokens}")
    else:
        max_tokens = max_tokens_thinking
    
    scale_func = get_scale_func(
        scale_func_name, 
        scale_token=thinking_n_ignore_str_tok,
        **generation_kwargs
    )
    indices = list(range(len(context)))
    for i in range(thinking_n_ignore + 1):
        input_ids = convert_to_tensor(
            context,
            pad_token_id=pad_token_id,
            device=hflm.device,
        )[indices]
        
        sequences, entropies = _generate_with_entropy(
            hflm.model,
            input_ids=input_ids[indices],
            pad_token_id=pad_token_id,
            stopping_criteria=end_thinking_criterion,
            **generation_kwargs,
        )
        
        new_indices = []
        for idx, seq, entropy in zip(indices, sequences, entropies):
            keep_scaling, scale_token = scale_func(
                iteration=i, seq=seq, entropies=entropy, hflm=hflm,
            )
                
            if keep_scaling and len(seq) + len(scale_token) < max_tokens:
                new_indices.append(idx)
                context[idx] = seq.tolist() + scale_token
        indices = new_indices
        
    for i, output in enumerate(context):
        if len(output) >= max_tokens:
            context[i] = output + thinking_end_max_tok
        else:
            context[i] = output + thinking_end_tok
    
    input_ids = convert_to_tensor(
        context,
        pad_token_id=pad_token_id,
        device=hflm.device,
    )
    return hflm.model.generate(
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=pad_token_id,
        stopping_criteria=stopping_criteria,
        use_cache=True,
        **generation_kwargs,
    )
    
    
def _generate_with_entropy(
    model,
    input_ids: torch.Tensor,
    pad_token_id: int,
    stopping_criteria: StoppingCriteriaList,
    max_tokens: int = 32768,
    normalize: bool = True,
    **generation_kwargs,
):
    """
    Generate text using a Hugging Face model, returning entropy values for each generated token.
    
    Args:
        model: Hugging Face language model to use for generation.
        input_ids (torch.Tensor): The input token ids as a tensor. 
        pad_token_id (int): The ID of the padding token used in the model.
        stopping_criteria (StoppingCriteriaList): A list of stopping criteria to apply during generation.
        max_tokens (int): The maximum length of the generated text.
        normalize (bool): Whether to normalize the entropy values to a range of [0, 1].
        **generation_kwargs: Additional keyword arguments for the model's generate method.
        
    Returns:
        List[List[int]]: The generated sequences as a list of token IDs.
        List[List[float]]: The entropy values for each generated token.
    """
    outputs = model.generate(
        input_ids=input_ids,
        pad_token_id=pad_token_id,
        max_length=max_tokens,
        stopping_criteria=stopping_criteria,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        **generation_kwargs,
    )
    
    entropies = []
    for score in outputs.scores:
        for logits in score:
            mask = logits != float("-inf")
            filtered = logits[mask]
            
            entropy = -torch.sum(filtered.softmax(dim=-1) * filtered.log_softmax(dim=-1), dim=-1).item()
            
            if normalize:
                max_entropy = math.log(mask.sum().item())
                entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                
            entropies.append(entropy)
    
    return outputs.sequences, entropies