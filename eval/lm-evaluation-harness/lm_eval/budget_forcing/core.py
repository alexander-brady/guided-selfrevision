import math
import torch
from typing import List, Optional, Union

from vllm import SamplingParams

from lm_eval.budget_forcing.scaler_registry import get_scale_func


def generate_with_budget_forcing(
    lm,
    requests: List[List[int]],
    max_tokens: int,
    stop_sequences: Optional[List[str]],
    scale_func_name: str,
    debug: bool = False,
    max_tokens_thinking: Union[int, str] = "auto",
    thinking_start: str = "<|im_start|>think",
    thinking_end: str = "<|im_start|>answer",
    thinking_n_ignore: int = 0,
    thinking_n_ignore_str: str = "Wait",
    until_thinking: str = "<|im_start|>",
    until_thinking_2: Optional[str] = None,
    **kwargs,
):
    """
    Generate text with budget forcing, forcing the model to think before answering.
    
    Args:
        lm: The language model wrapper to use for generation.
        requests (List[List[int]]): A list of input sequences as token ids to generate from.
        max_tokens (int): The maximum length of the generated text (including input).
        stop_sequences (Optional[List[str]]): A list of stopping criteria to apply during generation.
        pad_token_id (int): The ID of the padding token used in the model.
        scale_func_name (str): The name of the scaling function to use for budget forcing.
        max_tokens_thinking (str): The maximum number of tokens to generate during the thinking phase.
        thinking_start (str): The token to indicate the start of the thinking phase.
        thinking_end (str): The token to indicate the end of the thinking phase.
        thinking_n_ignore (int): The number of times to ignore the thinking phase.
        thinking_n_ignore_str (str): The string token to use for ignoring the thinking phase.
        until_thinking (str): The token to stop generation until the thinking phase is reached.
        until_thinking_2 (Optional[str]): An additional token to stop generation until the thinking phase is reached.
        **kwargs: Additional keyword arguments for the model's scale function or generate method.
        
    Returns:
        torch.Tensor: The generated text as a tensor of token ids.
    """
    default_scale_tok = lm.tok_encode(thinking_n_ignore_str)
    
    thinking_start_tok = lm.tok_encode(thinking_start)
    thinking_end_tok = lm.tok_encode(thinking_end)
    
    thinking_end_max = thinking_end + "\nFinal Answer:"
    thinking_end_max_tok = lm.tok_encode(thinking_end_max)
    
    until_thinking = [until_thinking] + (
        [until_thinking_2] if until_thinking_2 is not None else []
    )
    
    context = [
        req + thinking_start_tok 
        for req in requests
    ]
    
    kwargs.setdefault("min_length", 1)
    if max_tokens_thinking == "auto":
        # Leave 100 tokens for answer
        max_tokens -= max([len(x) for x in context]) - len(thinking_start_tok) - len(thinking_end_max_tok) - 100
        print(f"Auto setting max_tokens_thinking to {max_tokens}")
    else:
        max_tokens = max_tokens_thinking
    
    # Create scale function with extracted parameters
    scale_func = get_scale_func(
        scale_func_name, 
        scale_token=default_scale_tok,
        **kwargs
    )
    
    indices = list(range(len(context)))
    for i in range(thinking_n_ignore + 1):
        if not indices:
            break # No more sequences to process
        
        requests = [context[i] for i in indices]
        outputs, uncertainties = _generate_with_uncertainty(
            lm,
            prompt_token_ids=requests,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **kwargs
        )
        
        new_indices = []
        for idx, out, unc in zip(indices, outputs, uncertainties):
            full_sequence = context[idx] + out.outputs[0].token_ids
            keep_scaling, scale_token = scale_func(
                iteration=i,
                seq=full_sequence,
                uncertainties=unc,
                hflm=lm
            )
                
            if keep_scaling and len(full_sequence) + len(scale_token) < max_tokens:
                context[idx] = full_sequence + scale_token
                new_indices.append(idx)
            else:
                context[idx] = full_sequence
                
        indices = new_indices
        
    for i, output in enumerate(context):
        if len(output) >= max_tokens:
            context[i] = output + thinking_end_max_tok
        else:
            context[i] = output + thinking_end_tok

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        stop=stop_sequences,
        **kwargs
    )
    return lm.model.generate(
        prompt_token_ids=context,
        sampling_params=sampling_params,
        use_tqdm=lm.batch_size == "auto"
    )
    
    
def _generate_with_uncertainty(
    lm,
    prompt_token_ids: List[List[int]],
    max_tokens: int,
    stop_sequences: Optional[List[str]],
    **generation_kwargs,
):
    """
    Generate text using a Hugging Face model, returning entropy values for each generated token.
    
    Args:
        lm: Language model wrapper to use for generation.
        prompt_token_ids (List[List[int]]): The input token ids as to generate from.
        max_tokens (int): The maximum length of the generated text. 
        stop_sequences (Optional[List[str]]): A list of stopping criteria to apply during generation.
        normalize (bool): Whether to normalize the entropy values to a range of [0, 1].
        **generation_kwargs: Additional keyword arguments for the model's generate method.
        
    Returns:
        List[vllm.RequestOutput]: The generated sequences as a list of tensors.
        List[List[float]]: The entropy values for each sequence.
    """
    generation_kwargs = generation_kwargs.copy()
    generation_kwargs["logprobs"] = generation_kwargs.get("logprobs", 1)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        stop=stop_sequences,
        **generation_kwargs
    )
            
    outputs = lm.model.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=lm.batch_size == "auto"
    )
    
    uncertainties = []
    for out in outputs:
        gen = out.outputs[0]
        
        logprobs = gen.logprobs
        sample_uncertainties = []
        
        for i, item in enumerate(logprobs):            
            logprob = item.logprob
        
            # Uncertainty = (1 - exp(logprob))
            p = math.exp(min(logprob, 0))
            sample_uncertainties.append(1.0 - p)
        
        uncertainties.append(sample_uncertainties)
    
    return outputs, uncertainties