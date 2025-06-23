import numpy as np
from typing import List, Tuple, Callable, Optional, TYPE_CHECKING
from budget_forcing.scalers.util import scale_token_only

if TYPE_CHECKING:
    from lm_eval.models.vllm_causallms import VLLM
    
    
@scale_token_only
def uncertainty_driven_reevaluation(
    scale_token: List[int],
    iteration: int,
    tokens: List[int],
    uncertainties: List[float],
    lm: 'VLLM',
    min_threshold: float = -1.0,
    ablation: Optional[str] = None
) -> List[int]:
    """
    Prompts model to continue reasoning on uncertain sequences.
    
    Args:
        scale_token (List[int]): The token to use for scaling.
        iteration (int): The current thinking iteration.
        tokens (List[int]): The sequence of tokens, i.e. context and generated tokens.
        uncertainties (List[float]): The uncertainty for each generated token of the sequence (higher is more certain).
        lm (VLLM): The LM instance with the model and tokenizer.
        min_threshold (float): The minimum uncertainty threshold to consider a segment for reevaluation.
    Returns:
        List[int]: The scale token to continue reasoning with.
    """
    generated_tokens = tokens[-len(uncertainties):]
    segments = _split_uncertainties(uncertainties, generated_tokens, lm.tokenizer)
    
    if not segments:
        return scale_token # No segments found, return the scale token as is.

    uncertain_utterance, max_uncertainty = max(segments, key=lambda x: x[1])
    
    match ablation:
        case 'random': # Reevaluate a random segment.
            uncertain_utterance, max_uncertainty = segments[np.random.randint(len(segments))]
        case 'last': # Always reevaluate the last segment.
            uncertain_utterance, max_uncertainty = segments[-1]
        case 'third': # Take the third most uncertain segment.
            sorted_segments = sorted(segments, key=lambda x: x[1], reverse=True)
            uncertain_utterance, max_uncertainty = sorted_segments[2] if len(sorted_segments) > 2 else sorted_segments[0]
        case 'certain': # Use the most certain segment.
            uncertain_utterance, max_uncertainty = min(segments, key=lambda x: x[1]) 
        case _: # No ablation, use the most uncertain segment.
            uncertain_utterance, max_uncertainty = max(segments, key=lambda x: x[1])
    
    if max_uncertainty < min_threshold:
        return [] # Stop scaling if the model is confident enough.
    
    uncertain_utterance = ' '.join(uncertain_utterance.split()[:6]).strip() # Limit to first 6 words for brevity.
    continuation_prompt = f"Wait, I am not sure that '{uncertain_utterance}...' is correct. Let me reevaluate my answer."
    continuation_tokens = lm.tokenizer.encode(continuation_prompt, add_special_tokens=False)
    
    return continuation_tokens


def _split_uncertainties(
    uncertainties: List[float],
    tokens: List[str],
    tokenizer,
    boundary_tokens: List[str] = [
        '\\n', '\\r', '\\t', ',', '.', '!', '?', ';', ':',
    ],
    aggregation_fn: Callable[[list[float]], float] = np.mean,
    min_segment_length: int = 10
) -> List[Tuple[str, float]]:
    """
    Splits the uncertainties into segments based on the tokens.
    
    Args:
        uncertainties (List[float]): The uncertainty for each generated token of the sequence (higher is more certain).
        tokens (List[int]): The sequence of tokens, i.e. context and generated tokens.
        tokenizer: The tokenizer used to process the tokens.
        boundary_tokens (List[str]): Tokens that define boundaries for splitting sequences.
        aggregation_fn (Callable): Function to aggregate uncertainties within each segment.
        min_segment_length (int): Minimum length of a segment to be considered valid (otherwise will be concatenated with the next segment).
    
    Returns:
        List[Tuple[str, float]]: A list of tuples of a text utterance and its associated uncertainty.
    """
    assert len(uncertainties) == len(tokens), "Uncertainties and tokens must have the same length."
    text_sections, unc_sections = [[]], [[]]

    for token, uncertainty in zip(tokens, uncertainties):
        decoded = tokenizer.decode([token])
        if len(text_sections[-1]) >= min_segment_length and any(bt in decoded for bt in boundary_tokens):
            if any(c.isalpha() for c in decoded):
                text_sections[-1].append(token)
                unc_sections[-1].append(uncertainty)
                
            text_sections.append([])
            unc_sections.append([])
        else:
            text_sections[-1].append(token)
            unc_sections[-1].append(uncertainty)
            
        text_segments = [
            tokenizer.decode(section, skip_special_tokens=True)
            for section in text_sections if len(section) >= min_segment_length
        ]
        unc_values = [ # Apply aggregation function to first `min_segment_length` uncertainties
            aggregation_fn(unc) for unc in unc_sections[:min_segment_length]
            if len(unc) >= min_segment_length
        ]

        return list(zip(text_segments, unc_values))