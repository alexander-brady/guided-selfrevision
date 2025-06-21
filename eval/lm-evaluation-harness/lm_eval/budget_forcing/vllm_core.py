import math
import os
from typing import List, Optional

from vllm import SamplingParams

from lm_eval.budget_forcing.scaler_registry import get_scale_func
from lm_eval.utils import eval_logger


def _clean_vllm_kwargs(kwargs: dict, debug: bool = False) -> dict:
    """Clean kwargs to only include valid vLLM SamplingParams arguments.
    
    Args:
        kwargs: Dictionary of generation parameters
        debug: Whether to log the cleaning process
        
    Returns:
        Cleaned kwargs dict suitable for SamplingParams
    """
    cleaned = kwargs.copy()
    
    # Parameters that should be removed before passing to vLLM SamplingParams
    invalid_params = [
        # HF-specific parameters
        "do_sample",
        # Thinking/reasoning parameters (handled separately in thinking logic)
        "max_tokens_thinking", "thinking_n_ignore", "thinking_start", "thinking_end",
        "until_thinking", "until_thinking_2", "thinking_n_ignore_str", 
        "rejection_sample", "min_tokens_thinking",
        # Step-wise uncertainty parameters (handled in scaler)
        "scale_func_name", "step_selection_strategy", "max_steps",
        "use_min_uncertainty_filter", "min_step_uncertainty",
        "threshold", "decay_factor", "last_k",
        # Debug parameter (handled separately)
        "debug"
    ]
    
    removed = []
    for param in invalid_params:
        if param in cleaned:
            cleaned.pop(param)
            removed.append(param)
    
    if debug and removed:
        eval_logger.info(f"   Removed invalid vLLM params: {removed}")
    
    return cleaned


def _generate_with_uncertainty_vllm(
    llm,
    prompt_token_ids: List[List[int]],
    max_tokens: int,
    stop_sequences: Optional[List[str]],
    debug: bool = False,
    **generation_kwargs,
):
    """Helper that calls vLLM generate with log-probs enabled and
    converts them into per-token uncertainty values (1 - p(token)).

    Returns
    -------
    outputs : list[vllm.RequestOutput]
    uncertainties : list[list[float]]
        Per sample, per generated token uncertainty in [0,1].
    """
    if debug:
        eval_logger.info(f"🔬 _generate_with_uncertainty_vllm: {len(prompt_token_ids)} prompts, max_tokens={max_tokens}")
        eval_logger.info(f"   Generation kwargs: {generation_kwargs}")
    
    # Ensure we will receive log-probs of the sampled token.
    generation_kwargs = generation_kwargs.copy()
    generation_kwargs["logprobs"] = generation_kwargs.get("logprobs", 1)

    # Clean kwargs to only include valid SamplingParams arguments
    clean_kwargs = _clean_vllm_kwargs(generation_kwargs, debug=debug)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        stop=stop_sequences,
        **clean_kwargs,
    )

    if debug:
        eval_logger.info(f"   SamplingParams: logprobs={sampling_params.logprobs}, temperature={getattr(sampling_params, 'temperature', 'N/A')}")
    
    outputs = llm.model.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=True if llm.batch_size == "auto" else False,
    )

    if debug:
        eval_logger.info(f"   Generated {len(outputs)} outputs")
    
    per_sample_uncert = []
    for out in outputs:
        # vLLM guarantees one output per prompt when temperature == 0 / greedy.
        gen = out.outputs[0]
        
        # Access logprobs like in the user's example - they can be on RequestOutput or CompletionOutput
        token_logprobs = getattr(gen, "logprobs", None)
        if token_logprobs is None:
            # Fallback: check if logprobs are on the RequestOutput itself
            token_logprobs = getattr(out, "logprobs", None)
        
        if debug:
            eval_logger.info(f"   Sample: {len(gen.token_ids) if hasattr(gen, 'token_ids') else 'N/A'} tokens generated")
            eval_logger.info(f"   Logprobs available: {token_logprobs is not None}, length: {len(token_logprobs) if token_logprobs else 0}")
        
        sample_uncert = []
        if token_logprobs:
            for i, logprob_entry in enumerate(token_logprobs):
                if logprob_entry is None:
                    # first token or unknown logprob – assume medium uncertainty
                    sample_uncert.append(0.5)
                    if debug and i < 5:
                        eval_logger.info(f"     Token {i}: None logprob -> uncertainty=0.5")
                    continue
                
                # Handle different vLLM logprob formats
                if hasattr(logprob_entry, 'logprob'):
                    # vLLM >= 0.4: Logprob object with .logprob attribute
                    lp = logprob_entry.logprob
                elif isinstance(logprob_entry, dict):
                    # Older vLLM: dictionary with token_id -> logprob
                    # Take the first (or most likely) entry
                    if logprob_entry:
                        lp = next(iter(logprob_entry.values()))
                        if hasattr(lp, 'logprob'):
                            lp = lp.logprob
                    else:
                        lp = None
                elif isinstance(logprob_entry, (int, float)):
                    # Direct numeric logprob
                    lp = float(logprob_entry)
                else:
                    lp = None
                
                if lp is None:
                    sample_uncert.append(0.5)
                    if debug and i < 5:
                        eval_logger.info(f"     Token {i}: Could not extract logprob -> uncertainty=0.5")
                    continue
                    
                # Convert log probability to uncertainty (1 - p)
                p = math.exp(lp) if lp < 0 else 1.0
                uncertainty = 1.0 - p
                sample_uncert.append(uncertainty)
                
                if debug and i < 5:
                    eval_logger.info(f"     Token {i}: logprob={lp:.4f}, prob={p:.4f}, uncertainty={uncertainty:.4f}")
        else:
            # No logprobs available - use default uncertainty
            gen_length = len(gen.token_ids) if hasattr(gen, 'token_ids') else max_tokens
            sample_uncert = [0.5] * gen_length
            if debug:
                eval_logger.warning(f"   No logprobs available, using default uncertainty=0.5 for {gen_length} tokens")
        
        per_sample_uncert.append(sample_uncert)

    if debug:
        eval_logger.info(f"🔬 Uncertainty calculation complete: {len(per_sample_uncert)} samples")
        for i, uncerts in enumerate(per_sample_uncert):
            if len(uncerts) > 0:
                eval_logger.info(f"   Sample {i}: {len(uncerts)} uncertainties, avg={sum(uncerts)/len(uncerts):.4f}, range=[{min(uncerts):.4f}, {max(uncerts):.4f}]")
    
    return outputs, per_sample_uncert


def generate_with_budget_forcing_vllm(
    llm,
    requests: List[List[int]],
    *,
    max_tokens: int,
    stop_sequences: Optional[List[str]],
    scale_func_name: str,
    debug: bool = False,
    **generation_kwargs,
):
    """vLLM adaptation of generate_with_budget_forcing supporting the
    *step_wise_uncertainty_driven* and *entropy_thresholding* scalers.

    Parameters
    ----------
    llm : lm_eval.models.vllm_causallms.VLLM
    requests : List[List[int]]
        Batch of prompt token-id lists.
    max_tokens : int
        Maximum number of *new* tokens that may be generated in a single
        think/pass phase.
    stop_sequences : list[str] | None
        Stop strings passed through to vLLM.
    scale_func_name : str
        Name understood by lm_eval.budget_forcing.scaler_registry.get_scale_func.
    debug : bool
        Enable detailed per-iteration debugging output.
    generation_kwargs : dict
        Remaining kwargs forwarded to SamplingParams.

    Returns
    -------
    list[vllm.RequestOutput]
    """
    eval_logger.info(f"🚀 Starting vLLM budget forcing with {len(requests)} prompts")
    eval_logger.info(f"   Scale function: {scale_func_name}")
    eval_logger.info(f"   Max tokens: {max_tokens}")
    eval_logger.info(f"   Stop sequences: {stop_sequences}")
    eval_logger.info(f"   Debug mode: {debug}")
    
    # ------------------------------------------------------------------
    # Prepare scaling function
    # ------------------------------------------------------------------
    scale_token_default = llm.tok_encode("\n\nWait, let me think about this more carefully:")
    if debug:
        eval_logger.info(f"   Default scale token: {scale_token_default} ({len(scale_token_default)} tokens)")

    # Extract recognised custom kwargs (they will not be forwarded to vLLM)
    custom_keys = [
        "step_selection_strategy",
        "max_steps",
        "use_min_uncertainty_filter",
        "min_step_uncertainty",
        "threshold",
        "decay_factor",
        "last_k",
    ]
    custom_kwargs = {k: generation_kwargs.pop(k) for k in custom_keys if k in generation_kwargs}

    if debug:
        eval_logger.info(f"   Custom scaler kwargs: {custom_kwargs}")
        eval_logger.info(f"   Remaining generation kwargs: {generation_kwargs}")
    
    scale_func = get_scale_func(
        scale_func_name,
        scale_token=scale_token_default,
        **custom_kwargs,
    )

    # Active context tracking
    contexts = [req.copy() for req in requests]  # Deep copy to avoid mutating input
    active_indices = list(range(len(contexts)))
    iteration = 0
    MAX_ITER = custom_kwargs.get("max_steps", 10)  # safeguard

    eval_logger.info(f"🔄 Starting budget forcing loop (max {MAX_ITER} iterations)")
    
    while active_indices and iteration < MAX_ITER:
        eval_logger.info(f"   Iteration {iteration}: {len(active_indices)}/{len(contexts)} prompts active")
        
        if debug:
            eval_logger.info(f"   Active indices: {active_indices}")
            for i, idx in enumerate(active_indices[:3]):  # Show first 3
                ctx_len = len(contexts[idx])
                eval_logger.info(f"     Active prompt {i} (idx={idx}): {ctx_len} tokens")
        
        # Gather current prompts for active samples
        current_prompts = [contexts[i] for i in active_indices]
        outputs, per_tok_uncert = _generate_with_uncertainty_vllm(
            llm,
            current_prompts,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            debug=debug,
            **generation_kwargs,
        )

        eval_logger.info(f"   Generation complete for iteration {iteration}")
        
        new_active = []  # indices that will be revisited in next iter
        for local_idx, (out, uncerts) in enumerate(zip(outputs, per_tok_uncert)):
            global_idx = active_indices[local_idx]
            gen_tokens = out.outputs[0].token_ids  # continuation part
            full_sequence = contexts[global_idx] + gen_tokens

            if debug:
                eval_logger.info(f"   Sample {local_idx} (global={global_idx}): generated {len(gen_tokens)} tokens, total seq={len(full_sequence)}")
                eval_logger.info(f"     Uncertainties: {uncerts[:10]}{'...' if len(uncerts) > 10 else ''}")
            
            keep_scaling, scale_token = scale_func(
                iteration=iteration,
                seq=full_sequence,
                uncertainties=uncerts,
                hflm=llm,  # this arg name kept for compatibility
            )

            if debug:
                eval_logger.info(f"     Scaler decision: keep_scaling={keep_scaling}, scale_token={scale_token[:5]}{'...' if len(scale_token) > 5 else ''}")
            
            if keep_scaling and (len(full_sequence) + len(scale_token) < max_tokens):
                contexts[global_idx] = full_sequence + scale_token
                new_active.append(global_idx)
                if debug:
                    eval_logger.info(f"     → Will continue (new length: {len(contexts[global_idx])})")
            else:
                # finalise sequence for this sample
                contexts[global_idx] = full_sequence
                if debug:
                    reason = "scaler said stop" if not keep_scaling else "would exceed max_tokens"
                    eval_logger.info(f"     → Stopping ({reason})")

        active_indices = new_active
        iteration += 1
        
        eval_logger.info(f"   End of iteration {iteration-1}: {len(active_indices)} prompts remaining")

    if iteration >= MAX_ITER:
        eval_logger.warning(f"⚠️  Reached maximum iterations ({MAX_ITER}), stopping budget forcing loop")
    else:
        eval_logger.info(f"✅ Budget forcing loop completed after {iteration} iterations")
    
    # Final generation pass to complete answers after last revision token
    eval_logger.info(f"🏁 Final generation pass for {len(contexts)} prompts")
    
    # Clean generation_kwargs for final pass  
    final_kwargs = _clean_vllm_kwargs(generation_kwargs, debug=debug)
    
    final_outputs = llm.model.generate(
        prompt_token_ids=contexts,
        sampling_params=SamplingParams(max_tokens=max_tokens, stop=stop_sequences, **final_kwargs),
        use_tqdm=True if llm.batch_size == "auto" else False,
    )

    eval_logger.info(f"✅ vLLM budget forcing complete: {len(final_outputs)} outputs generated")
    
    # Log summary statistics
    if debug:
        eval_logger.info("📊 BUDGET FORCING SUMMARY:")
        eval_logger.info(f"   Total iterations: {iteration}")
        eval_logger.info(f"   Initial prompts: {len(requests)}")
        eval_logger.info(f"   Final outputs: {len(final_outputs)}")
        for i, out in enumerate(final_outputs[:3]):  # Show first 3
            gen_text = out.outputs[0].text
            eval_logger.info(f"   Output {i} preview: '{gen_text[:100]}{'...' if len(gen_text) > 100 else ''}'")
    
    return final_outputs 