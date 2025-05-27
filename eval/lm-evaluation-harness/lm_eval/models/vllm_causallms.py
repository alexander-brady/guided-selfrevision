import copy
import random
from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

from more_itertools import distribute
from packaging.version import parse as parse_version
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, configure_pad_token, undistribute
from lm_eval.utils import (
    eval_logger,
    get_rolling_token_windows,
    make_disjoint_window,
)


try:
    import ray
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass

eval_logger = eval_logger


@register_model("vllm")
class VLLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: Union[str, int] = 1,
        max_batch_size=None,
        max_length: int = None,
        max_model_len: int = None,
        seed: int = 1234,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        lora_local_path: str = None,
        **kwargs,
    ):
        super().__init__()

        if not find_spec("vllm"):
            raise Exception(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install lm-eval[vllm]` or `pip install -e .[vllm]`"
            )

        assert "cuda" in device or device is None, "vLLM only supports CUDA"
        assert (
            max_length is None or max_model_len is None
        ), "Either max_length or max_model_len may be provided, but not both"

        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.data_parallel_size = int(data_parallel_size)
        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "swap_space": int(swap_space),
            "quantization": quantization,
            "seed": int(seed),
        }
        self.model_args.update(kwargs)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else batch_size
        )
        if self.data_parallel_size <= 1:
            self.model = LLM(**self.model_args)
        else:
            eval_logger.warning(
                "You might experience occasional issues with model weight downloading when data_parallel is in use. To ensure stable performance, run with data_parallel_size=1 until the weights are downloaded and cached."
            )
            self.model_args["worker_use_ray"] = True
            self.batch_size = "auto"
            eval_logger.info("Manual batching is not compatible with data parallelism.")

            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(
                pretrained, trust_remote_code=trust_remote_code, revision=revision
            )
        self.tokenizer = get_tokenizer(
            tokenizer if tokenizer else pretrained,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
        )
        self.tokenizer = configure_pad_token(self.tokenizer)
        self.add_bos_token = add_bos_token
        if "gemma" in pretrained.lower():
            self.add_bos_token = True
            eval_logger.info(
                "Found 'gemma' in model name, a BOS token will be used as Gemma series models underperform without it."
            )

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

        self._max_gen_toks = max_gen_toks

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version(
                "0.3.0"
            ), "lora adapters only compatible with vllm > v0.3.0."
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        encoding: Union[List[List[int]], List[int]] = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]

        return encoding


    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        true_original_requests_toks = copy.deepcopy(requests)
        
        outputs_thinking = None # To store results from the thinking phase
        # This will hold parameters for the final vLLM call (either answer or loglikelihood)
        final_params_for_vllm_call = {}

        # Define known vLLM SamplingParam keys to help filter and build param dicts
        # This list can be made more comprehensive or a class/module constant if preferred.
        known_sampling_param_keys = [
            "n", "best_of", "presence_penalty", "frequency_penalty", "repetition_penalty", 
            "temperature", "top_p", "top_k", "min_p", "seed", "use_beam_search", 
            "length_penalty", "early_stopping", "stop", "stop_token_ids", 
            "include_stop_str_in_output", "ignore_eos", "max_tokens", "min_tokens", 
            "logprobs", "prompt_logprobs", "skip_special_tokens", "spaces_between_special_tokens"
        ]

        if generate:
            # kwargs is modified by modify_gen_kwargs (e.g. temperature for do_sample=False, skip_special_tokens)
            # This `kwargs` still contains _thinking suffixed params and other thinking control params.
            kwargs = self.modify_gen_kwargs(kwargs)

            rejection_sample = kwargs.get("rejection_sample") # Use .get(), don't pop yet
            if rejection_sample:
                # Check effective temperature for thinking
                effective_temp_thinking = kwargs.get("temperature_thinking", kwargs.get("temperature", 0))
                if effective_temp_thinking == 0:
                    eval_logger.warning("Rejection sampling works best with temperature/temperature_thinking > 0.")
                if "max_tokens_thinking" not in kwargs: # Ensure it's present if rejection_sample is on
                    raise ValueError("Rejection sampling requires max_tokens_thinking to be set in gen_kwargs.")

            is_thinking_phase_active = any(["thinking" in k for k in kwargs]) or rejection_sample

            if is_thinking_phase_active:
                eval_logger.info("Separating thinking and answering generation.")
                # Pop thinking-specific control flow parameters (not for SamplingParams directly)
                thinking_start = kwargs.pop("thinking_start", "<|im_start|>think")
                thinking_end = kwargs.pop("thinking_end", "<|im_start|>answer")
                thinking_n_ignore = kwargs.pop("thinking_n_ignore", None)
                thinking_n_ignore_str_or_list = kwargs.pop("thinking_n_ignore_str", None)
                
                until_thinking_str_list_input = [kwargs.pop("until_thinking", "<|im_start|▶")]
                if "until_thinking_2" in kwargs:
                    until_thinking_str_list_input.append(kwargs.pop("until_thinking_2"))
                
                # Incorporate general 'stop' from kwargs into thinking stops, if provided
                # The 'stop' argument to this function is for the final answer stage.
                general_stop_for_thinking_phase = kwargs.get("stop", []) 
                if general_stop_for_thinking_phase is not None:
                    if isinstance(general_stop_for_thinking_phase, str): 
                        general_stop_for_thinking_phase = [general_stop_for_thinking_phase]
                    until_thinking_str_list_input.extend(s for s in general_stop_for_thinking_phase if s not in until_thinking_str_list_input)

                # Add EOS token to thinking stops
                eos_token_str = self.tokenizer.decode(self.eot_token_id)
                if eos_token_str not in until_thinking_str_list_input:
                    until_thinking_str_list_input.append(eos_token_str)

                eval_logger.info(f"Thinking start: {thinking_start}, Thinking end: {thinking_end}, Stop during thinking: {until_thinking_str_list_input}")
                
                thinking_start_tok = self.tok_encode(thinking_start)
                thinking_end_tok = self.tok_encode(thinking_end)
                thinking_end_max = thinking_end + "\\nFinal Answer:" # Escaped newline
                thinking_end_max_tok = self.tok_encode(thinking_end_max)
                newline_tok = self.tok_encode("\\n") # Escaped newline

                # --- Prepare parameters for THINKING phase ---
                sampling_params_for_thinking = {}

                # 1. Populate with general sampling params from kwargs (these will be defaults for thinking)
                for key in known_sampling_param_keys:
                    if key in kwargs:
                        sampling_params_for_thinking[key] = kwargs[key]

                # 2. Override with specific _thinking versions, and POP them from main kwargs
                # Iterate over a copy of keys as kwargs is being modified
                thinking_params_to_pop = []
                for k_original_kwarg in kwargs.keys():
                    if k_original_kwarg.endswith("_thinking"):
                        thinking_params_to_pop.append(k_original_kwarg)
                
                for k_thinking_param in thinking_params_to_pop:
                    value = kwargs.pop(k_thinking_param) # Pop from main kwargs
                    param_name_no_suffix = k_thinking_param.replace("_thinking", "")
                    if param_name_no_suffix in known_sampling_param_keys:
                         sampling_params_for_thinking[param_name_no_suffix] = value

                # 3. Specifically handle max_tokens_thinking (in case it wasn't caught above)
                if "max_tokens_thinking" in kwargs:
                    val_max_tokens_thinking = kwargs.pop("max_tokens_thinking")
                    if isinstance(val_max_tokens_thinking, str) and val_max_tokens_thinking.lower() == "auto":
                        auto_max_val = (self.max_length // 2) 
                        if true_original_requests_toks:
                             auto_max_val = self.max_length - (max([len(x) for x in true_original_requests_toks]) + len(thinking_start_tok) + len(thinking_end_max_tok) + 50)
                        sampling_params_for_thinking["max_tokens"] = max(20, auto_max_val)
                        eval_logger.info(f"Auto setting max_tokens for thinking to {sampling_params_for_thinking['max_tokens']}")
                    elif val_max_tokens_thinking is not None:
                        sampling_params_for_thinking["max_tokens"] = int(val_max_tokens_thinking)

                # 4. Set default max_tokens for thinking if not set
                if "max_tokens" not in sampling_params_for_thinking:
                    sampling_params_for_thinking["max_tokens"] = self.max_gen_toks // 2 or 128
                    eval_logger.info(f"Defaulting max_tokens for thinking to {sampling_params_for_thinking['max_tokens']}")
                
                if rejection_sample and "max_tokens" in sampling_params_for_thinking:
                    sampling_params_for_thinking["max_tokens"] += 1

                # Now, `kwargs` has been cleaned of `_thinking` suffixed params & `max_tokens_thinking`.
                # `sampling_params_for_thinking` has the correct values for the thinking phase.

                # Handle stop sequences for the thinking phase
                until_thinking_tok_list_encoded = self.tok_encode(until_thinking_str_list_input)
                
                single_token_stops_thinking = [t[0] for t in until_thinking_tok_list_encoded if len(t) == 1]
                multi_token_stop_strings_thinking = [s for s, t_list in zip(until_thinking_str_list_input, until_thinking_tok_list_encoded) if len(t_list) > 1]
                
                # Set stop sequences for thinking - be more explicit
                if single_token_stops_thinking:
                    sampling_params_for_thinking["stop_token_ids"] = single_token_stops_thinking
                
                if multi_token_stop_strings_thinking:
                    sampling_params_for_thinking["stop"] = multi_token_stop_strings_thinking
                elif until_thinking_str_list_input:  # Fallback to original if no multi-token stops
                    sampling_params_for_thinking["stop"] = until_thinking_str_list_input

                # Debug: Print the thinking parameters before creating SamplingParams
                print(f"DEBUG: thinking params keys: {list(sampling_params_for_thinking.keys())}")
                print(f"DEBUG: thinking stop sequences: {sampling_params_for_thinking.get('stop', [])}")
                print(f"DEBUG: thinking stop_token_ids: {sampling_params_for_thinking.get('stop_token_ids', [])}")
                
                # Construct SamplingParams for thinking
                vllm_sampling_params_thinking = SamplingParams(**sampling_params_for_thinking)
                
                # Static thinking_n_ignore_str_text and tok_static (from existing lines 252-257, ensure these vars are defined)
                static_thinking_n_ignore_str_text = ""
                thinking_n_ignore_str_tok_static = []
                if isinstance(thinking_n_ignore_str_or_list, str):
                    eval_logger.info(f"Thinking ignore string (static): {thinking_n_ignore_str_or_list}")
                    thinking_n_ignore_str_tok_static = self.tok_encode(thinking_n_ignore_str_or_list)
                    static_thinking_n_ignore_str_text = thinking_n_ignore_str_or_list
                elif isinstance(thinking_n_ignore_str_or_list, list) and thinking_n_ignore_str_or_list:
                    eval_logger.info(f"Thinking ignore string options (dynamic): {thinking_n_ignore_str_or_list}")

                base_prompts_for_thinking = [req + thinking_start_tok for req in true_original_requests_toks]

                if rejection_sample:
                    # ... (Your rejection sampling logic - needs to populate `outputs_thinking`) ...
                    eval_logger.warning("Rejection sampling logic needs to be fully integrated here.")
                    # Fallback example if rejection sampling is not yet integrated:
                    if not outputs_thinking or all(o is None for o in outputs_thinking):
                        outputs_thinking = self.model.generate(
                            prompt_token_ids=base_prompts_for_thinking,
                            sampling_params=vllm_sampling_params_thinking,
                            use_tqdm=True if self.batch_size == "auto" else False)

                elif thinking_n_ignore is not None:
                    eval_logger.info(f"S1-style thinking: Will ignore stops up to {thinking_n_ignore} times.")
                    thinking_n_ignore_iterations = int(thinking_n_ignore) + 1
                    outputs_thinking = [None] * len(base_prompts_for_thinking)
                    current_prompts_iter = copy.deepcopy(base_prompts_for_thinking)
                    active_request_indices_iter = list(range(len(base_prompts_for_thinking)))

                    for i_iter_step in range(thinking_n_ignore_iterations):
                        if not active_request_indices_iter: break
                        prompts_for_vllm_call_this_step = [current_prompts_iter[idx] for idx in active_request_indices_iter]
                        if not prompts_for_vllm_call_this_step: continue

                        generated_segments = self.model.generate(
                            prompt_token_ids=prompts_for_vllm_call_this_step,
                            sampling_params=vllm_sampling_params_thinking,
                            use_tqdm=False)

                        next_iter_active_indices = []
                        for batch_loop_idx, segment_output_obj in enumerate(generated_segments):
                            original_request_idx_in_batch = active_request_indices_iter[batch_loop_idx]

                            if outputs_thinking[original_request_idx_in_batch] is None:
                                class _PseudoCompOut:
                                    def __init__(self): self.token_ids, self.text, self.finish_reason, self.logprobs = [], "", None, None
                                class _PseudoReqOut:
                                    def __init__(self): self.outputs = [_PseudoCompOut()]
                                outputs_thinking[original_request_idx_in_batch] = _PseudoReqOut()
                            
                            accumulated_thought_data = outputs_thinking[original_request_idx_in_batch].outputs[0]
                            segment_tokens = list(segment_output_obj.outputs[0].token_ids)
                            segment_text = segment_output_obj.outputs[0].text
                            segment_finish_reason = segment_output_obj.outputs[0].finish_reason
                            
                            tokens_to_add_this_step = list(segment_tokens) # Start with the full segment
                            text_to_add_this_step = segment_text         # Start with the full segment text

                            continue_thinking_for_this_req = False
                            max_total_thinking_toks = vllm_sampling_params_thinking.max_tokens # Use max_tokens from the thinking params
                            current_total_len = len(accumulated_thought_data.token_ids) + len(segment_tokens)

                            if segment_finish_reason == "length" or \
                               i_iter_step == thinking_n_ignore_iterations - 1 or \
                               current_total_len >= max_total_thinking_toks:
                                accumulated_thought_data.finish_reason = segment_finish_reason
                                if current_total_len >= max_total_thinking_toks:
                                    accumulated_thought_data.finish_reason = "length"
                                    # Truncate tokens_to_add_this_step if it causes overflow
                                    if len(accumulated_thought_data.token_ids) + len(tokens_to_add_this_step) > max_total_thinking_toks:
                                        can_add = max_total_thinking_toks - len(accumulated_thought_data.token_ids)
                                        tokens_to_add_this_step = tokens_to_add_this_step[:can_add]
                                        text_to_add_this_step = self.tokenizer.decode(tokens_to_add_this_step)

                            else: # Not a final step, check for stop sequence to ignore
                                for stop_seq_idx, stop_seq_toks_candidate in enumerate(until_thinking_tok_list_encoded):
                                    stop_seq_str_candidate = until_thinking_str_list_input[stop_seq_idx]
                                    if len(segment_tokens) >= len(stop_seq_toks_candidate) and \
                                       segment_tokens[-len(stop_seq_toks_candidate):] == stop_seq_toks_candidate:
                                        continue_thinking_for_this_req = True
                                        
                                        tokens_to_add_this_step = segment_tokens[:-len(stop_seq_toks_candidate)]
                                        # Adjust text_to_add_this_step carefully if stop string was part of it
                                        if segment_text.endswith(stop_seq_str_candidate):
                                            text_to_add_this_step = segment_text[:-len(stop_seq_str_candidate)]
                                        else: # Fallback if text doesn't match exactly (e.g. due to tokenization nuances)
                                            text_to_add_this_step = self.tokenizer.decode(tokens_to_add_this_step)

                                        chosen_ignore_toks_for_step, chosen_ignore_text_for_step = [], ""
                                        if isinstance(thinking_n_ignore_str_or_list, list) and thinking_n_ignore_str_or_list:
                                            selected_signal = random.choice(thinking_n_ignore_str_or_list)
                                            chosen_ignore_toks_for_step = self.tok_encode(selected_signal)
                                            chosen_ignore_text_for_step = selected_signal
                                            eval_logger.debug(f"Req {original_request_idx_in_batch}: Dynamic: '{selected_signal}'")
                                        elif thinking_n_ignore_str_tok_static: # A single string was provided
                                            chosen_ignore_toks_for_step = thinking_n_ignore_str_tok_static
                                            chosen_ignore_text_for_step = static_thinking_n_ignore_str_text
                                            eval_logger.debug(f"Req {original_request_idx_in_batch}: Static: '{static_thinking_n_ignore_str_text}'")
                                        
                                        tokens_to_add_this_step.extend(chosen_ignore_toks_for_step)
                                        text_to_add_this_step += chosen_ignore_text_for_step
                                        break
                            
                            accumulated_thought_data.token_ids.extend(tokens_to_add_this_step)
                            accumulated_thought_data.text += text_to_add_this_step
                            if not continue_thinking_for_this_req and accumulated_thought_data.finish_reason is None:
                                accumulated_thought_data.finish_reason = segment_finish_reason if segment_finish_reason else "done_step"

                            if continue_thinking_for_this_req:
                                next_iter_active_indices.append(original_request_idx_in_batch)
                                current_prompts_iter[original_request_idx_in_batch] = \
                                    list(base_prompts_for_thinking[original_request_idx_in_batch]) + \
                                    list(accumulated_thought_data.token_ids)
                        active_request_indices_iter = sorted(list(set(next_iter_active_indices)))

                    max_think_toks_overall = vllm_sampling_params_thinking.max_tokens
                    for req_idx_final_check in range(len(outputs_thinking)):
                        if outputs_thinking[req_idx_final_check] and outputs_thinking[req_idx_final_check].outputs:
                            out_obj_data = outputs_thinking[req_idx_final_check].outputs[0]
                            if len(out_obj_data.token_ids) > max_think_toks_overall:
                                eval_logger.warning(f"Req {req_idx_final_check} (S1): Total {len(out_obj_data.token_ids)} > {max_think_toks_overall}. Cut.")
                                out_obj_data.token_ids = out_obj_data.token_ids[:max_think_toks_overall]
                                out_obj_data.text = self.tokenizer.decode(out_obj_data.token_ids)
                                out_obj_data.finish_reason = "length"
                
                else: # Standard single-pass thinking
                    eval_logger.info("Standard single-pass thinking generation.")
                    outputs_thinking = self.model.generate(
                        prompt_token_ids=base_prompts_for_thinking,
                        sampling_params=vllm_sampling_params_thinking,
                        use_tqdm=True if self.batch_size == "auto" else False)

                # --- Post-thinking: Prepare prompts for the ANSWER stage ---
                requests_for_answer_stage = []
                if outputs_thinking: # Ensure outputs_thinking is not None and potentially populated
                    for i, thought_result_obj in enumerate(outputs_thinking):
                        current_answer_prompt_toks = list(true_original_requests_toks[i]) # Start with true original
                        current_answer_prompt_toks.extend(thinking_start_tok)
                        full_wrapped_thought_text_for_record = thinking_start

                        if thought_result_obj and thought_result_obj.outputs:
                            final_thought_tokens = thought_result_obj.outputs[0].token_ids
                            final_thought_text = thought_result_obj.outputs[0].text
                            final_thought_finish_reason = thought_result_obj.outputs[0].finish_reason
                            current_answer_prompt_toks.extend(final_thought_tokens)
                            full_wrapped_thought_text_for_record += final_thought_text
                            if final_thought_finish_reason == "length":
                                current_answer_prompt_toks.extend(newline_tok + thinking_end_max_tok)
                                full_wrapped_thought_text_for_record += "\n" + thinking_end_max
                        else:
                                current_answer_prompt_toks.extend(thinking_end_tok)
                                full_wrapped_thought_text_for_record += thinking_end
                            thought_result_obj.outputs[0].text = full_wrapped_thought_text_for_record
                    else:
                            eval_logger.warning(f"Req {i}: No valid thought. Appending only thinking_end.")
                            current_answer_prompt_toks.extend(thinking_end_tok)
                            if outputs_thinking[i] is None: # If it was None, create a minimal placeholder
                                class _PseudoCompOut:
                                    def __init__(self): self.text = thinking_start + thinking_end; self.token_ids = []
                                class _PseudoReqOut:
                                    def __init__(self): self.outputs = [_PseudoCompOut()]
                                outputs_thinking[i] = _PseudoReqOut() # Assign minimal for consistency
                            elif outputs_thinking[i].outputs:
                                outputs_thinking[i].outputs[0].text = thinking_start + thinking_end
                        requests_for_answer_stage.append(current_answer_prompt_toks)
                    
                    requests = requests_for_answer_stage # Update `requests` for the final vLLM call
                else: # No thinking occurred or outputs_thinking is empty
                    requests = true_original_requests_toks # Fallback to original if no thoughts generated
                
                # `requests` is updated to `requests_for_answer_stage`
                # `kwargs` has been cleaned of _thinking specific params.
                # It now contains general sampling params (e.g. "temperature" if "temperature_thinking" was not used and "temperature" was in original gen_kwargs)
                # and any other non-vLLM, non-thinking-control kwargs that were originally passed.
                
                # Prepare final parameters for ANSWER generation
                final_params_for_vllm_call = {"max_tokens": max_tokens, "stop": stop} # `max_tokens` & `stop` are from func signature
                
                # Add other relevant sampling parameters from the now-cleaned kwargs
                for key in known_sampling_param_keys:
                    if key in kwargs and key not in ["max_tokens", "stop"]: # Avoid overriding explicit ones for answer phase
                        final_params_for_vllm_call[key] = kwargs[key]

            else: # No thinking phase was active at all
                requests = true_original_requests_toks
                # Prepare parameters for direct answer generation (no thinking)
                final_params_for_vllm_call = {"max_tokens": max_tokens, "stop": stop}
                for key in known_sampling_param_keys: # Add general sampling params from kwargs
                     if key in kwargs and key not in ["max_tokens", "stop"]:
                        final_params_for_vllm_call[key] = kwargs[key]
        
        else: # Loglikelihood mode
            requests = true_original_requests_toks
            # Loglikelihood has its own fixed params
            final_params_for_vllm_call = {"temperature": 0, "prompt_logprobs": 1, "max_tokens": 1, "detokenize": False} 
        
        # Ensure critical defaults for vLLM if not set, especially for generation
        if generate:
            if final_params_for_vllm_call.get("max_tokens") is None: # Should be set if generate=True
                 final_params_for_vllm_call["max_tokens"] = self.max_gen_toks # Fallback for safety
            # vLLM SamplingParams has its own defaults for temp, top_p, top_k, so explicit setdefault might not be needed
            # unless specific overrides are desired when not provided.
            # final_params_for_vllm_call.setdefault("temperature", 1.0) 
            # final_params_for_vllm_call.setdefault("top_p", 1.0)
            # final_params_for_vllm_call.setdefault("top_k", -1) # -1 often means no top_k filtering

        final_vllm_sampling_params = SamplingParams(**final_params_for_vllm_call)

        # Data parallel logic (currently simplified/placeholder in previous attempts)
        # if self.data_parallel_size > 1: ...

        if self.lora_request is not None:
            final_model_outputs = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=final_vllm_sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request)
        else:
            final_model_outputs = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=final_vllm_sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False)
            
        if generate and outputs_thinking is not None: # Ensure outputs_thinking exists
            for i, answer_out_obj in enumerate(final_model_outputs):
                if i < len(outputs_thinking) and outputs_thinking[i] and \
                   outputs_thinking[i].outputs and answer_out_obj.outputs: # Defensive checks
                    thinking_text_part = outputs_thinking[i].outputs[0].text
                    answer_out_obj.outputs[0].text = thinking_text_part + answer_out_obj.outputs[0].text
        
        return final_model_outputs

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        # max_seq_len - (1 for context)
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)

        return loglikelihoods

## generate_until modified to accept a list 
    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
    
        res = []
        
        # ADDED CODE BLOCK START: Handle multiple continuation signals
        # Process all_gen_kwargs to have multiple thinking_n_ignore_str 

        for i, req in enumerate(requests):
            # _context instead of context to avoid confusion
            _context, gen_kwargs = req.arguments

            # Check if thinking_n_ignore_str is a list or comma-separated string
            if "thinking_n_ignore_str" in gen_kwargs:
                signals = gen_kwargs["thinking_n_ignore_str"]

                # Handle comma-separated string
                if isinstance(signals, str) and "###" in signals:
                    signals = [s.strip() for s in signals.split("###")]
                    # gen_kwargs now has a list of of signals
                    gen_kwargs["thinking_n_ignore_str"] = signals
                
                # Assuming req.arguments is a tuple (context, kwargs_dict)
                # and kwargs_dict is mutable:
                # req.arguments[1]["thinking_n_ignore_str"] = signals 
                    if hasattr(requests[i], 'arguments') and isinstance(requests[i].arguments, tuple) and len(requests[i].arguments) == 2:
                        current_context, current_gen_kwargs = requests[i].arguments
                        current_gen_kwargs["thinking_n_ignore_str"] = signals
                        print(f"current_gen_kwargs: {current_gen_kwargs}")
                        requests[i] = Instance(
                            arguments=(current_context, current_gen_kwargs), # type: ignore
                            # Re-pass other necessary attributes from req to new Instance
                            request_type=req.request_type,
                            doc=req.doc,
                            idx=req.idx,
                            metadata=req.metadata,
                            resps=req.resps,
                            task_name=req.task_name,
                            doc_id=req.doc_id,
                            repeats=req.repeats,
                        ) 
                    else: # Fallback if structure is different, or direct mutation if args[1] is the dict
                         gen_kwargs["thinking_n_ignore_str"] = signals

                # The part that selected a single random signal is removed/commented:
                # if isinstance(signals, list) and len(signals) > 0:
                #     import random # Should be module-level
                #     selected_signal = random.choice(signals)
                #     gen_kwargs["thinking_n_ignore_str"] = selected_signal # REMOVED
                #     requests[i].arguments = (_context, gen_kwargs) # Update if necessary
        # ADDED CODE BLOCK END


        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]] = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                )
            # add EOS token to stop sequences
            eos = self.tokenizer.decode(self.eot_token_id)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            # max len for inputs = max length, minus room to generate the max new tokens
            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            # perform batched generation
            cont = self._model_generate(
                requests=context_encoding,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )

            # cache generations
            for i, (output, context) in tqdm(enumerate(zip(cont, context)), desc="final processing"):
                generated_text = output.outputs[0].text
                # swj hack
                # from ipdb import set_trace as bp
                # bp()
                # check if "Answer:" in generated_text, if not resample cont = self._model_generate(requests=context_encoding, generate=True, max_tokens=max_gen_toks, stop=until, **kwargs) until it reaches "Answer:"
                # max_attemp = 5
                # while "Answer:" not in generated_text:
                #     if max_attemp == 0:
                #         print(f"max_attemp reached, question: {i}")
                #         break
                #     max_attemp -= 1
                #     cont_new = self._model_generate(requests=[context_encoding[i]], generate=True, max_tokens=max_gen_toks, stop=until, **kwargs)
                #     generated_text = cont_new[0].outputs[0].text
                #     print(f"resample until 'Answer:', question: {i}")
                res.append(generated_text)

                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)

            for output, ctxlen, (cache_key, _, _), inp in zip(
                outputs, ctxlens, chunk, inputs
            ):
                answer = self._parse_logprobs(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                )

                res.append(answer)

                if cache_key is not None:
                    # special case: loglikelihood_rolling produces a number of loglikelihood requests
                    # all with cache key None. instead do add_partial on the per-example level
                    # in the loglikelihood_rolling() function for those.
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            # vLLM changed the return type of logprobs from float
            # to a Logprob object storing the float value + extra data
            # (https://github.com/vllm-project/vllm/pull/3065).
            # If we are dealing with vllm's Logprob object, return
            # the logprob value stored as an attribute. Otherwise,
            # return the object itself (which should be a float
            # for older versions of vLLM).
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
            for logprob_dict in continuation_logprobs_dicts
        ]

        # Calculate continuation_logprobs
        # assume ctxlen always >= 1
        continuation_logprobs = sum(
            logprob_dict.get(token)
            for token, logprob_dict in zip(
                tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
            )
        )

        # Determine if is_greedy
        is_greedy = True
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        ):
            # Get the token with the maximum log probability from the logprob_dict
            if logprob_dict:  # Ensure the logprob_dict is not None
                top_token = max(logprob_dict, key=logprob_dict.get)
                if top_token != token:
                    is_greedy = False
                    break

        return continuation_logprobs, is_greedy

    @staticmethod
    def modify_gen_kwargs(kwargs: dict) -> dict:
        # sampling_params
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False and "temperature" not in kwargs:
            eval_logger.debug(
                "Got `do_sample=False` and no temperature value, setting VLLM temperature to 0.0 ..."
            )
            kwargs["temperature"] = 0.0
        # hf defaults
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get(
            "spaces_between_special_tokens", False
        )
        # More thorough cleaning of kwargs
        kwargs_keys_to_remove = []
        for k in kwargs.keys():
            if k.endswith("_thinking") or k in ["rejection_sample", "max_tokens_thinking"]:
                kwargs_keys_to_remove.append(k)
        
        for k in kwargs_keys_to_remove:
            kwargs.pop(k, None)
        return kwargs
