# Budget Forcing 

Implementation for budget forcing with huggingface models.

## Example Usage

To use budget forcing, set the following parameters when running:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR lm_eval \
    --model hf \
    --model_args pretrained=simplescaling/s1.1-1.5B,dtype=float16,tensor_parallel_size=1 \
    --tasks aime24_nofigures,openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path s1.1_1.5B_eval/$USER/$SLURM_JOB_ID \
    --log_samples \
    --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=6,thinking_n_ignore_str=Wait,scale_func_name=default" 
```

The key parameter to note here is `scale_func_name=default` in `gen_kwargs`. This will apply budget forcing during generation. Also, ensure that the `model` is set to `hf`.

## Scaling Functions

In addition to the default scaling function, you can also use other scaling functions by specifying `scale_func_name` in `gen_kwargs`. The currently available scaling functions are:

**default**: This is the standard budget forcing function that scales the budget based on the model's capabilities.

**entropy_thresholding**: This function calculates the average entropy of the model's predictions, and continues budget forcing until the entropy is below a certain threshold, i.e. the model is confident enough in its predictions.

Parameters (set in `gen_kwargs`):
- `threshold`: The threshold for the average entropy of the model's predictions. Default is 0.5.
- `decay_factor`: The factor by which the threshold is decayed after each generation step. Default is 1.0.
- `last_k`: The number of last tokens to consider for the entropy calculation. Default is -1, which means all tokens are considered. If less than 1, will use the last `k`% of generated tokens. Only considers tokens in the last generation step.

## Custom Scaling Functions
A custom scaling function can be implemented. A scaling function must accept the following parameters:

- `iteration (int)`: The current iteration of the generation process.
- `seq (List[str])`: The sequence of tokens generated so far.
- `entropies (List[float])`: The entropies of the tokens generated so far.
- `scale_token (List[int])`: The default token used to continue the generation process (e.g. the "wait" token).

The scaling function should return a tuple containing:
- `continue_scaling (bool)`: Whether to continue the budget forcing process.
- `scale_token (int)`: The token id to use for the next generation step.

> To define a scaling function that only checks if the model should continue budget forcing, without modifying the scaling token, you can use the `@should_continue_scaling` wrapper. 

> If you want to define a scaling function that only modifies the scaling token but does not check if budget forcing should continue, you can use the `@scale_token_only` wrapper (this will return `True` for `continue_scaling`).

Check `scalers.py` for examples of scaling functions.

A scaling function must be registered in the `get_scale_func` function in the `scaler_registry.py` file. The function should take the name of the scaling function as a parameter and return the corresponding scaling function. Additional parameters can be passed to the scaling function by specifying them in `gen_kwargs` (see the `entropy_thresholding` example in `get_scale_func`).
