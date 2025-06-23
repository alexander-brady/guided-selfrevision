# Uncertainty Driven Test-Time Scaling

Recent research has shown that even in small language models, forcing reasoning steps during generation can significantly improve the model's performance on reasoning tasks. This technique, introduced by [Muennighof et al.](https://arxiv.org/abs/2501.19393), is known as budget forcing. It involves prompting the model with a continuation token, such as "Wait", after each generation, encouraging it to generate additional reasoning steps before arriving at a final answer.

This repository aims to extend the budget forcing technique by incorporating uncertainty-aware inference techniques. The goal is to improve the reasoning capabilities of large language models by allowing them to dynamically adjust their reasoning process based on their confidence in the generated tokens.

**Table of Contents**
- [Overview](#uncertainty-driven-test-time-scaling)
- [Evaluation Framework](#evaluation-framework)
- [Setup](#setup)
- [Scaling Functions](#scaling-functions)
    - [Custom Scaling Functions](#custom-scaling-functions)


## Evaluation Framework

Much like the original [budget forcing implementation](https://github.com/simplescaling/s1), we use the [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness) framework to evaluate our approaches. See `eval/README.md` for more details.


## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```
2. Install the required dependencies:
```bash
pip install -e .
```
3. Move to and download the evaluation harness:
```bash
cd eval/lm-evaluation-harness
pip install -e .[math,vllm]
```

**Example Evaluation Command**

Scripts used to run evaluations are in `eval/scripts.sh`. Below is an example command to run an evaluation using basic budget forcing:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR lm_eval \
    --model vllm \
    --model_args pretrained=simplescaling/s1.1-1.5B,dtype=float16,tensor_parallel_size=1 \
    --tasks aime24_nofigures,openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path ../../results \
    --log_samples \
    --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=6,thinking_n_ignore_str=Wait,scale_func_name=default" 
```

The key parameter to note here is `scale_func_name=default` in `gen_kwargs`. This will apply budget forcing during generation. Also, ensure that the `model` is set to `vllm`.

## Scaling Functions

We define a set of scaling functions that can be used to control the budget forcing process during generation. These functions determine how the model should continue generating tokens based on its confidence in the generated output. 

In addition to the default scaling function, you can also use other scaling functions by specifying `scale_func_name` in `gen_kwargs`. The currently available scaling functions are:

**default**: This is the standard budget forcing function that appends the continuation token (e.g. "Wait") after each generation step, up to a maximum number of times or until the maximum number of tokens is reached. 

**entropy_thresholding**: This method continues budget forcing until the average uncertainty is below a certain threshold, i.e. the model is confident enough in its predictions.

_Parameters (set in `gen_kwargs`):_
- `threshold`: The threshold for the average uncertainty of the model's predictions. Default is 0.5.
- `decay_factor`: The factor by which the threshold is decayed after each generation step. Default is 1.0 (no decay).
- `last_k`: The number of last tokens to consider for the mean uncertainty calculation. Default is -1, which means all tokens are considered. If less than 1, will use the last `k`% of generated tokens. Only considers tokens in the last generation step.

**uncertainty_driven_reevaluation**: This function used the uncertainty of the model's predictions to identify a sequence of tokens that the model is uncertain about, and prompts the model to re-evaluate those tokens. A sequence is defined as contiguous tokens after a punctuation mark.

_Parameters (set in `gen_kwargs`):_
- `min_threshold`: The minimum uncertainty threshold for the model to reevaluate an utterance sequence. If no utterance sequence is found with uncertainty above this threshold, the model will not be prompted to re-evaluate. Default is -1, which means that the model will always be prompted to re-evaluate.
- `ablation`: Various ablation strategies can be applied to change the selection of utterance sequences to re-evaluate. The available ablation strategies are:
  - `none`: No ablation (default); always re-evaluate the utterance sequence with the highest uncertainty.
  - `random`: Randomly select an utterance sequence to re-evaluate.
  - `last`: Re-evaluate the last utterance sequence.
  - `certain`: Re-evaluate the utterance sequence with the lowest uncertainty.
  - `third`: Re-evaluate the utterance sequence with the third highest uncertainty.

### Custom Scaling Functions
Custom scaling functions can be defined to implement new budget forcing strategies or to modify the behavior of existing ones. A scaling function must accept the following parameters:

- `iteration (int)`: The current iteration of the generation process.
- `tokens (List[str])`: The sequence of tokens generated so far (including the initial prompt).
- `entropies (List[float])`: The entropies of the tokens generated so far.
- `scale_token (List[int])`: The default token used to continue the generation process (e.g. the "wait" token).

The scaling function should return a tuple containing:
- `continue_scaling (bool)`: Whether to continue the budget forcing process.
- `scale_token (int)`: The token id to use for the next generation step.

> To define a scaling function that only checks if the model should continue budget forcing, without modifying the scaling token, you can use the `@should_continue_scaling` wrapper. 

> If you want to define a scaling function that only modifies the scaling token but does not check if budget forcing should continue, you can use the `@scale_token_only` wrapper (this will return `True` for `continue_scaling`). 

Check `src/budget_forcing/scalers` for examples of scaling functions.

A scaling function must be registered in the `get_scale_func` function in the `src/budget_forcing/scaler_registry.py` file. The function should take the name of the scaling function as a parameter and return the corresponding scaling function. Additional parameters can be passed to the scaling function by specifying them in `gen_kwargs` (see the examples in `get_scale_func`).