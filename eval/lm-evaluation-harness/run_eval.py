#!/usr/bin/env python3
from lm_eval import evaluator, tasks
import os

# Create results directory if it doesn't exist
os.makedirs("./results", exist_ok=True)

# Define your model arguments (WITHOUT thinking_n_ignore_str)
model_args = {
    "pretrained": "simplescaling/s1.1-1.5B",
    "tensor_parallel_size": 8,
    "max_model_len": 2048,
    "gpu_memory_utilization": 0.85
}

# Define thinking tokens as a string but not passed directly
thinking_tokens = "Wait;Let us reconsider...;Hmm, I should double-check this.;On second thought...;Actually...;Wait, that cannot be right.;Let me verify this calculation.;I need to analyze this from another angle.;Let us try a different approach.;I should break this down further.;Let me work through this step-by-step.;I am not confident in this result yet.;Let me trace through the logic again.;There may be something I am overlooking here.;This deserves more careful analysis.;Let me reason through this systematically.;Is this even right?;Is this correct;Could this be wrong;So, given this and that;Is this the right conclusion?;Is this sound logic?;Is this a logical conclusion..."

# The gen_kwargs need to be a STRING with proper quoting
gen_kwargs = f'max_gen_toks=2000,max_tokens_thinking=1800,thinking_n_ignore_str="{thinking_tokens}"'

# Run the evaluation
print("Starting evaluation...")
results = evaluator.simple_evaluate(
    model="vllm",
    model_args=model_args,
    tasks=["aime24_nofigures", "openai_math"],
    batch_size="auto",
    log_samples=True,
    gen_kwargs=gen_kwargs
)

# Print results table
print("\nResults Summary:")
print(evaluator.make_table(results))

# Save results to file
results_path = "./results/results.json"
evaluator.save_results(results=results, path=results_path)
print(f"\nDetailed results saved to: {results_path}")