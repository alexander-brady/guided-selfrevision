#!/bin/bash
#SBATCH --job-name=bror_eval
#SBATCH --output=logs/bror_eval_%j.out
#SBATCH --error=logs/bror_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# =====================================================================
# Bayes-Risk-Optimal Reasoning (BROR) Evaluation Script
# =====================================================================
# 
# Mathematical Framework:
# - Estimates P(A = correct | H_t) using Bayesian ensemble methods
# - Forecasts E[p_{t+1} | H_t] - p_t using Monte Carlo and regression
# - Applies optimal stopping rule: continue iff Δp_t > C
# - Minimizes expected Bayes risk: R = (1-p)·error_cost + C·compute_cost
#
# Configuration: Balanced performance for general evaluation
# =====================================================================

echo "Starting BROR evaluation at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Create logs directory if it doesn't exist
mkdir -p logs

# Environment setup
source venv/bin/activate

# Export necessary environment variables
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Configuration parameters
MODEL="simplescaling/s1.1-1.5B"
TASK="aime24_nofigures"
BATCH_SIZE="auto"
MAX_GEN_TOKS=32768
MAX_TOKENS_THINKING="auto"
THINKING_N_IGNORE=6

# BROR-specific parameters
# These create a balanced trade-off between accuracy and computational efficiency
COST_PER_STEP=0.01        # Marginal cost C (continue iff Δp_t > C)
ENSEMBLE_SIZE=8           # Number of ensemble members for uncertainty estimation
MC_SAMPLES=6              # Monte Carlo samples for forecasting
SAMPLE_LENGTH=48          # Length of each MC continuation
MAX_REASONING_STEPS=50    # Maximum reasoning iterations
CALIBRATION_ALPHA=1.0     # Logistic calibration parameter α
CALIBRATION_BETA=0.0      # Logistic calibration parameter β
MAX_COMPUTATION_TIME=30.0 # Time budget per BROR computation (seconds)

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Task: $TASK"
echo "  Batch size: $BATCH_SIZE"
echo "  Max generation tokens: $MAX_GEN_TOKS"
echo "  Max thinking tokens: $MAX_TOKENS_THINKING"
echo "  Thinking ignore: $THINKING_N_IGNORE"
echo ""
echo "BROR Parameters:"
echo "  Cost per step: $COST_PER_STEP"
echo "  Ensemble size: $ENSEMBLE_SIZE"
echo "  MC samples: $MC_SAMPLES"
echo "  Sample length: $SAMPLE_LENGTH"
echo "  Max reasoning steps: $MAX_REASONING_STEPS"
echo "  Calibration α: $CALIBRATION_ALPHA"
echo "  Calibration β: $CALIBRATION_BETA"
echo "  Max computation time: $MAX_COMPUTATION_TIME"

# Construct gen_kwargs for BROR
GEN_KWARGS="max_gen_toks=$MAX_GEN_TOKS,max_tokens_thinking=$MAX_TOKENS_THINKING,thinking_n_ignore=$THINKING_N_IGNORE,scale_func_name=bayes_risk_optimal_reasoning,cost_per_step=$COST_PER_STEP,ensemble_size=$ENSEMBLE_SIZE,mc_samples=$MC_SAMPLES,sample_length=$SAMPLE_LENGTH,max_reasoning_steps=$MAX_REASONING_STEPS,calibration_alpha=$CALIBRATION_ALPHA,calibration_beta=$CALIBRATION_BETA,max_computation_time=$MAX_COMPUTATION_TIME"

echo ""
echo "Generated gen_kwargs: $GEN_KWARGS"
echo ""

# Output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/bror_balanced_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Starting evaluation with BROR reasoning..."

# Run evaluation
python -m lm_eval \
    --model hf \
    --model_args pretrained="$MODEL",dtype=float16 \
    --tasks "$TASK" \
    --batch_size "$BATCH_SIZE" \
    --apply_chat_template \
    --gen_kwargs "$GEN_KWARGS" \
    --output_path "$OUTPUT_DIR/results.json" \
    --log_samples \
    --show_config \
    --verbosity INFO 2>&1 | tee "$OUTPUT_DIR/evaluation.log"

# Capture exit code
EXIT_CODE=$?

echo "Evaluation completed at $(date)"
echo "Exit code: $EXIT_CODE"

# Print summary
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ BROR evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/"
    
    # Display results summary if available
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo ""
        echo "📊 Results Summary:"
        python -c "
import json
import os
try:
    with open('$OUTPUT_DIR/results.json', 'r') as f:
        results = json.load(f)
    
    print('Task Results:')
    for task, metrics in results.get('results', {}).items():
        print(f'  {task}:')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'    {metric}: {value:.4f}')
            else:
                print(f'    {metric}: {value}')
    
    # Print configuration info
    config = results.get('config', {})
    print(f'\nConfiguration:')
    print(f'  Model: {config.get(\"model_args\", \"unknown\")}')
    print(f'  Gen kwargs: {config.get(\"gen_kwargs\", \"unknown\")}')
    
except Exception as e:
    print(f'Could not parse results: {e}')
"
    fi
    
    echo ""
    echo "🎯 BROR Reasoning Analysis:"
    echo "Mathematical Foundation: Bayes risk minimization with optimal stopping"
    echo "Decision Rule: Continue reasoning iff E[Δp_t] > C = $COST_PER_STEP"
    echo "Expected Benefits: Principled cost-benefit trade-offs for reasoning depth"
    echo ""
    echo "📈 Post-evaluation Analysis:"
    echo "1. Check logs for BROR decision summaries"
    echo "2. Analyze cost-effectiveness metrics"
    echo "3. Compare against baseline and other methods"
    echo "4. Tune cost_per_step if needed for optimal performance"
    
else
    echo "❌ BROR evaluation failed with exit code $EXIT_CODE"
    echo "Check the logs for detailed error information:"
    echo "  Evaluation log: $OUTPUT_DIR/evaluation.log"
    echo "  SLURM output: logs/bror_eval_${SLURM_JOB_ID}.out"
    echo "  SLURM error: logs/bror_eval_${SLURM_JOB_ID}.err"
fi

echo ""
echo "🧠 MATHEMATICAL INTERPRETATION:"
echo "BROR implements optimal stopping theory for LLM reasoning:"
echo "- R_stop = (1 - p_t) + C·t"
echo "- R_cont = C·(t+1) + (1 - E[p_{t+1}|H_t])"
echo "- Continue iff R_cont < R_stop ⟺ Δp_t > C"
echo "where p_t = P(answer correct | reasoning so far)"
echo "and Δp_t = expected improvement in correctness probability"

echo ""
echo "Script completed at $(date)"
exit $EXIT_CODE 