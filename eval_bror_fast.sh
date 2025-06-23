#!/bin/bash
#SBATCH --job-name=bror_fast
#SBATCH --output=logs/bror_fast_%j.out
#SBATCH --error=logs/bror_fast_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

# =====================================================================
# BROR Fast Execution - Efficient Configuration
# =====================================================================
# 
# Configuration optimized for speed and resource efficiency.
# Higher cost_per_step leads to more conservative reasoning decisions.
# Reduced ensemble size and MC samples for faster computation.
# 
# Use case: Quick validation, resource-constrained environments,
# or when computational budget is limited.
# =====================================================================

echo "Starting BROR Fast evaluation at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Create logs directory
mkdir -p logs

# Environment setup
source venv/bin/activate
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

# BROR Fast Configuration - optimized for speed
COST_PER_STEP=0.02        # Higher cost = more conservative reasoning
ENSEMBLE_SIZE=6           # Smaller ensemble for faster computation
MC_SAMPLES=4              # Fewer MC samples for speed
SAMPLE_LENGTH=32          # Shorter continuations
MAX_REASONING_STEPS=30    # Lower maximum steps
CALIBRATION_ALPHA=0.8     # Conservative calibration
CALIBRATION_BETA=0.0      # No bias term
MAX_COMPUTATION_TIME=20.0 # Shorter time budget

echo "BROR Fast Configuration:"
echo "  Cost per step: $COST_PER_STEP (higher = more conservative)"
echo "  Ensemble size: $ENSEMBLE_SIZE (smaller = faster)"
echo "  MC samples: $MC_SAMPLES (fewer = faster)"
echo "  Sample length: $SAMPLE_LENGTH (shorter = faster)"
echo "  Max reasoning steps: $MAX_REASONING_STEPS"
echo "  Max computation time: $MAX_COMPUTATION_TIME seconds"

# Construct gen_kwargs
GEN_KWARGS="max_gen_toks=$MAX_GEN_TOKS,max_tokens_thinking=$MAX_TOKENS_THINKING,thinking_n_ignore=$THINKING_N_IGNORE,scale_func_name=bayes_risk_optimal_reasoning,cost_per_step=$COST_PER_STEP,ensemble_size=$ENSEMBLE_SIZE,mc_samples=$MC_SAMPLES,sample_length=$SAMPLE_LENGTH,max_reasoning_steps=$MAX_REASONING_STEPS,calibration_alpha=$CALIBRATION_ALPHA,calibration_beta=$CALIBRATION_BETA,max_computation_time=$MAX_COMPUTATION_TIME"

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/bror_fast_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Starting fast BROR evaluation..."

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

EXIT_CODE=$?

echo "Fast BROR evaluation completed at $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ BROR Fast evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/"
    echo ""
    echo "⚡ Fast Configuration Analysis:"
    echo "- Higher cost per step ($COST_PER_STEP) promotes early stopping"
    echo "- Reduced ensemble and MC samples for computational efficiency"
    echo "- Expected: Faster execution with potentially lower accuracy"
    echo "- Trade-off: Speed vs thoroughness of reasoning"
else
    echo "❌ BROR Fast evaluation failed with exit code $EXIT_CODE"
fi

echo ""
echo "💡 Performance Insights:"
echo "This configuration prioritizes computational efficiency over exhaustive reasoning."
echo "Use for quick validation or when operating under strict resource constraints."
echo "Compare results with standard BROR to quantify speed-accuracy trade-offs."

exit $EXIT_CODE 