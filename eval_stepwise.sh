#!/bin/bash
#SBATCH --job-name=stepwise_uncertainty_eval
#SBATCH --output=logs/stepwise_eval_%j.out
#SBATCH --error=logs/stepwise_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:64g
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

echo "🚀 Starting Step-wise Uncertainty Evaluation at $(date)"

source ./.env

VENV_PATH="$SCRATCH/csnlp/.venv"
export HF_HOME="$SCRATCH/csnlp/cache"

# Set environment for numbered step prompting
export PROMPTNUMBERED=1

cd eval/lm-evaluation-harness

# RESET_ENV can also be set to "true" to force a fresh environment
if [ ! -d "$VENV_PATH" ]; then
  RESET_ENV="true"
fi

if [ "$RESET_ENV" == "true" ]; then
  rm -rf "$VENV_PATH"
  python3 -m venv "$VENV_PATH"
  echo "Virtual environment created at $VENV_PATH at $(date)"
fi

source "$VENV_PATH/bin/activate"

if [ "$RESET_ENV" == "true" ]; then
  pip install --upgrade pip --quiet
  pip install -e .[math,vllm] --quiet
else
  echo "Using existing virtual environment at $VENV_PATH"
fi

echo "🔬 Starting step-wise uncertainty evaluation with numbered steps at $(date)"

# Configuration
MODEL_NAME="simplescaling/s1.1-1.5B"
OUTPUT_PATH="stepwise_uncertainty_eval/$USER/$SLURM_JOB_ID"

# Parameters for step-wise uncertainty
STEP_SELECTION_STRATEGY="highest_uncertainty"  # Options: highest_uncertainty, lowest_uncertainty, random
MAX_STEPS=8                                    # Maximum number of reasoning steps

# NEW: Threshold filtering options
USE_MIN_UNCERTAINTY_FILTER="false"            # Set to "true" to enable threshold filtering
MIN_STEP_UNCERTAINTY=0.3                      # Only used if USE_MIN_UNCERTAINTY_FILTER=true

# Enable DEBUG logging for detailed step-wise uncertainty analysis
export LOGLEVEL=DEBUG

echo "📋 Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Step selection strategy: $STEP_SELECTION_STRATEGY"
echo "   Max steps: $MAX_STEPS"
echo "   Use uncertainty threshold filtering: $USE_MIN_UNCERTAINTY_FILTER"
if [ "$USE_MIN_UNCERTAINTY_FILTER" = "true" ]; then
    echo "   Min step uncertainty threshold: $MIN_STEP_UNCERTAINTY"
else
    echo "   Min step uncertainty threshold: DISABLED (always revise most uncertain step)"
fi
echo "   Output path: $OUTPUT_PATH"
echo "   Debug logging: ENABLED (LOGLEVEL=$LOGLEVEL)"

OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_NAME,dtype=float16,max_model_len=32768,gpu_memory_utilization=0.9" \
    --tasks openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path $OUTPUT_PATH \
    --log_samples \
    --verbosity DEBUG \
    --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=6,scale_func_name=step_wise_uncertainty_driven,step_selection_strategy=$STEP_SELECTION_STRATEGY,max_steps=$MAX_STEPS,use_min_uncertainty_filter=$USE_MIN_UNCERTAINTY_FILTER,min_step_uncertainty=$MIN_STEP_UNCERTAINTY,debug=true" \
    --limit 10

echo "✅ Step-wise uncertainty evaluation completed at $(date)"

echo "📊 Collecting step-wise uncertainty metrics..."
python - <<'PY'
try:
    import sys
    sys.path.insert(0, '.')
    import lm_eval.budget_forcing.scalers as scalers
    print("\n" + "="*80)
    print("📊 STEP-WISE UNCERTAINTY FINAL METRICS")
    print("="*80)
    scalers.print_stepwise_metrics()
    print("="*80)
except Exception as e:
    print(f"⚠️  Could not collect metrics: {e}")
PY

echo "📊 Final results saved to: $OUTPUT_PATH"

deactivate 