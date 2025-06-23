#!/bin/bash
#SBATCH --job-name=eig_reasoning_eval
#SBATCH --output=logs/eig_eval_%j.out
#SBATCH --error=logs/eig_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:64g
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

echo "🔬 Starting Expected Information Gain (EIG) Reasoning Evaluation at $(date)"

source ./.env

VENV_PATH="$SCRATCH/csnlp/.venv"
export HF_HOME="$SCRATCH/csnlp/cache"

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

echo "🔬 Starting Expected Information Gain reasoning evaluation at $(date)"

# Configuration
MODEL_NAME="simplescaling/s1.1-1.5B"
OUTPUT_PATH="eig_reasoning_eval/$USER/$SLURM_JOB_ID"

# EIG Configuration Presets - Choose one by uncommenting
# =======================================================

# BALANCED Configuration (Default - Good for most cases)
EIG_CONFIG="balanced"
LAMBDA_COST=0.05                    # Information cost threshold (lower = more reasoning)
BEAM_SIZE=8                         # Number of answer candidates for posterior estimation
MC_SAMPLES=5                        # Monte Carlo samples for forecasting
SAMPLE_LENGTH=64                    # Length of each MC sample continuation
TEMPERATURE=1.0                     # Sampling temperature for MC
TOP_P=0.9                          # Top-p sampling for MC
MAX_COMPUTATION_TIME=30.0          # Max computation time per EIG calculation (seconds)

# Uncomment for HIGH PRECISION configuration:
# EIG_CONFIG="high_precision"
# LAMBDA_COST=0.02                   # Lower threshold = more reasoning
# BEAM_SIZE=12                       # More answer candidates
# MC_SAMPLES=8                       # More MC samples for better forecasting
# SAMPLE_LENGTH=128                  # Longer sample continuations
# TEMPERATURE=1.0
# TOP_P=0.9
# MAX_COMPUTATION_TIME=60.0          # Longer computation budget

# Uncomment for FAST configuration:
# EIG_CONFIG="fast"
# LAMBDA_COST=0.1                    # Higher threshold = less reasoning
# BEAM_SIZE=4                        # Fewer answer candidates
# MC_SAMPLES=3                       # Fewer MC samples
# SAMPLE_LENGTH=32                   # Shorter sample continuations
# TEMPERATURE=1.0
# TOP_P=0.9
# MAX_COMPUTATION_TIME=15.0          # Shorter time budget

# Uncomment for EXPERIMENTAL configuration:
# EIG_CONFIG="experimental"
# LAMBDA_COST=0.03                   # Very low threshold for maximum reasoning
# BEAM_SIZE=16                       # Maximum answer candidates
# MC_SAMPLES=10                      # Maximum MC samples
# SAMPLE_LENGTH=256                  # Very long sample continuations
# TEMPERATURE=0.8                    # Lower temperature for more focused sampling
# TOP_P=0.95                        # Higher top-p for more diversity
# MAX_COMPUTATION_TIME=120.0         # Extended computation budget

echo "📋 EIG Reasoning Configuration ($EIG_CONFIG):"
echo "   Model: $MODEL_NAME"
echo "   Lambda cost threshold (λ): $LAMBDA_COST"
echo "   Beam size: $BEAM_SIZE"
echo "   Monte Carlo samples: $MC_SAMPLES"
echo "   Sample length: $SAMPLE_LENGTH"
echo "   Temperature: $TEMPERATURE"
echo "   Top-p: $TOP_P"
echo "   Max computation time: $MAX_COMPUTATION_TIME seconds"
echo "   Output path: $OUTPUT_PATH"
echo ""
echo "🎯 MATHEMATICAL FOUNDATION:"
echo "   EIG_t = H_t - E[H_{t+1} | H_t]"
echo "   Continue reasoning iff EIG_t > λ ($LAMBDA_COST)"
echo "   Maximizes mutual information per unit cost"
echo ""
echo "🔧 BEHAVIOR:"
echo "   - Estimates answer posterior entropy H_t via beam search"
echo "   - Forecasts future entropy E[H_{t+1}] via Monte Carlo sampling"  
echo "   - Continues reasoning only when information gain exceeds cost"
echo "   - Adapts continuation prompts based on information gain magnitude"

OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_NAME,dtype=float16,max_model_len=32768,gpu_memory_utilization=0.9" \
    --tasks openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path $OUTPUT_PATH \
    --log_samples \
    --verbosity DEBUG \
    --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=6,scale_func_name=expected_information_gain_reasoning,lambda_cost=$LAMBDA_COST,beam_size=$BEAM_SIZE,mc_samples=$MC_SAMPLES,sample_length=$SAMPLE_LENGTH,temperature=$TEMPERATURE,top_p=$TOP_P,max_computation_time=$MAX_COMPUTATION_TIME,debug=true" 

echo "✅ EIG reasoning evaluation completed at $(date)"

echo "📊 Collecting EIG reasoning metrics..."
python - <<'PY'
try:
    import sys
    sys.path.insert(0, '.')
    import lm_eval.budget_forcing.scalers as scalers
    print("\n" + "="*80)
    print("📊 EIG REASONING FINAL METRICS")
    print("="*80)
    scalers.print_eig_metrics()
    print("="*80)
except Exception as e:
    print(f"⚠️  Could not collect EIG metrics: {e}")
PY

echo "📊 Final results saved to: $OUTPUT_PATH"

deactivate 