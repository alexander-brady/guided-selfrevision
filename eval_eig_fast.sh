#!/bin/bash
#SBATCH --job-name=eig_fast_eval
#SBATCH --output=logs/eig_fast_%j.out
#SBATCH --error=logs/eig_fast_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:64g
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

echo "🚀 Starting FAST Expected Information Gain (EIG) Reasoning Evaluation at $(date)"

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

echo "⚡ Starting FAST Expected Information Gain reasoning evaluation at $(date)"

# Configuration
MODEL_NAME="simplescaling/s1.1-1.5B"
OUTPUT_PATH="eig_fast_eval/$USER/$SLURM_JOB_ID"

# FAST EIG Configuration - Optimized for speed
EIG_CONFIG="fast"
LAMBDA_COST=0.1                     # Higher threshold = less reasoning (faster)
BEAM_SIZE=4                         # Fewer answer candidates (faster)
MC_SAMPLES=3                        # Fewer MC samples (faster)
SAMPLE_LENGTH=32                    # Shorter sample continuations (faster)
TEMPERATURE=1.0                     # Standard sampling temperature
TOP_P=0.9                          # Standard top-p sampling
MAX_COMPUTATION_TIME=15.0          # Shorter time budget (faster timeout)

echo "📋 FAST EIG Reasoning Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Lambda cost threshold (λ): $LAMBDA_COST (HIGHER = less reasoning)"
echo "   Beam size: $BEAM_SIZE (REDUCED for speed)"
echo "   Monte Carlo samples: $MC_SAMPLES (REDUCED for speed)"
echo "   Sample length: $SAMPLE_LENGTH (REDUCED for speed)"
echo "   Temperature: $TEMPERATURE"
echo "   Top-p: $TOP_P"
echo "   Max computation time: $MAX_COMPUTATION_TIME seconds (REDUCED for speed)"
echo "   Output path: $OUTPUT_PATH"
echo ""
echo "⚡ FAST MODE OPTIMIZATIONS:"
echo "   - Higher λ threshold reduces reasoning iterations"
echo "   - Smaller beam size reduces answer candidate analysis"
echo "   - Fewer MC samples speed up entropy forecasting"
echo "   - Shorter sample length reduces generation time"
echo "   - Reduced timeout prevents long computations"
echo ""
echo "🎯 EXPECTED BEHAVIOR:"
echo "   - Faster execution with moderate reasoning quality"
echo "   - Good for time-constrained scenarios or quick prototyping"
echo "   - May miss some beneficial reasoning opportunities"

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

echo "✅ FAST EIG reasoning evaluation completed at $(date)"

echo "📊 Collecting EIG reasoning metrics..."
python - <<'PY'
try:
    import sys
    sys.path.insert(0, '.')
    import lm_eval.budget_forcing.scalers as scalers
    print("\n" + "="*80)
    print("📊 FAST EIG REASONING FINAL METRICS")
    print("="*80)
    scalers.print_eig_metrics()
    print("="*80)
except Exception as e:
    print(f"⚠️  Could not collect EIG metrics: {e}")
PY

echo "📊 Final results saved to: $OUTPUT_PATH"

deactivate 