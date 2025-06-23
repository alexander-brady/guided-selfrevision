#!/bin/bash
#SBATCH --job-name=eig_precision_eval
#SBATCH --output=logs/eig_precision_%j.out
#SBATCH --error=logs/eig_precision_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:80g
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

echo "🎯 Starting HIGH PRECISION Expected Information Gain (EIG) Reasoning Evaluation at $(date)"

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

echo "🔬 Starting HIGH PRECISION Expected Information Gain reasoning evaluation at $(date)"

# Configuration
MODEL_NAME="simplescaling/s1.1-1.5B"
OUTPUT_PATH="eig_precision_eval/$USER/$SLURM_JOB_ID"

# HIGH PRECISION EIG Configuration - Optimized for maximum accuracy
EIG_CONFIG="high_precision"
LAMBDA_COST=0.02                    # Lower threshold = more reasoning (higher accuracy)
BEAM_SIZE=12                        # More answer candidates (better posterior estimation)
MC_SAMPLES=8                        # More MC samples (better entropy forecasting)
SAMPLE_LENGTH=128                   # Longer sample continuations (better analysis)
TEMPERATURE=1.0                     # Standard sampling temperature
TOP_P=0.9                          # Standard top-p sampling
MAX_COMPUTATION_TIME=60.0          # Extended time budget (allows thorough analysis)

echo "📋 HIGH PRECISION EIG Reasoning Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Lambda cost threshold (λ): $LAMBDA_COST (LOWER = more reasoning)"
echo "   Beam size: $BEAM_SIZE (INCREASED for better analysis)"
echo "   Monte Carlo samples: $MC_SAMPLES (INCREASED for better forecasting)"
echo "   Sample length: $SAMPLE_LENGTH (INCREASED for thorough analysis)"
echo "   Temperature: $TEMPERATURE"
echo "   Top-p: $TOP_P"
echo "   Max computation time: $MAX_COMPUTATION_TIME seconds (EXTENDED for thoroughness)"
echo "   Output path: $OUTPUT_PATH"
echo ""
echo "🎯 HIGH PRECISION OPTIMIZATIONS:"
echo "   - Lower λ threshold allows more beneficial reasoning"
echo "   - Larger beam size improves answer posterior estimation"
echo "   - More MC samples enhance entropy forecasting accuracy"
echo "   - Longer sample length enables deeper analysis"
echo "   - Extended timeout allows complete computations"
echo ""
echo "🔬 EXPECTED BEHAVIOR:"
echo "   - Maximum reasoning quality and accuracy"
echo "   - Longer execution time due to thorough analysis"
echo "   - Ideal for complex mathematical problems"
echo "   - Best performance on challenging reasoning tasks"
echo ""
echo "⚠️  COMPUTATIONAL REQUIREMENTS:"
echo "   - Higher memory usage (32GB allocated)"
echo "   - Extended GPU memory (80GB allocated)"  
echo "   - Longer wall-time (48 hours allocated)"
echo "   - Suitable for high-end compute environments"

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

echo "✅ HIGH PRECISION EIG reasoning evaluation completed at $(date)"

echo "📊 Collecting detailed EIG reasoning metrics..."
python - <<'PY'
try:
    import sys
    sys.path.insert(0, '.')
    import lm_eval.budget_forcing.scalers as scalers
    print("\n" + "="*80)
    print("📊 HIGH PRECISION EIG REASONING FINAL METRICS")
    print("="*80)
    scalers.print_eig_metrics()
    print("="*80)
    print("\n📈 ANALYSIS RECOMMENDATIONS:")
    print("Compare results with other methods using:")
    print("- Average reasoning steps per problem")
    print("- Information gain distribution analysis")
    print("- Computational efficiency (accuracy/time)")
    print("- Error type analysis on incorrect answers")
except Exception as e:
    print(f"⚠️  Could not collect EIG metrics: {e}")
PY

echo "📊 Final results saved to: $OUTPUT_PATH"

deactivate 