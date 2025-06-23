#!/bin/bash
#SBATCH --job-name=eig_debug_eval
#SBATCH --output=logs/eig_debug_%j.out
#SBATCH --error=logs/eig_debug_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:80g
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL


module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

echo "🔬 Starting EIG DEBUG Evaluation at $(date)"

source ./.env

VENV_PATH="$SCRATCH/csnlp/.venv"
export HF_HOME="$SCRATCH/csnlp/cache"

cd eval/lm-evaluation-harness

# Use existing virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run the regular EIG evaluation script first to set up the environment"
    exit 1
fi

source "$VENV_PATH/bin/activate"

echo "🔬 Starting EIG DEBUG evaluation at $(date)"

# Configuration
MODEL_NAME="simplescaling/s1.1-1.5B"
OUTPUT_PATH="eig_debug_eval/$USER/$SLURM_JOB_ID"

# CONSERVATIVE EIG Configuration for debugging
LAMBDA_COST=0.02                     # Higher threshold for debugging (less aggressive reasoning)
BEAM_SIZE=12                         # Smaller beam for faster debugging
MC_SAMPLES=8                        # Fewer samples for faster debugging
SAMPLE_LENGTH=128                    # Shorter samples for debugging
TEMPERATURE=1.0                     # Standard temperature
TOP_P=0.9                          # Standard top-p
MAX_COMPUTATION_TIME=30.0          # Shorter timeout for debugging

echo "📋 EIG DEBUG Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Lambda cost threshold (λ): $LAMBDA_COST (CONSERVATIVE for debugging)"
echo "   Beam size: $BEAM_SIZE (REDUCED for speed)"
echo "   Monte Carlo samples: $MC_SAMPLES (REDUCED for speed)"
echo "   Sample length: $SAMPLE_LENGTH (REDUCED for speed)"
echo "   Temperature: $TEMPERATURE"
echo "   Top-p: $TOP_P"
echo "   Max computation time: $MAX_COMPUTATION_TIME seconds (REDUCED for debugging)"
echo "   Output path: $OUTPUT_PATH"
echo ""
echo "🔬 DEBUG OPTIMIZATIONS:"
echo "   - Conservative λ threshold to reduce reasoning iterations"
echo "   - Smaller beam size for faster computation"
echo "   - Fewer MC samples for faster debugging"
echo "   - Shorter timeout to avoid hanging"
echo "   - Enhanced logging and error reporting"
echo ""
echo "🎯 DEBUGGING OBJECTIVES:"
echo "   - Verify EIG function is being called correctly"
echo "   - Check answer extraction and matching"
echo "   - Identify vLLM integration issues"
echo "   - Test with a small subset of problems first"

# Run evaluation with ENHANCED DEBUGGING
echo "🚀 Starting evaluation with detailed debugging..."

OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_NAME,dtype=float16,max_model_len=32768,gpu_memory_utilization=0.9" \
    --tasks openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path $OUTPUT_PATH \
    --limit 150 \
    --log_samples \
    --verbosity DEBUG \
    --gen_kwargs "max_gen_toks=2048,max_tokens_thinking=auto,thinking_n_ignore=3,scale_func_name=expected_information_gain_reasoning,lambda_cost=$LAMBDA_COST,beam_size=$BEAM_SIZE,mc_samples=$MC_SAMPLES,sample_length=$SAMPLE_LENGTH,temperature=$TEMPERATURE,top_p=$TOP_P,max_computation_time=$MAX_COMPUTATION_TIME,debug=true" 

echo "✅ EIG DEBUG evaluation completed at $(date)"

echo "📊 Collecting EIG debugging metrics..."
python3 - <<'PY'
try:
    import sys
    sys.path.insert(0, '.')
    import lm_eval.budget_forcing.scalers as scalers
    print("\n" + "="*80)
    print("📊 EIG DEBUG EVALUATION METRICS")
    print("="*80)
    scalers.print_eig_metrics()
    print("="*80)
    print("\n🔬 DEBUG ANALYSIS:")
    print("If you see EIG computations above, the basic integration is working.")
    print("If accuracy is still low, the issue is likely in:")
    print("- Answer extraction patterns")
    print("- Mathematical reasoning quality")
    print("- Answer matching logic in the evaluation task")
    print("\n📈 NEXT STEPS:")
    print("1. Check the generated samples in the output directory")
    print("2. Compare generated answers with expected answers")
    print("3. Verify the OpenAI Math task answer processing")
except Exception as e:
    print(f"⚠️  Could not collect EIG metrics: {e}")
    print("This suggests the EIG scaler might not have been called at all.")
    print("Check the evaluation logs for:")
    print("- EIG initialization messages")
    print("- Scale function errors") 
    print("- vLLM model loading issues")
PY

echo "📊 Final results saved to: $OUTPUT_PATH"

# Show a preview of results
echo "📋 Results preview:"
if [ -f "$OUTPUT_PATH/results_*.json" ]; then
    python3 -c "
import json
import glob
import os
result_files = glob.glob('$OUTPUT_PATH/results_*.json')
if result_files:
    with open(result_files[0], 'r') as f:
        results = json.load(f)
    if 'results' in results and 'openai_math' in results['results']:
        math_results = results['results']['openai_math']
        print(f'📊 OpenAI Math Results:')
        for metric, value in math_results.items():
            print(f'   {metric}: {value}')
    else:
        print('❌ No openai_math results found in output')
else:
    print('❌ No result files found')
"
else
    echo "❌ No results file found"
fi

echo "🔬 EIG Debug evaluation complete. Check logs for detailed analysis."

deactivate 