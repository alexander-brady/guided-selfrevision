#!/bin/bash
#SBATCH --job-name=eig_precision_fixed
#SBATCH --output=logs/eig_precision_fixed_%j.out
#SBATCH --error=logs/eig_precision_fixed_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:80G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

echo "🎯 Starting FIXED HIGH PRECISION Expected Information Gain (EIG) Reasoning Evaluation at $(date)"

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

echo "🔬 Starting FIXED HIGH PRECISION Expected Information Gain reasoning evaluation at $(date)"

# Configuration
MODEL_NAME="simplescaling/s1.1-1.5B"
OUTPUT_PATH="eig_precision_fixed_eval/$USER/$SLURM_JOB_ID"

# IMPROVED PRECISION EIG Configuration - More reliable parameters
EIG_CONFIG="precision_fixed"
LAMBDA_COST=0.05                    # Balanced threshold (not too aggressive)
BEAM_SIZE=6                         # Moderate beam size (reduced from 12 for stability)
MC_SAMPLES=4                        # Moderate MC samples (reduced from 8 for stability)
SAMPLE_LENGTH=64                    # Moderate sample length (reduced from 128 for stability)
TEMPERATURE=1.0                     # Standard sampling temperature
TOP_P=0.9                          # Standard top-p sampling
MAX_COMPUTATION_TIME=30.0          # Moderate timeout (reduced from 60 for stability)

echo "📋 FIXED HIGH PRECISION EIG Reasoning Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Lambda cost threshold (λ): $LAMBDA_COST (BALANCED for reliability)"
echo "   Beam size: $BEAM_SIZE (MODERATE for stability)"
echo "   Monte Carlo samples: $MC_SAMPLES (MODERATE for reliability)"
echo "   Sample length: $SAMPLE_LENGTH (MODERATE for stability)"
echo "   Temperature: $TEMPERATURE"
echo "   Top-p: $TOP_P"
echo "   Max computation time: $MAX_COMPUTATION_TIME seconds (MODERATE for stability)"
echo "   Output path: $OUTPUT_PATH"
echo ""
echo "🔧 RELIABILITY IMPROVEMENTS:"
echo "   - Balanced λ threshold for stable reasoning decisions"
echo "   - Moderate beam size to avoid vLLM issues"
echo "   - Fewer MC samples to reduce computation failures"
echo "   - Shorter samples to avoid timeout issues"
echo "   - Enhanced error handling and fallback strategies"
echo ""
echo "🎯 EXPECTED BEHAVIOR:"
echo "   - More reliable EIG computations"
echo "   - Better answer extraction for math problems"
echo "   - Improved vLLM integration stability"
echo "   - Higher accuracy through more robust reasoning"
echo ""
echo "⚠️  DEBUGGING FEATURES:"
echo "   - Enhanced logging for first few problems"
echo "   - Better error reporting and recovery"
echo "   - Improved answer pattern matching"
echo "   - More robust vLLM logprob handling"

# Pre-evaluation test
echo "🧪 Running pre-evaluation diagnostics..."
python3 - <<'PY'
try:
    print("Testing EIG components...")
    
    # Test imports
    from lm_eval.budget_forcing.eig_core import ExpectedInformationGainCalculator
    from lm_eval.budget_forcing.scalers import expected_information_gain_reasoning
    from lm_eval.budget_forcing.scaler_registry import get_scale_func
    print("✅ All EIG components import successfully")
    
    # Test scaler registry
    scale_func = get_scale_func(
        'expected_information_gain_reasoning',
        [1, 2, 3],
        beam_size=6,
        mc_samples=4,
        lambda_cost=0.05
    )
    print("✅ EIG scaler can be retrieved from registry")
    
    # Test answer extraction patterns
    from lm_eval.budget_forcing.eig_core import AnswerPosteriorEstimator
    estimator = AnswerPosteriorEstimator()
    
    test_text = "Step 1: We solve x + 2 = 5\nStep 2: x = 3\nTherefore, \\boxed{3}"
    class MockModel:
        def tok_encode(self, text): return [1, 2, 3]
    
    candidates = estimator.extract_answer_candidates(test_text, MockModel())
    print(f"✅ Answer extraction test: {candidates}")
    
    if "3" in candidates:
        print("✅ Answer extraction working correctly")
    else:
        print("⚠️  Answer extraction may need improvement")
    
    print("🚀 Pre-evaluation diagnostics completed successfully")
    
except Exception as e:
    print(f"❌ Pre-evaluation diagnostics failed: {e}")
    print("⚠️  Proceeding with evaluation, but issues may occur")
PY

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

echo "✅ FIXED HIGH PRECISION EIG reasoning evaluation completed at $(date)"

echo "📊 Collecting detailed EIG reasoning metrics..."
python3 - <<'PY'
try:
    import sys
    sys.path.insert(0, '.')
    import lm_eval.budget_forcing.scalers as scalers
    print("\n" + "="*80)
    print("📊 FIXED HIGH PRECISION EIG REASONING FINAL METRICS")
    print("="*80)
    scalers.print_eig_metrics()
    print("="*80)
    print("\n📈 IMPROVEMENT ANALYSIS:")
    print("Compare this run with the previous results:")
    print("- Did EIG computations succeed more often?")
    print("- Is the accuracy improved?")
    print("- Are the reasoning patterns more coherent?")
    print("\n🔧 FURTHER DEBUGGING:")
    print("If accuracy is still low, check:")
    print("1. Generated answer samples for quality")
    print("2. Answer extraction success rate")
    print("3. vLLM model reasoning capability") 
    print("4. OpenAI Math task answer format expectations")
except Exception as e:
    print(f"⚠️  Could not collect EIG metrics: {e}")
    print("This suggests the EIG integration still has issues")
PY

# Analyze results in detail
echo "📊 Detailed results analysis:"
python3 - <<'PY'
import json
import glob
import os

try:
    result_files = glob.glob(f"{os.environ.get('OUTPUT_PATH', '.')}/results_*.json")
    if result_files:
        with open(result_files[0], 'r') as f:
            results = json.load(f)
        
        if 'results' in results and 'openai_math' in results['results']:
            math_results = results['results']['openai_math']
            
            print("📊 FIXED EIG Results:")
            print(f"   Exact Match: {math_results.get('exact_match,none', 'N/A')}")
            print(f"   Extracted Answers: {math_results.get('extracted_answers,none', 'N/A')}")
            
            # Check for EIG-specific metrics
            eig_metrics = [k for k in math_results.keys() if 'stepwise' in k.lower() or 'eig' in k.lower()]
            if eig_metrics:
                print("\n📈 EIG-Specific Metrics:")
                for metric in eig_metrics:
                    print(f"   {metric}: {math_results[metric]}")
            else:
                print("\n⚠️  No EIG-specific metrics found in results")
        
        # Check configuration
        if 'config' in results:
            gen_kwargs = results['config'].get('gen_kwargs', {})
            if 'scale_func_name' in gen_kwargs:
                print(f"\n✅ Scale function used: {gen_kwargs['scale_func_name']}")
            else:
                print(f"\n❌ No scale function found in config")
        
    else:
        print("❌ No result files found")
        
except Exception as e:
    print(f"❌ Error analyzing results: {e}")
PY

echo "📊 Final results saved to: $OUTPUT_PATH"

echo "🎯 EVALUATION SUMMARY:"
echo "This improved version should be more reliable than the original."
echo "Check the accuracy metrics and EIG computation statistics above."
echo "If accuracy is still very low, the issue may be fundamental to the"
echo "model's mathematical reasoning capability rather than the EIG implementation."

deactivate 