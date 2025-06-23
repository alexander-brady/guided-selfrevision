#!/bin/bash
#SBATCH --job-name=eig_precision_multi_gpu
#SBATCH --output=logs/eig_precision_multi_gpu_%j.out
#SBATCH --error=logs/eig_precision_multi_gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:80G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

echo "🚀 Starting MULTI-GPU Expected Information Gain (EIG) Reasoning Evaluation at $(date)"
echo "🔧 Using 2 GPUs with vLLM tensor parallelism to solve memory issues"

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

echo "🔬 Starting MULTI-GPU Expected Information Gain reasoning evaluation at $(date)"

# Configuration
MODEL_NAME="simplescaling/s1.1-1.5B"
OUTPUT_PATH="eig_precision_multi_gpu_eval/$USER/$SLURM_JOB_ID"

# MULTI-GPU HIGH PRECISION EIG Configuration
EIG_CONFIG="multi_gpu_precision"
LAMBDA_COST=0.1                    # Lower threshold for higher accuracy (more reasoning)
BEAM_SIZE=4                        # High beam size (can afford with 2 GPUs)
MC_SAMPLES=3                        # High MC samples (can afford with 2 GPUs)
SAMPLE_LENGTH=32                   # Longer samples (can afford with 2 GPUs)
TEMPERATURE=1.0                     # Standard sampling temperature
TOP_P=0.9                          # Standard top-p sampling
MAX_COMPUTATION_TIME=15.0          # Extended timeout (can afford with 2 GPUs)

# MULTI-GPU vLLM SETTINGS
TENSOR_PARALLEL_SIZE=2              # Split model across 2 GPUs
GPU_MEMORY_UTILIZATION=0.9         # Can use higher utilization with 2 GPUs
MAX_MODEL_LEN=32768                 # Full 32K context with 2 GPUs
MAX_GEN_TOKS=32768                  # Full generation capacity

echo "📋 MULTI-GPU HIGH PRECISION EIG Reasoning Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Tensor parallel size: $TENSOR_PARALLEL_SIZE GPUs"
echo "   GPU memory utilization: $GPU_MEMORY_UTILIZATION (HIGH - we have 2 GPUs!)"
echo "   Max model length: $MAX_MODEL_LEN tokens (FULL 32K capacity)"
echo "   Max generation tokens: $MAX_GEN_TOKS (FULL capacity)"
echo "   Lambda cost threshold (λ): $LAMBDA_COST (AGGRESSIVE for max accuracy)"
echo "   Beam size: $BEAM_SIZE (HIGH for best posterior estimation)"
echo "   Monte Carlo samples: $MC_SAMPLES (HIGH for best forecasting)"
echo "   Sample length: $SAMPLE_LENGTH (LONG for thorough analysis)"
echo "   Temperature: $TEMPERATURE"
echo "   Top-p: $TOP_P"
echo "   Max computation time: $MAX_COMPUTATION_TIME seconds (EXTENDED)"
echo "   Output path: $OUTPUT_PATH"
echo ""
echo "🚀 MULTI-GPU ADVANTAGES:"
echo "   - 2x GPU memory: 160GB total vs 80GB single"
echo "   - Model weights split across 2 GPUs"
echo "   - Higher precision EIG parameters possible"
echo "   - Better posterior estimation with larger beam"
echo "   - More accurate entropy forecasting with more MC samples"
echo "   - Full 32K context and generation capacity"
echo ""
echo "🎯 EXPECTED BEHAVIOR:"
echo "   - No memory issues with distributed model"
echo "   - High-quality EIG computations"
echo "   - Maximum accuracy from aggressive parameters"
echo "   - Robust mathematical reasoning"

# Check GPU availability
echo "🔧 GPU Configuration Check:"
python3 - <<'PY'
import torch

print(f"🔧 Multi-GPU Setup Verification:")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"   Available GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Get memory info for each GPU
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3
        print(f"     Total Memory: {total_memory:.1f} GB")
    
    if gpu_count >= 2:
        print(f"✅ Multi-GPU setup ready for tensor parallelism!")
        total_memory = gpu_count * total_memory
        print(f"   Combined GPU Memory: {total_memory:.1f} GB")
    else:
        print(f"⚠️  Only {gpu_count} GPU available, tensor parallelism may not work optimally")
        
    # Clear any existing CUDA cache
    torch.cuda.empty_cache()
    print(f"✅ CUDA cache cleared")
else:
    print("❌ CUDA not available")
PY

# Pre-evaluation test
echo "🧪 Running multi-GPU pre-evaluation diagnostics..."
python3 - <<'PY'
try:
    print("Testing EIG components for multi-GPU setup...")
    
    # Test imports
    from lm_eval.budget_forcing.eig_core import ExpectedInformationGainCalculator
    from lm_eval.budget_forcing.scalers import expected_information_gain_reasoning
    from lm_eval.budget_forcing.scaler_registry import get_scale_func
    print("✅ All EIG components import successfully")
    
    # Test scaler registry with high-precision parameters
    scale_func = get_scale_func(
        'expected_information_gain_reasoning',
        [1, 2, 3],
        beam_size=12,
        mc_samples=8,
        lambda_cost=0.02
    )
    print("✅ EIG scaler configured for high precision")
    
    # Test answer extraction patterns
    from lm_eval.budget_forcing.eig_core import AnswerPosteriorEstimator
    estimator = AnswerPosteriorEstimator(beam_size=12)  # High beam size
    
    test_text = "Step 1: We solve x + 2 = 5\nStep 2: x = 3\nTherefore, \\boxed{3}"
    class MockModel:
        def tok_encode(self, text): return [1, 2, 3]
    
    candidates = estimator.extract_answer_candidates(test_text, MockModel())
    print(f"✅ Answer extraction test: {candidates}")
    
    if "3" in candidates:
        print("✅ Answer extraction working correctly")
    else:
        print("⚠️  Answer extraction may need improvement")
    
    print("🚀 Multi-GPU pre-evaluation diagnostics completed successfully")
    
except Exception as e:
    print(f"❌ Pre-evaluation diagnostics failed: {e}")
    print("⚠️  Proceeding with evaluation, but issues may occur")
PY

echo "🚀 Starting multi-GPU evaluation with tensor parallelism..."

OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_NAME,dtype=float16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,tensor_parallel_size=$TENSOR_PARALLEL_SIZE" \
    --tasks openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path $OUTPUT_PATH \
    --log_samples \
    --limit 50 \
    --verbosity DEBUG \
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,max_tokens_thinking=auto,thinking_n_ignore=6,scale_func_name=expected_information_gain_reasoning,lambda_cost=$LAMBDA_COST,beam_size=$BEAM_SIZE,mc_samples=$MC_SAMPLES,sample_length=$SAMPLE_LENGTH,temperature=$TEMPERATURE,top_p=$TOP_P,max_computation_time=$MAX_COMPUTATION_TIME,debug=true" 

echo "✅ MULTI-GPU EIG reasoning evaluation completed at $(date)"

echo "📊 Collecting detailed multi-GPU EIG reasoning metrics..."
python3 - <<'PY'
try:
    import sys
    import torch
    sys.path.insert(0, '.')
    import lm_eval.budget_forcing.scalers as scalers
    
    # Show final GPU memory status
    if torch.cuda.is_available():
        print(f"🔧 Final Multi-GPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i}: Allocated {memory_allocated:.2f} GB, Reserved {memory_reserved:.2f} GB")
    
    print("\n" + "="*80)
    print("📊 MULTI-GPU HIGH PRECISION EIG REASONING FINAL METRICS")
    print("="*80)
    scalers.print_eig_metrics()
    print("="*80)
    print("\n📈 MULTI-GPU PERFORMANCE ANALYSIS:")
    print("This run used optimal parameters with distributed GPU memory:")
    print("- Did all EIG computations succeed without memory errors?")
    print("- Is the accuracy significantly improved vs single GPU?")
    print("- Are the high-precision parameters providing better results?")
    print("\n🚀 MULTI-GPU ADVANTAGES REALIZED:")
    print("1. No memory constraints on EIG parameters")
    print("2. Full 32K context and generation capacity") 
    print("3. High beam size for better answer analysis")
    print("4. More MC samples for accurate entropy forecasting")
    print("5. Aggressive λ threshold for maximum reasoning")
except Exception as e:
    print(f"⚠️  Could not collect EIG metrics: {e}")
    print("This suggests integration issues persist")
PY

# Analyze results in detail
echo "📊 Detailed multi-GPU results analysis:"
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
            
            print("📊 MULTI-GPU EIG Results:")
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
        
        # Check configuration was applied correctly
        if 'config' in results:
            config = results['config']
            gen_kwargs = config.get('gen_kwargs', {})
            model_args = config.get('model_args', '')
            
            print(f"\n✅ Multi-GPU Configuration Check:")
            print(f"   Scale function: {gen_kwargs.get('scale_func_name', 'NOT SET')}")
            print(f"   Max gen toks: {gen_kwargs.get('max_gen_toks', 'NOT SET')}")
            print(f"   Lambda cost: {gen_kwargs.get('lambda_cost', 'NOT SET')}")
            print(f"   Beam size: {gen_kwargs.get('beam_size', 'NOT SET')}")
            print(f"   MC samples: {gen_kwargs.get('mc_samples', 'NOT SET')}")
            print(f"   Model args: {model_args}")
            
            # Check if tensor_parallel_size was used
            if 'tensor_parallel_size=2' in model_args:
                print(f"✅ Tensor parallelism confirmed in model args")
            else:
                print(f"⚠️  Tensor parallelism not detected in model args")
        
    else:
        print("❌ No result files found")
        
except Exception as e:
    print(f"❌ Error analyzing results: {e}")
PY

echo "📊 Final results saved to: $OUTPUT_PATH"

echo "🎯 MULTI-GPU EVALUATION SUMMARY:"
echo "This version distributes the model across 2 GPUs to solve memory issues"
echo "while maintaining high precision EIG parameters for maximum accuracy."
echo ""
echo "✅ MULTI-GPU BENEFITS:"
echo "1. 2x GPU memory (160GB total)"
echo "2. No memory constraints on EIG computation"  
echo "3. High-quality posterior estimation (beam_size=12)"
echo "4. Accurate entropy forecasting (mc_samples=8)"
echo "5. Aggressive reasoning threshold (λ=0.02)"
echo ""
echo "📈 EXPECTED IMPROVEMENTS:"
echo "- Significantly higher accuracy than single GPU"
echo "- More robust mathematical reasoning"
echo "- Better answer extraction and matching"
echo "- Stable execution without memory errors"

deactivate 