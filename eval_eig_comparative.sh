#!/bin/bash
#SBATCH --job-name=eig_comparative_eval
#SBATCH --output=logs/eig_comparative_%j.out
#SBATCH --error=logs/eig_comparative_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:80g
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

echo "🔬 Starting COMPARATIVE Expected Information Gain (EIG) Reasoning Evaluation at $(date)"
echo "This script runs multiple EIG configurations for comprehensive comparison"

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

echo "📊 Starting COMPARATIVE EIG reasoning evaluation at $(date)"

# Configuration
MODEL_NAME="simplescaling/s1.1-1.5B"
BASE_OUTPUT_PATH="eig_comparative_eval/$USER/$SLURM_JOB_ID"

# Evaluation configurations to compare
declare -A CONFIGS

# Configuration 1: FAST
CONFIGS["fast_lambda_cost"]=0.1
CONFIGS["fast_beam_size"]=4
CONFIGS["fast_mc_samples"]=3
CONFIGS["fast_sample_length"]=32
CONFIGS["fast_max_computation_time"]=15.0

# Configuration 2: BALANCED
CONFIGS["balanced_lambda_cost"]=0.05
CONFIGS["balanced_beam_size"]=8
CONFIGS["balanced_mc_samples"]=5
CONFIGS["balanced_sample_length"]=64
CONFIGS["balanced_max_computation_time"]=30.0

# Configuration 3: PRECISION
CONFIGS["precision_lambda_cost"]=0.02
CONFIGS["precision_beam_size"]=12
CONFIGS["precision_mc_samples"]=8
CONFIGS["precision_sample_length"]=128
CONFIGS["precision_max_computation_time"]=60.0

# Configuration 4: BASELINE (for comparison)
CONFIGS["baseline_scale_func"]="entropy_thresholding"
CONFIGS["baseline_threshold"]=0.5

echo "🎯 COMPARATIVE EVALUATION PLAN:"
echo "   Model: $MODEL_NAME"
echo "   Base output path: $BASE_OUTPUT_PATH"
echo "   Configurations: FAST, BALANCED, PRECISION, BASELINE"
echo "   Total estimated runtime: ~6-12 hours"
echo ""

# Function to run evaluation with specific configuration
run_eig_evaluation() {
    local config_name=$1
    local output_path="$BASE_OUTPUT_PATH/$config_name"
    
    echo "================================================================"
    echo "🚀 Running $config_name configuration at $(date)"
    echo "================================================================"
    
    if [ "$config_name" = "baseline" ]; then
        echo "📋 BASELINE Configuration (entropy_thresholding):"
        echo "   Scale function: entropy_thresholding"
        echo "   Threshold: ${CONFIGS["baseline_threshold"]}"
        echo "   Output path: $output_path"
        
        OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR lm_eval \
            --model vllm \
            --model_args "pretrained=$MODEL_NAME,dtype=float16,max_model_len=32768,gpu_memory_utilization=0.9" \
            --tasks openai_math \
            --batch_size auto \
            --apply_chat_template \
            --output_path $output_path \
            --log_samples \
            --verbosity DEBUG \
            --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=6,scale_func_name=${CONFIGS["baseline_scale_func"]},threshold=${CONFIGS["baseline_threshold"]},debug=true"
    else
        echo "📋 $config_name EIG Configuration:"
        echo "   Lambda cost: ${CONFIGS[${config_name}_lambda_cost]}"
        echo "   Beam size: ${CONFIGS[${config_name}_beam_size]}"
        echo "   MC samples: ${CONFIGS[${config_name}_mc_samples]}"
        echo "   Sample length: ${CONFIGS[${config_name}_sample_length]}"
        echo "   Max computation time: ${CONFIGS[${config_name}_max_computation_time]}s"
        echo "   Output path: $output_path"
        
        OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR lm_eval \
            --model vllm \
            --model_args "pretrained=$MODEL_NAME,dtype=float16,max_model_len=32768,gpu_memory_utilization=0.9" \
            --tasks openai_math \
            --batch_size auto \
            --apply_chat_template \
            --output_path $output_path \
            --log_samples \
            --verbosity DEBUG \
            --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=6,scale_func_name=expected_information_gain_reasoning,lambda_cost=${CONFIGS[${config_name}_lambda_cost]},beam_size=${CONFIGS[${config_name}_beam_size]},mc_samples=${CONFIGS[${config_name}_mc_samples]},sample_length=${CONFIGS[${config_name}_sample_length]},temperature=1.0,top_p=0.9,max_computation_time=${CONFIGS[${config_name}_max_computation_time]},debug=true"
    fi
    
    echo "✅ $config_name configuration completed at $(date)"
    echo ""
}

# Run all configurations
echo "🚀 Starting comparative evaluation sequence..."

# Run baseline first for reference
run_eig_evaluation "baseline"

# Run EIG configurations
run_eig_evaluation "fast"
run_eig_evaluation "balanced" 
run_eig_evaluation "precision"

echo "================================================================"
echo "🎉 COMPARATIVE EVALUATION COMPLETED at $(date)"
echo "================================================================"

echo "📊 RESULTS SUMMARY:"
echo "   Baseline (entropy_thresholding): $BASE_OUTPUT_PATH/baseline"
echo "   FAST EIG: $BASE_OUTPUT_PATH/fast"
echo "   BALANCED EIG: $BASE_OUTPUT_PATH/balanced"
echo "   PRECISION EIG: $BASE_OUTPUT_PATH/precision"
echo ""

echo "📊 Collecting final EIG reasoning metrics..."
python - <<'PY'
try:
    import sys
    sys.path.insert(0, '.')
    import lm_eval.budget_forcing.scalers as scalers
    print("\n" + "="*80)
    print("📊 COMPARATIVE EIG REASONING FINAL METRICS")
    print("="*80)
    scalers.print_eig_metrics()
    print("="*80)
except Exception as e:
    print(f"⚠️  Could not collect EIG metrics: {e}")
PY

echo "📈 ANALYSIS SCRIPT:"
cat > "${BASE_OUTPUT_PATH}/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
"""
Comparative analysis script for EIG evaluation results.
"""
import json
import os
from pathlib import Path

def analyze_results(base_path):
    configs = ["baseline", "fast", "balanced", "precision"]
    results = {}
    
    for config in configs:
        config_path = Path(base_path) / config
        result_files = list(config_path.glob("*.json"))
        
        if result_files:
            with open(result_files[0], 'r') as f:
                data = json.load(f)
                results[config] = {
                    "accuracy": data.get("results", {}).get("openai_math", {}).get("acc,none", 0),
                    "samples": len(data.get("samples", [])),
                    "config_path": str(config_path)
                }
    
    print("🔬 COMPARATIVE RESULTS ANALYSIS")
    print("=" * 50)
    
    for config, result in results.items():
        print(f"{config.upper()}:")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Samples: {result['samples']}")
        print(f"  Path: {result['config_path']}")
        print()
    
    # Find best performing configuration
    if results:
        best_config = max(results.items(), key=lambda x: x[1]["accuracy"])
        print(f"🏆 BEST PERFORMANCE: {best_config[0].upper()} with {best_config[1]['accuracy']:.3f} accuracy")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    analyze_results(base_path)
EOF

chmod +x "${BASE_OUTPUT_PATH}/analyze_results.py"

echo "📊 NEXT STEPS:"
echo "1. Run the analysis script:"
echo "   cd $BASE_OUTPUT_PATH && python3 analyze_results.py"
echo ""
echo "2. View detailed EIG metrics:"
echo "   python3 -c \"from lm_eval.budget_forcing.scalers import print_eig_metrics; print_eig_metrics()\""
echo ""
echo "3. Compare reasoning traces in the generated samples"
echo ""
echo "4. Analyze computational efficiency (accuracy per token/time)"

deactivate 