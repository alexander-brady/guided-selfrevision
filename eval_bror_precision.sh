#!/bin/bash
#SBATCH --job-name=bror_precision
#SBATCH --output=logs/bror_precision_%j.out
#SBATCH --error=logs/bror_precision_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# =====================================================================
# BROR High-Precision Evaluation - Maximum Accuracy Configuration
# =====================================================================
# 
# Configuration optimized for maximum accuracy and thorough reasoning.
# Lower cost_per_step encourages extensive reasoning exploration.
# Larger ensemble and more MC samples for better uncertainty estimation.
# 
# Use case: Critical applications, research experiments,
# or when computational resources are abundant and accuracy is paramount.
# =====================================================================

echo "Starting BROR High-Precision evaluation at $(date)"
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

# BROR High-Precision Configuration - optimized for maximum accuracy
COST_PER_STEP=0.005       # Lower cost = more aggressive reasoning
ENSEMBLE_SIZE=12          # Larger ensemble for better uncertainty estimation
MC_SAMPLES=8              # More MC samples for accurate forecasting
SAMPLE_LENGTH=64          # Longer continuations for better prediction
MAX_REASONING_STEPS=75    # Higher maximum steps for thorough reasoning
CALIBRATION_ALPHA=1.2     # Enhanced calibration for better confidence
CALIBRATION_BETA=0.1      # Slight bias adjustment for precision
MAX_COMPUTATION_TIME=45.0 # Extended time budget for thorough computation

echo "BROR High-Precision Configuration:"
echo "  Cost per step: $COST_PER_STEP (lower = more aggressive reasoning)"
echo "  Ensemble size: $ENSEMBLE_SIZE (larger = better uncertainty estimation)"
echo "  MC samples: $MC_SAMPLES (more = better forecasting)"
echo "  Sample length: $SAMPLE_LENGTH (longer = better prediction)"
echo "  Max reasoning steps: $MAX_REASONING_STEPS"
echo "  Calibration α: $CALIBRATION_ALPHA (enhanced calibration)"
echo "  Calibration β: $CALIBRATION_BETA (bias adjustment)"
echo "  Max computation time: $MAX_COMPUTATION_TIME seconds"

echo ""
echo "🎯 High-Precision Reasoning Strategy:"
echo "- Very low cost threshold encourages extensive reasoning"
echo "- Large ensemble provides robust uncertainty estimates"
echo "- Extended MC sampling improves future belief forecasting"
echo "- Enhanced calibration for better confidence estimation"
echo "- Suitable for critical applications requiring maximum accuracy"

# Construct gen_kwargs
GEN_KWARGS="max_gen_toks=$MAX_GEN_TOKS,max_tokens_thinking=$MAX_TOKENS_THINKING,thinking_n_ignore=$THINKING_N_IGNORE,scale_func_name=bayes_risk_optimal_reasoning,cost_per_step=$COST_PER_STEP,ensemble_size=$ENSEMBLE_SIZE,mc_samples=$MC_SAMPLES,sample_length=$SAMPLE_LENGTH,max_reasoning_steps=$MAX_REASONING_STEPS,calibration_alpha=$CALIBRATION_ALPHA,calibration_beta=$CALIBRATION_BETA,max_computation_time=$MAX_COMPUTATION_TIME"

echo ""
echo "Generated gen_kwargs: $GEN_KWARGS"

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/bror_precision_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting high-precision BROR evaluation..."

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

echo "High-precision BROR evaluation completed at $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ BROR High-Precision evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/"
    
    # Display extended results analysis
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo ""
        echo "📊 High-Precision Results Summary:"
        python -c "
import json
try:
    with open('$OUTPUT_DIR/results.json', 'r') as f:
        results = json.load(f)
    
    print('Task Results:')
    for task, metrics in results.get('results', {}).items():
        print(f'  {task}:')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'    {metric}: {value:.6f}')  # Higher precision display
            else:
                print(f'    {metric}: {value}')
    
    config = results.get('config', {})
    print(f'\nConfiguration Summary:')
    print(f'  Model: {config.get(\"model_args\", \"unknown\")}')
    gen_kwargs = config.get('gen_kwargs', {})
    if isinstance(gen_kwargs, dict):
        print('  BROR Parameters:')
        for key, value in gen_kwargs.items():
            if 'bror' in key.lower() or key in ['cost_per_step', 'ensemble_size', 'mc_samples']:
                print(f'    {key}: {value}')
    
except Exception as e:
    print(f'Could not parse results: {e}')
"
    fi
    
    echo ""
    echo "🔬 High-Precision Analysis Summary:"
    echo "Mathematical Framework: Enhanced Bayes risk minimization"
    echo "Decision Threshold: E[Δp_t] > $COST_PER_STEP (very low for extensive reasoning)"
    echo "Uncertainty Estimation: $ENSEMBLE_SIZE-member ensemble with advanced calibration"
    echo "Forecasting: $MC_SAMPLES Monte Carlo samples with extended continuations"
    echo ""
    echo "📈 Expected Outcomes:"
    echo "- Maximum reasoning depth for complex problems"
    echo "- Best possible accuracy within computational constraints"
    echo "- Detailed uncertainty quantification"
    echo "- Optimal stopping based on rigorous cost-benefit analysis"
    
else
    echo "❌ BROR High-Precision evaluation failed with exit code $EXIT_CODE"
    echo "Check the logs for detailed error information:"
    echo "  Evaluation log: $OUTPUT_DIR/evaluation.log"
    echo "  SLURM output: logs/bror_precision_${SLURM_JOB_ID}.out"
    echo "  SLURM error: logs/bror_precision_${SLURM_JOB_ID}.err"
fi

echo ""
echo "🧠 Mathematical Interpretation - High-Precision Mode:"
echo "This configuration implements the most thorough Bayes-optimal reasoning:"
echo "- Extensive ensemble for robust P(correct|H_t) estimation"
echo "- Comprehensive MC forecasting for accurate E[Δp_t] prediction"
echo "- Very low cost threshold promotes deep reasoning exploration"
echo "- Enhanced calibration for precise confidence assessment"
echo ""
echo "🎯 Research Applications:"
echo "- Benchmark maximum achievable accuracy"
echo "- Study reasoning depth vs accuracy relationships" 
echo "- Analyze BROR theoretical performance limits"
echo "- Compare against other state-of-the-art methods"

echo ""
echo "Script completed at $(date)"
exit $EXIT_CODE 