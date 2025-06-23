#!/bin/bash
#SBATCH --job-name=bror_comparative
#SBATCH --output=logs/bror_comparative_%j.out
#SBATCH --error=logs/bror_comparative_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# =====================================================================
# BROR Comprehensive Comparative Evaluation
# =====================================================================
# 
# This script runs multiple BROR configurations plus baseline methods
# for comprehensive comparison and analysis.
# 
# Configurations tested:
# 1. Baseline (fixed budget forcing)
# 2. BROR Conservative (high cost, fast execution)
# 3. BROR Balanced (standard cost, balanced performance)
# 4. BROR Aggressive (low cost, maximum reasoning)
# 
# Use case: Research comparison, parameter sensitivity analysis,
# comprehensive evaluation for publication or decision-making.
# =====================================================================

echo "Starting BROR Comprehensive Comparative Evaluation at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Create logs directory
mkdir -p logs

# Environment setup
source venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Common configuration
MODEL="simplescaling/s1.1-1.5B"
TASK="aime24_nofigures"
BATCH_SIZE="auto"
MAX_GEN_TOKS=32768
MAX_TOKENS_THINKING="auto"
THINKING_N_IGNORE=6

# Create master output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_OUTPUT_DIR="results/bror_comparative_${TIMESTAMP}"
mkdir -p "$MASTER_OUTPUT_DIR"

echo "Comparative evaluation will test multiple configurations:"
echo "1. Baseline (fixed budget forcing)"
echo "2. BROR Conservative (high cost, fast execution)"
echo "3. BROR Balanced (standard cost, balanced performance)"
echo "4. BROR Aggressive (low cost, maximum reasoning)"
echo ""

# Function to run evaluation with specific configuration
run_evaluation() {
    local config_name="$1"
    local gen_kwargs="$2"
    local description="$3"
    
    echo "================================================================"
    echo "Running $config_name evaluation"
    echo "Description: $description"
    echo "Gen kwargs: $gen_kwargs"
    echo "================================================================"
    
    local output_dir="$MASTER_OUTPUT_DIR/${config_name}"
    mkdir -p "$output_dir"
    
    local start_time=$(date)
    echo "Started: $start_time"
    
    python -m lm_eval \
        --model hf \
        --model_args pretrained="$MODEL",dtype=float16 \
        --tasks "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --apply_chat_template \
        --gen_kwargs "$gen_kwargs" \
        --output_path "$output_dir/results.json" \
        --log_samples \
        --show_config \
        --verbosity INFO 2>&1 | tee "$output_dir/evaluation.log"
    
    local exit_code=$?
    local end_time=$(date)
    
    echo "Completed: $end_time"
    echo "Exit code: $exit_code"
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $config_name evaluation completed successfully"
        
        # Save evaluation metadata
        cat > "$output_dir/metadata.json" << EOF
{
    "configuration": "$config_name",
    "description": "$description",
    "start_time": "$start_time",
    "end_time": "$end_time",
    "exit_code": $exit_code,
    "gen_kwargs": "$gen_kwargs"
}
EOF
        
    else
        echo "❌ $config_name evaluation failed with exit code $exit_code"
    fi
    
    echo ""
    return $exit_code
}

# Configuration 1: Baseline (fixed budget forcing)
BASELINE_KWARGS="max_gen_toks=$MAX_GEN_TOKS,max_tokens_thinking=$MAX_TOKENS_THINKING,thinking_n_ignore=$THINKING_N_IGNORE"
run_evaluation "baseline" "$BASELINE_KWARGS" "Fixed budget forcing (S1 baseline)"

# Configuration 2: BROR Conservative
CONSERVATIVE_KWARGS="max_gen_toks=$MAX_GEN_TOKS,max_tokens_thinking=$MAX_TOKENS_THINKING,thinking_n_ignore=$THINKING_N_IGNORE,scale_func_name=bayes_risk_optimal_reasoning,cost_per_step=0.02,ensemble_size=6,mc_samples=4,sample_length=32,max_reasoning_steps=30,calibration_alpha=0.8,max_computation_time=20.0"
run_evaluation "bror_conservative" "$CONSERVATIVE_KWARGS" "High cost, efficient execution"

# Configuration 3: BROR Balanced
BALANCED_KWARGS="max_gen_toks=$MAX_GEN_TOKS,max_tokens_thinking=$MAX_TOKENS_THINKING,thinking_n_ignore=$THINKING_N_IGNORE,scale_func_name=bayes_risk_optimal_reasoning,cost_per_step=0.01,ensemble_size=8,mc_samples=6,sample_length=48,max_reasoning_steps=50,calibration_alpha=1.0,max_computation_time=30.0"
run_evaluation "bror_balanced" "$BALANCED_KWARGS" "Standard cost, balanced performance"

# Configuration 4: BROR Aggressive
AGGRESSIVE_KWARGS="max_gen_toks=$MAX_GEN_TOKS,max_tokens_thinking=$MAX_TOKENS_THINKING,thinking_n_ignore=$THINKING_N_IGNORE,scale_func_name=bayes_risk_optimal_reasoning,cost_per_step=0.005,ensemble_size=12,mc_samples=8,sample_length=64,max_reasoning_steps=75,calibration_alpha=1.2,calibration_beta=0.1,max_computation_time=45.0"
run_evaluation "bror_aggressive" "$AGGRESSIVE_KWARGS" "Low cost, maximum reasoning"

echo "================================================================"
echo "COMPREHENSIVE COMPARATIVE ANALYSIS"
echo "================================================================"

# Generate comparative analysis
python -c "
import json
import os
from pathlib import Path

master_dir = Path('$MASTER_OUTPUT_DIR')
configs = ['baseline', 'bror_conservative', 'bror_balanced', 'bror_aggressive']

print('📊 COMPARATIVE RESULTS SUMMARY')
print('=' * 60)

results_data = {}

for config in configs:
    results_file = master_dir / config / 'results.json'
    metadata_file = master_dir / config / 'metadata.json'
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            results_data[config] = {
                'results': results.get('results', {}),
                'metadata': metadata
            }
            
        except Exception as e:
            print(f'Error reading {config}: {e}')

# Display comparative results
for config, data in results_data.items():
    print(f'\n🔬 {config.upper()}:')
    print(f'   Description: {data[\"metadata\"][\"description\"]}')
    
    task_results = data['results']
    for task, metrics in task_results.items():
        print(f'   Task {task}:')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'     {metric}: {value:.6f}')

# Comparative analysis
print(f'\n📈 COMPARATIVE ANALYSIS:')
print('=' * 40)

if len(results_data) >= 2:
    # Extract main metrics for comparison
    main_metrics = {}
    for config, data in results_data.items():
        task_results = data['results']
        for task, metrics in task_results.items():
            if task not in main_metrics:
                main_metrics[task] = {}
            main_metrics[task][config] = metrics
    
    for task, configs_metrics in main_metrics.items():
        print(f'\n📊 {task} COMPARISON:')
        
        # Find common metrics
        all_metrics = set()
        for config_metrics in configs_metrics.values():
            all_metrics.update(config_metrics.keys())
        
        for metric in sorted(all_metrics):
            if all(metric in config_metrics for config_metrics in configs_metrics.values()):
                values = {config: metrics.get(metric, 0) for config, metrics in configs_metrics.items()}
                if all(isinstance(v, (int, float)) for v in values.values()):
                    print(f'   {metric}:')
                    for config, value in values.items():
                        print(f'     {config}: {value:.6f}')
                    
                    # Highlight best performer
                    best_config = max(values, key=values.get)
                    print(f'     → Best: {best_config} ({values[best_config]:.6f})')

print(f'\n🧠 THEORETICAL INSIGHTS:')
print('=' * 40)
print('BASELINE: Fixed reasoning depth regardless of uncertainty')
print('CONSERVATIVE: Early stopping for computational efficiency')
print('BALANCED: Principled cost-benefit trade-offs')
print('AGGRESSIVE: Maximum reasoning for highest accuracy')
print('')
print('Expected trends:')
print('- Baseline → BROR: Improved reasoning allocation')
print('- Conservative → Aggressive: Higher accuracy, higher compute cost')
print('- Optimal configuration depends on accuracy/efficiency requirements')

" > "$MASTER_OUTPUT_DIR/comparative_analysis.txt"

# Display the analysis
cat "$MASTER_OUTPUT_DIR/comparative_analysis.txt"

# Generate summary report
cat > "$MASTER_OUTPUT_DIR/summary_report.md" << 'EOF'
# BROR Comprehensive Comparative Evaluation Report

## Overview
This evaluation compares Bayes-Risk-Optimal Reasoning (BROR) across multiple configurations against the baseline fixed budget forcing approach.

## Configurations Tested

### 1. Baseline
- **Method**: Fixed budget forcing (S1 baseline)
- **Description**: Standard reasoning depth without adaptation
- **Use case**: Reference performance

### 2. BROR Conservative  
- **Cost per step**: 0.02 (high cost → early stopping)
- **Ensemble size**: 6
- **MC samples**: 4
- **Description**: Efficient execution for resource-constrained scenarios
- **Use case**: Fast validation, limited computational budget

### 3. BROR Balanced
- **Cost per step**: 0.01 (standard cost)
- **Ensemble size**: 8
- **MC samples**: 6
- **Description**: Balanced accuracy-efficiency trade-off
- **Use case**: General purpose reasoning optimization

### 4. BROR Aggressive
- **Cost per step**: 0.005 (low cost → extensive reasoning)
- **Ensemble size**: 12
- **MC samples**: 8
- **Description**: Maximum accuracy through thorough reasoning
- **Use case**: Critical applications, research benchmarks

## Mathematical Framework

BROR implements optimal stopping theory for reasoning:
- **Risk of stopping**: R_stop = (1 - p_t) + C·t
- **Risk of continuing**: R_cont = C·(t+1) + (1 - E[p_{t+1}|H_t])
- **Decision rule**: Continue iff R_cont < R_stop ⟺ Δp_t > C

Where:
- p_t = P(answer correct | reasoning so far)
- Δp_t = expected improvement in correctness probability
- C = cost per reasoning step

## Expected Outcomes

1. **Accuracy**: Baseline < Conservative < Balanced < Aggressive
2. **Efficiency**: Aggressive < Balanced < Conservative < Baseline  
3. **Reasoning depth**: Baseline (fixed) < Conservative < Balanced < Aggressive
4. **Cost-effectiveness**: Balanced should provide optimal trade-off

## Key Research Questions

1. How does BROR improve over fixed budget forcing?
2. What is the optimal cost parameter for different scenarios?
3. How does reasoning depth correlate with accuracy improvements?
4. Can BROR achieve better accuracy-efficiency Pareto frontiers?

## Analysis Guidelines

1. Compare accuracy metrics across configurations
2. Analyze computational cost vs accuracy trade-offs
3. Examine reasoning depth statistics
4. Evaluate cost-effectiveness ratios
5. Identify optimal configurations for different use cases
EOF

echo ""
echo "✅ Comprehensive comparative evaluation completed!"
echo "Results saved to: $MASTER_OUTPUT_DIR/"
echo ""
echo "📋 Generated files:"
echo "  - Individual results in subdirectories"
echo "  - comparative_analysis.txt: Detailed numerical comparison"
echo "  - summary_report.md: Research summary and analysis guide"
echo ""
echo "🔬 Next steps:"
echo "1. Review comparative_analysis.txt for performance differences"
echo "2. Analyze cost-effectiveness trade-offs"
echo "3. Determine optimal configuration for your use case"
echo "4. Consider running additional parameter sweeps if needed"

echo "Script completed at $(date)" 