# Bayes-Risk-Optimal Reasoning (BROR) Evaluation Scripts

## Overview

This directory contains comprehensive SLURM evaluation scripts for **Bayes-Risk-Optimal Reasoning (BROR)**, a novel extension that applies optimal stopping theory to LLM reasoning. BROR implements principled cost-benefit analysis to determine when to continue or stop reasoning based on expected improvements in correctness probability.

## Mathematical Foundation

BROR is based on **Bayesian decision theory** and **optimal stopping theory**:

### Core Framework
- **Belief State**: p_t = P(answer correct | reasoning so far)
- **Expected Improvement**: Δp_t = E[p_{t+1} | H_t] - p_t
- **Cost Function**: C = cost per reasoning step (in probability units)
- **Decision Rule**: Continue reasoning iff Δp_t > C

### Bayes Risk Minimization
- **Risk of stopping**: R_stop = (1 - p_t) + C·t
- **Risk of continuing**: R_cont = C·(t+1) + (1 - E[p_{t+1}|H_t])
- **Optimal decision**: Choose action that minimizes expected risk

### Key Components
1. **Bayesian Belief Estimator**: Uses ensemble methods with Monte Carlo dropout to estimate P(correct | H_t)
2. **Expected Improvement Forecaster**: Combines Monte Carlo sampling and regression to predict Δp_t
3. **Bayes Risk Calculator**: Applies optimal stopping criterion based on risk minimization

## Available Scripts

### 1. `eval_bror.sh` - Balanced Configuration
- **Runtime**: 24 hours
- **Memory**: 16GB
- **Configuration**: Balanced performance for general evaluation
- **Parameters**:
  - `cost_per_step=0.01` (standard cost threshold)
  - `ensemble_size=8` (balanced uncertainty estimation)
  - `mc_samples=6` (standard forecasting accuracy)
  - `sample_length=48` (moderate continuation length)

```bash
sbatch eval_bror.sh
```

**Use case**: General-purpose BROR evaluation with balanced accuracy-efficiency trade-off.

### 2. `eval_bror_fast.sh` - Efficient Configuration
- **Runtime**: 12 hours
- **Memory**: 12GB
- **Configuration**: Optimized for speed and resource efficiency
- **Parameters**:
  - `cost_per_step=0.02` (higher cost → early stopping)
  - `ensemble_size=6` (smaller ensemble for speed)
  - `mc_samples=4` (fewer samples for efficiency)
  - `sample_length=32` (shorter continuations)

```bash
sbatch eval_bror_fast.sh
```

**Use case**: Quick validation, resource-constrained environments, or computational budget limitations.

### 3. `eval_bror_precision.sh` - High-Accuracy Configuration
- **Runtime**: 48 hours
- **Memory**: 32GB
- **Configuration**: Maximum accuracy for critical applications
- **Parameters**:
  - `cost_per_step=0.005` (lower cost → extensive reasoning)
  - `ensemble_size=12` (larger ensemble for better estimates)
  - `mc_samples=8` (more samples for accurate forecasting)
  - `sample_length=64` (longer continuations for better prediction)

```bash
sbatch eval_bror_precision.sh
```

**Use case**: Research benchmarks, critical applications, or when computational resources are abundant.

### 4. `eval_bror_comparative.sh` - Comprehensive Analysis
- **Runtime**: 72 hours
- **Memory**: 32GB
- **Configuration**: Runs multiple configurations plus baseline
- **Tests**:
  1. Baseline (fixed budget forcing)
  2. BROR Conservative (high cost, fast execution)
  3. BROR Balanced (standard cost, balanced performance)
  4. BROR Aggressive (low cost, maximum reasoning)

```bash
sbatch eval_bror_comparative.sh
```

**Use case**: Research comparison, parameter sensitivity analysis, comprehensive evaluation for publication.

## Configuration Parameters

### Core BROR Parameters

| Parameter | Description | Range | Default | Impact |
|-----------|-------------|-------|---------|---------|
| `cost_per_step` | Marginal cost C (probability units) | 0.001-0.05 | 0.01 | Lower = more reasoning |
| `ensemble_size` | Number of ensemble members | 4-16 | 8 | Higher = better uncertainty |
| `mc_samples` | Monte Carlo samples for forecasting | 3-12 | 6 | Higher = better prediction |
| `sample_length` | Length of MC continuations | 16-96 | 48 | Longer = better forecasting |
| `max_reasoning_steps` | Maximum reasoning iterations | 20-100 | 50 | Higher = more thorough |
| `calibration_alpha` | Logistic calibration parameter α | 0.5-2.0 | 1.0 | Adjusts confidence scaling |
| `calibration_beta` | Logistic calibration parameter β | -0.5-0.5 | 0.0 | Adjusts confidence bias |
| `max_computation_time` | Time budget per computation (sec) | 10-60 | 30 | Higher = more thorough |

### Parameter Optimization Guidelines

#### Cost per Step (`cost_per_step`)
- **0.001-0.005**: Aggressive reasoning (high accuracy, high compute)
- **0.01**: Balanced reasoning (standard trade-off)
- **0.02-0.05**: Conservative reasoning (efficiency-focused)

#### Ensemble Size (`ensemble_size`)
- **4-6**: Basic uncertainty estimation
- **8-10**: Standard uncertainty quantification
- **12-16**: High-precision uncertainty estimation

#### MC Samples (`mc_samples`)
- **3-4**: Fast forecasting (may be less accurate)
- **6-8**: Balanced forecasting accuracy
- **10-12**: High-accuracy forecasting (slower)

## Usage Examples

### Basic Usage
```bash
# Standard BROR evaluation
sbatch eval_bror.sh
```

### Custom Configuration
```bash
# Modify parameters in the script or create custom gen_kwargs
lm_eval \
    --model hf \
    --model_args pretrained=simplescaling/s1.1-1.5B,dtype=float16 \
    --tasks aime24_nofigures \
    --batch_size auto \
    --apply_chat_template \
    --gen_kwargs "scale_func_name=bayes_risk_optimal_reasoning,cost_per_step=0.015,ensemble_size=10,mc_samples=7"
```

### Parameter Sweep Example
```bash
# Run multiple cost values for sensitivity analysis
for cost in 0.005 0.01 0.02; do
    sbatch --job-name="bror_cost_${cost}" --export="COST_PER_STEP=${cost}" eval_bror.sh
done
```

## Output Structure

Each script generates results in timestamped directories:

```
results/
├── bror_balanced_20240315_143022/
│   ├── results.json           # Main evaluation results
│   ├── evaluation.log         # Detailed execution log
│   └── samples.jsonl          # Individual sample results (if --log_samples)
├── bror_fast_20240315_150145/
│   └── ...
└── bror_comparative_20240315_160234/
    ├── baseline/
    ├── bror_conservative/
    ├── bror_balanced/
    ├── bror_aggressive/
    ├── comparative_analysis.txt
    └── summary_report.md
```

## Performance Analysis

### Accessing BROR Metrics
After running evaluations, you can analyze BROR-specific metrics:

```python
from lm_eval.budget_forcing.scalers import print_bror_metrics
print_bror_metrics()  # Comprehensive BROR analysis
```

### Key Metrics to Monitor
1. **Decision Statistics**: Continue vs stop rates, reasoning depth
2. **Belief State Analysis**: Belief trajectories, calibration accuracy
3. **Cost-Effectiveness**: Improvement/cost ratios, beneficial decisions
4. **Bayes Risk Analysis**: Risk minimization success, optimal decisions
5. **Timing Statistics**: Computation times by component

### Expected Performance Patterns
- **Conservative**: Fast execution, lower accuracy, early stopping
- **Balanced**: Moderate execution time, good accuracy, selective reasoning
- **Aggressive**: Slower execution, highest accuracy, extensive reasoning

## Comparison with Other Methods

### Baseline Methods
- **Fixed Budget**: Always reasons for fixed depth
- **Entropy Thresholding**: Stops based on token-level uncertainty
- **Stepwise Uncertainty**: Revises specific uncertain steps

### BROR Advantages
1. **Principled Decision Making**: Based on formal decision theory
2. **Cost-Benefit Analysis**: Explicit trade-off between accuracy and computation
3. **Adaptive Behavior**: Responds to problem difficulty and uncertainty
4. **Calibrated Uncertainty**: Ensemble methods with proper calibration
5. **Optimal Stopping**: Mathematically grounded termination criterion

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
- Reduce `ensemble_size` or `mc_samples`
- Decrease `sample_length`
- Use smaller batch sizes

#### 2. Slow Execution
- Increase `cost_per_step` for earlier stopping
- Reduce `max_computation_time`
- Use `eval_bror_fast.sh` configuration

#### 3. Poor Performance
- Check calibration parameters (`calibration_alpha`, `calibration_beta`)
- Ensure sufficient ensemble size for uncertainty estimation
- Verify cost parameter is appropriate for task difficulty

#### 4. Evaluation Failures
- Check SLURM logs in `logs/` directory
- Verify environment setup and dependencies
- Ensure sufficient memory allocation

### Debugging Commands
```bash
# Check SLURM queue status
squeue -u $USER

# Monitor resource usage
seff <job_id>

# View real-time logs
tail -f logs/bror_eval_<job_id>.out

# Check evaluation logs
less results/bror_*_*/evaluation.log
```

## Research Applications

### Parameter Studies
- **Cost Sensitivity**: How does `cost_per_step` affect accuracy vs efficiency?
- **Ensemble Analysis**: What is the optimal `ensemble_size` for different tasks?
- **Forecasting Accuracy**: How do `mc_samples` and `sample_length` impact performance?

### Comparative Studies
- **BROR vs Baseline**: Quantify improvements over fixed budget forcing
- **Configuration Comparison**: Find optimal settings for different scenarios
- **Method Comparison**: Compare with entropy thresholding, stepwise uncertainty

### Theoretical Analysis
- **Optimal Stopping**: Validate theoretical predictions in practice
- **Bayesian Inference**: Analyze belief state evolution during reasoning
- **Cost-Effectiveness**: Study Pareto frontiers of accuracy vs efficiency

## Advanced Usage

### Custom Parameter Sweeps
```bash
# Systematic parameter exploration
for cost in 0.005 0.01 0.02; do
    for ensemble in 6 8 12; do
        sbatch --job-name="bror_${cost}_${ensemble}" \
               --export="COST=${cost},ENSEMBLE=${ensemble}" \
               custom_bror_sweep.sh
    done
done
```

### Multi-Task Evaluation
```bash
# Test across multiple benchmarks
for task in aime24_nofigures gsm8k math; do
    sbatch --job-name="bror_${task}" \
           --export="TASK=${task}" \
           eval_bror.sh
done
```

### Integration with Other Methods
BROR can be combined with other uncertainty-driven methods by modifying the `scale_func_name` parameter or using ensemble approaches.

## Mathematical Insights

### Decision Theory Foundation
BROR implements optimal stopping theory, ensuring that reasoning decisions minimize expected Bayes risk under uncertainty. This provides theoretical guarantees about decision optimality.

### Uncertainty Quantification
The ensemble approach with Monte Carlo dropout provides well-calibrated uncertainty estimates, crucial for accurate belief state estimation.

### Information-Theoretic Connection
Expected improvement Δp_t can be related to expected information gain, connecting BROR to information-theoretic reasoning optimization.

### Computational Complexity
- **Belief Estimation**: O(ensemble_size × sequence_length)
- **Improvement Forecasting**: O(mc_samples × sample_length × ensemble_size)
- **Decision Computation**: O(1)

## Future Extensions

### Potential Improvements
1. **Adaptive Ensembles**: Dynamic ensemble size based on uncertainty
2. **Learned Calibration**: Train calibration parameters on validation data
3. **Hierarchical Reasoning**: Multi-level reasoning with nested BROR decisions
4. **Resource-Aware Costs**: Dynamic cost adjustment based on computational budget

### Research Directions
1. **Theoretical Analysis**: Convergence guarantees and optimality properties
2. **Empirical Studies**: Large-scale evaluation across diverse tasks
3. **Comparative Analysis**: Systematic comparison with other methods
4. **Application Studies**: Domain-specific optimization and deployment

---

## Summary

BROR evaluation scripts provide comprehensive tools for evaluating Bayes-Risk-Optimal Reasoning across different configurations and scenarios. The mathematical foundation in optimal stopping theory ensures principled reasoning decisions, while the implementation provides practical tools for research and application.

Choose the appropriate script based on your computational budget and accuracy requirements:
- **Fast**: Quick validation and resource-constrained scenarios
- **Balanced**: General-purpose evaluation with good trade-offs
- **Precision**: Maximum accuracy for critical applications
- **Comparative**: Comprehensive analysis across multiple configurations

The BROR framework represents a significant advance in uncertainty-driven reasoning optimization, providing both theoretical grounding and practical performance improvements. 