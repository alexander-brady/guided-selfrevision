# Expected Information Gain (EIG) Reasoning - Evaluation Scripts

This document provides a complete guide to the EIG reasoning evaluation scripts following the same pattern as the existing budget forcing evaluation scripts.

## Available Scripts

### 1. `eval_eig.sh` - Balanced Configuration (Recommended)
**Default configuration for most use cases**

```bash
sbatch eval_eig.sh
```

**Configuration:**
- Lambda cost (λ): 0.05 (balanced threshold)
- Beam size: 8 (standard answer candidates)
- MC samples: 5 (standard forecasting)
- Sample length: 64 (balanced analysis depth)
- Max computation time: 30 seconds
- SLURM time: 24 hours

**Use when:** General evaluation, comparing with other methods, standard research

### 2. `eval_eig_fast.sh` - Fast Execution
**Optimized for speed and quick prototyping**

```bash
sbatch eval_eig_fast.sh
```

**Configuration:**
- Lambda cost (λ): 0.1 (higher threshold = less reasoning)
- Beam size: 4 (fewer candidates for speed)
- MC samples: 3 (minimal forecasting)
- Sample length: 32 (shorter analysis)
- Max computation time: 15 seconds
- SLURM time: 12 hours

**Use when:** Quick testing, limited compute budget, time-constrained scenarios

### 3. `eval_eig_precision.sh` - High Precision
**Maximum accuracy with extended computational resources**

```bash
sbatch eval_eig_precision.sh
```

**Configuration:**
- Lambda cost (λ): 0.02 (lower threshold = more reasoning)
- Beam size: 12 (more answer candidates)
- MC samples: 8 (enhanced forecasting)
- Sample length: 128 (deeper analysis)
- Max computation time: 60 seconds
- SLURM time: 48 hours
- Memory: 32GB, GPU memory: 80GB

**Use when:** Maximum accuracy needed, sufficient compute resources, final evaluation

### 4. `eval_eig_comparative.sh` - Comprehensive Comparison
**Runs multiple configurations in sequence for benchmarking**

```bash
sbatch eval_eig_comparative.sh
```

**Runs:**
1. Baseline (entropy_thresholding)
2. Fast EIG configuration
3. Balanced EIG configuration  
4. Precision EIG configuration

**SLURM time:** 72 hours
**Output:** Separate results for each configuration + analysis script

**Use when:** Comprehensive evaluation, research paper results, method comparison

## Mathematical Foundation

All EIG scripts implement the core information-theoretic principle:

```
EIG_t = H_t - E[H_{t+1} | H_t]
Continue reasoning iff EIG_t > λ
```

Where:
- **H_t**: Current answer posterior entropy
- **E[H_{t+1} | H_t]**: Expected entropy after one more reasoning step  
- **λ (lambda_cost)**: Information cost threshold

## Quick Start Guide

### For most users (recommended):
```bash
# Standard evaluation
sbatch eval_eig.sh
```

### For quick testing:
```bash
# Fast evaluation for prototyping
sbatch eval_eig_fast.sh
```

### For research/publication:
```bash
# Comprehensive comparison
sbatch eval_eig_comparative.sh
```

## Parameter Tuning Guide

### Lambda Cost (λ) - Information Threshold
- **Lower values (0.01-0.03)**: More reasoning, higher accuracy, slower execution
- **Balanced values (0.04-0.06)**: Good trade-off for most problems
- **Higher values (0.08-0.15)**: Less reasoning, faster execution, may miss insights

### Beam Size - Answer Candidates
- **4-6**: Fast execution, basic answer analysis
- **8-10**: Balanced performance for most problems
- **12-16**: Thorough analysis for complex problems

### MC Samples - Entropy Forecasting
- **3-4**: Minimal forecasting, fastest execution
- **5-7**: Good forecasting accuracy for most cases
- **8-12**: High accuracy forecasting, slower execution

## Output Analysis

After running any script, analyze results with:

### 1. Basic Results
```bash
# View accuracy and basic metrics
cd <output_path>
cat results.json | grep -A 5 "openai_math"
```

### 2. EIG Metrics (Detailed)
```bash
# View comprehensive EIG reasoning metrics
python3 -c "from lm_eval.budget_forcing.scalers import print_eig_metrics; print_eig_metrics()"
```

### 3. Comparative Analysis (for comparative script)
```bash
# Run automated analysis
cd <comparative_output_path>
python3 analyze_results.py
```

## Integration with Existing Workflow

These EIG scripts follow the exact same pattern as existing evaluation scripts:

- **Same SLURM configuration** as `eval.sh`, `eval_stepwise.sh`
- **Same environment setup** and dependency management
- **Same output format** for easy comparison
- **Compatible with existing analysis tools**

## Troubleshooting

### Common Issues:

1. **Memory errors**: Use `eval_eig_fast.sh` or reduce `beam_size`
2. **Timeout errors**: Increase `max_computation_time` or use higher `lambda_cost`
3. **Import errors**: Ensure the EIG modules are properly installed
4. **SLURM errors**: Check cluster resource availability

### Debug Mode:
Add `--verbose` to the `lm_eval` command in any script for detailed logging.

## Research Applications

### Experiment Design Templates:

**Accuracy vs Computational Cost:**
```bash
# Run multiple lambda values
sbatch eval_eig_fast.sh     # λ=0.1
sbatch eval_eig.sh          # λ=0.05  
sbatch eval_eig_precision.sh # λ=0.02
```

**Method Comparison:**
```bash
sbatch eval_eig_comparative.sh  # Runs all methods
```

**Parameter Sensitivity Analysis:**
Edit script parameters and run systematic variations.

## Next Steps

1. **Run basic evaluation**: `sbatch eval_eig.sh`
2. **Monitor job**: `squeue -u $USER`
3. **Analyze results**: Follow output analysis guide
4. **Compare with baselines**: Use comparative script
5. **Tune parameters**: Based on initial results

The EIG reasoning implementation provides a principled, mathematically grounded approach to uncertainty-driven test-time reasoning that integrates seamlessly with your existing evaluation pipeline. 