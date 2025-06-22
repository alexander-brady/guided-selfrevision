# Step-Wise Uncertainty: Runtime Signals & Debugging Guide

> **Audience**  Engineers who run long-form evaluations with the vLLM backend and need
>   confidence that the *step-wise uncertainty* pipeline is operating correctly as
>   well as actionable breadcrumbs when something goes wrong.

---

## 1. Key Execution Phases and Where to Look

| Phase | Main Function(s) | Key Signals | Where to Hook Extra Logs |
|-------|------------------|-------------|--------------------------|
| Prompt Encoding | `VLLM.tok_encode()` | • Prompt length<br/>• Truncation vs. `max_model_len` | Inside `tok_encode()` or immediately after building the batch in the shell-script. |
| Budget-Forcing Loop | `generate_with_budget_forcing_vllm()` | • `iteration` counter<br/>• `active_indices` length<br/>• Scaler decision (`keep_scaling`) and `selected_step` | Add `logger.debug()` inside the *while* loop; optionally expose a `debug=True` kwarg that prints JSON blobs per iteration. |
| Uncertainty Calc | `_generate_with_uncertainty_vllm()` | • Raw `logprobs` list length<br/>• Derived `uncertainty` per token | Already prints warnings when logprobs missing; add DEBUG block to dump first N `(token, p)` pairs. |
| Step Selection | `step_wise_uncertainty_driven()` | • Calculated `step_uncertainties` (list)<br/>• `selected_idx` & `selected_uncertainty` | Code already prints; ensure logger level `INFO` covers these lines. |
| Metrics Summary | `print_stepwise_metrics()` | • Global counters (success rate, failures by reason) | Call this once at the end of every SLURM script (see § 4). |

---

## 2. Runtime Log Levels

We use the root **`logger`** defined in `lm_eval.utils.eval_logger`.

* `INFO` (default in scripts) – high-level progress, step selection & scaler output.
* `DEBUG` – token-level dumps *and* per-iteration uncertainty values.

Enable full debug in any CLI by exporting

```bash
export LOGLEVEL=DEBUG  # or set --verbosity DEBUG on lm_eval CLI
```

This automatically propagates because every print in the new code goes via
`eval_logger` (or through the `debug` flag shown below).

---

## 3. Per-Problem Deep Dive

When investigating a single sample:

```python
from lm_eval.budget_forcing.vllm_core import generate_with_budget_forcing_vllm

outputs = generate_with_budget_forcing_vllm(
    llm=my_vllm,
    requests=[prompt_ids],
    max_tokens=512,
    stop_sequences=["</s>"],
    scale_func_name="step_wise_uncertainty_driven",
    debug=True,               # <-- add this kwarg locally
    # other scaler kwargs …
)
```

Implementing `debug=True` inside `generate_with_budget_forcing_vllm()` is the
recommended place to print the *entire* uncertainty vector and scaler decision
for that single prompt **without** enabling global DEBUG (keeps logs concise).

---

## 4. Hooking Global Metrics at Script End

Both `step_wise_uncertainty_driven` and `entropy_thresholding` update the
module-level dict `_STEPWISE_METRICS`.

Add the following to any evaluation script (e.g. **`eval_stepwise.sh`**) *after*
`lm_eval` finishes but **before** the virtual-env deactivates:

```bash
python - <<'PY'
import lm_eval.budget_forcing.scalers as s
s.print_stepwise_metrics()
PY
```

This prints a compact dashboard:

```
📊 STEP-WISE UNCERTAINTY COMPREHENSIVE METRICS
──────────────────────────────────────────────
Total calls: 550   Successful extractions: 545 (99.1 %)
Avg steps / call: 4.3   Steps revised: 312
Failure breakdown: {"tokenizer_decode_failed": 2, …}
…
```

Store it alongside your usual evaluation JSON for quick post-mortems.

---

## 5. WandB / TensorBoard Integration (Optional)

*Inside `generate_with_budget_forcing_vllm()`* you can emit custom metrics:

```python
if os.getenv("WANDB_STEPWISE", "0") == "1":
    wandb.log({
        "uncertainty/iteration": iteration,
        "uncertainty/selected": selected_uncertainty,
        "uncertainty/avg": float(np.mean(uncerts)),
    })
```

Enable via environment variable to avoid extra deps during normal runs.

---

## 6. Fail-Fast Checks

1. **No `logprobs`** – `_generate_with_uncertainty_vllm()` logs *WARNING* and
   falls back to uncertainty = 0.5.
2. **Scaler returns invalid tuple** – handled by `safe_wrapper()` in
   `scaler_registry.py`, automatically reverts to default (always continue).
3. **Max-Iterations** – `MAX_ITER` safeguard: printed at *INFO* if reached.

Set `export RAISE_ON_SCALER_ERROR=1` to convert any fallback -> `RuntimeError`.

---

## 7. Minimal Checklist Before Long Run

1. Run `python test_imports.py` (very fast) – ensures environment OK.
2. Run `CUDA_VISIBLE_DEVICES=0 ./test_vllm_stepwise.py --debug` to verify end-to-end.
3. Check `logs/stepwise_eval_*.out` for:
   * "FIRST STEP-WISE UNCERTAINTY ANALYSIS" block
   * Regular "📊 STEP-WISE UNCERTAINTY COMPREHENSIVE METRICS" at end.
4. Only then submit the full `sbatch` job.

These hooks will give you high-signal checkpoints without drowning you in
per-token noise unless explicitly requested.

---

*Last updated: June 2025* 