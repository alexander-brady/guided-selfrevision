#!/bin/bash
#SBATCH --job-name=s1_eval_array
#SBATCH --output=logs/third_eval_%j_%a.out
#SBATCH --error=logs/third_eval_%j_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:64g
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --array=1-3,5-10

MODEL="s1.1-1.5B" # From simplescaling repo
MAX_BUDGET_FORCING_STEPS=${SLURM_ARRAY_TASK_ID} # Use the array task ID as the max budget forcing steps
WAIT_TOKEN="Wait"

ABLATION="third"

# SCALE_FUNCTION="default"
# SCALE_FUNCTION="entropy_thresholding"
SCALE_FUNCTION="uncertainty_driven_reevaluation"
OUTPUT_PATH="../../results/$USER/${SCALE_FUNCTION}_ablation_${ABLATION}_${MAX_BUDGET_FORCING_STEPS}_${SLURM_JOB_ID}"

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

echo "Job started at $(date)"

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
  pip install -e ../.. --quiet
  pip install -e .[math,vllm] --quiet
  echo "Installed dependencies in virtual environment at $VENV_PATH at $(date)"
else
  echo "Using existing virtual environment at $VENV_PATH"
fi

echo "Starting $MODEL model evaluation at $(date)"
echo "Using scale function: $SCALE_FUNCTION"
echo "Wait token: $WAIT_TOKEN, with max budget forcing steps: $MAX_BUDGET_FORCING_STEPS"

OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR="gpt-4o-mini" lm_eval \
    --model vllm \
    --model_args "pretrained=simplescaling/$MODEL,dtype=float16,max_length=32768" \
    --tasks openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path "$OUTPUT_PATH" \
    --log_samples \
    --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=$MAX_BUDGET_FORCING_STEPS,thinking_n_ignore_str=$WAIT_TOKEN,scale_func_name=$SCALE_FUNCTION,ablation=$ABLATION" \
    # --limit 10

echo "Job completed at $(date)"
echo "Results saved to "$OUTPUT_PATH""

deactivate