#!/bin/bash
#SBATCH --job-name=s1_eval_job
#SBATCH --output=logs/s1_eval_%j.out
#SBATCH --error=logs/s1_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:16g
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

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
  pip install -e .[math,vllm] --quiet
else
  echo "Using existing virtual environment at $VENV_PATH"
fi

echo "Starting s1.1-1.5B model evaluation at $(date)"

OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR lm_eval \
    --model hf \
    --model_args "pretrained=simplescaling/s1.1-1.5B,dtype=float16,max_length=32768" \
    --tasks aime24_nofigures,openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path s1_1_1_5B_eval/$USER/$SLURM_JOB_ID \
    --log_samples \
    --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=6,thinking_n_ignore_str=Wait,scale_func_name=entropy_thresholding" 

echo "Job completed at $(date)"

deactivate 