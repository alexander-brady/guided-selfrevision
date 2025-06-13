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


module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

source ./.env

if [ ! -d "$SCRATCH/pmlr/.venv" ]; then
  python3 -m venv "$SCRATCH/csnlp/.venv"
  echo "Virtual environment created at $SCRATCH/csnlp/.venv"
fi

source "$SCRATCH/csnlp/.venv/bin/activate"

echo "Starting s1.1-1.5B model evaluation"
echo "Job started at $(date)"

export HF_HOME="$SCRATCH/csnlp/model_cache"

cd eval/lm-evaluation-harness

pip install --upgrade pip --quiet
pip install -e .[math,vllm] --quiet

OPENAI_API_KEY=$OPENAI_API_KEY PROCESSOR=$PROCESSOR VLLM_WORKER_MULTIPROC_METHOD=spawn lm_eval \
    --model vllm \
    --model_args pretrained=simplescaling/s1.1-1.5B,dtype=float16,tensor_parallel_size=1 \
    --tasks aime24_nofigures,openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path s1.1_1.5B_eval/$USER/$SLURM_JOB_ID \
    --log_samples \
    --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto"

echo "Job completed at $(date)"

deactivate 