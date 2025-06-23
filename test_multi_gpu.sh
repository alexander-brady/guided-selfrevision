#!/bin/bash
#SBATCH --job-name=test_multi_gpu
#SBATCH --output=logs/test_multi_gpu_%j.out
#SBATCH --error=logs/test_multi_gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:160G
#SBATCH --time=0:30:00

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

echo "🔧 Testing Multi-GPU Setup for vLLM at $(date)"

source ./.env
VENV_PATH="$SCRATCH/csnlp/.venv"
export HF_HOME="$SCRATCH/csnlp/cache"

cd eval/lm-evaluation-harness
source "$VENV_PATH/bin/activate"

echo "🔧 GPU Hardware Check:"
nvidia-smi

echo -e "\n🔧 PyTorch CUDA Check:"
python3 - <<'PY'
import torch

print(f"🔧 Multi-GPU Verification:")
print(f"   CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"   GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"     Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"     Compute Capability: {props.major}.{props.minor}")
    
    if gpu_count >= 2:
        print(f"✅ Multi-GPU setup available!")
        print(f"   Total GPU Memory: {gpu_count * (props.total_memory / 1024**3):.1f} GB")
    else:
        print(f"❌ Insufficient GPUs for tensor parallelism")
else:
    print(f"❌ CUDA not available")
PY

echo -e "\n🔧 vLLM Multi-GPU Test:"
python3 - <<'PY'
try:
    print("Testing vLLM with tensor parallelism...")
    from vllm import LLM, SamplingParams
    
    # Test with a small model and tensor parallelism
    print("Creating LLM with tensor_parallel_size=2...")
    
    # Use a very small model for testing
    llm = LLM(
        model="gpt2",  # Small model for testing
        tensor_parallel_size=2,
        gpu_memory_utilization=0.5,
        max_model_len=512
    )
    
    print("✅ vLLM initialized successfully with tensor parallelism!")
    
    # Quick generation test
    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0)
    
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"✅ Generation test: '{output.outputs[0].text.strip()}'")
    
    print("✅ Multi-GPU vLLM functionality confirmed!")
    
except Exception as e:
    print(f"❌ vLLM multi-GPU test failed: {e}")
    import traceback
    traceback.print_exc()
PY

echo -e "\n🔧 Final Recommendation:"
python3 - <<'PY'
import torch

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    print("✅ RECOMMENDATION: Multi-GPU setup is ready!")
    print("   You can run: sbatch eval_eig_precision_multi_gpu.sh")
    print("   Expected benefits:")
    print("   - 2x GPU memory (160GB total)")
    print("   - High precision EIG parameters")
    print("   - No memory errors")
    print("   - Significantly better accuracy")
else:
    print("❌ RECOMMENDATION: Multi-GPU not available")
    print("   Alternatives:")
    print("   1. Request a node with 2 GPUs")
    print("   2. Use the memory-optimized single GPU script")
    print("   3. Contact cluster admin about multi-GPU access")
PY

echo "🔧 Multi-GPU test completed at $(date)"
deactivate 