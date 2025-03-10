import torch

def clear_gpu_memory():
    """Clear GPU memory by emptying the CUDA cache."""
    torch.cuda.empty_cache()
    print("GPU memory cache cleared.")