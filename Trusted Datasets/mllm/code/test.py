import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# If CUDA is available, print detailed GPU information
if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
        print(f"  - Memory Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
        print(f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory/1024**2:.2f} MB")
        print(f"  - CUDA Capability: {torch.cuda.get_device_capability(i)}")
else:
    print("CUDA is not available. Please check your hardware and software setup.")
