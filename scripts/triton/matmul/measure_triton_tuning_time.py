import torch

import triton
from triton_matmuls import matmul_tuned

M = 2048

def measure(a, b, tuning_level):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    matmul_tuned(a, b, tuning_level=tuning_level)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # records in milliseconds


def main():
    print("Measuring triton tuning time...")

    torch.manual_seed(0)
    a = torch.randn((M, M), device='cuda', dtype=torch.float16)
    b = torch.randn((M, M), device='cuda', dtype=torch.float16)
    
    tuning_levels = ["none", "default", "advanced", "max"]
    results =[]

    for level in tuning_levels: 
        results.append(measure(a, b, tuning_level=level))

    with open("/home/fjr38/rds/hpc-work/EvalTritonTVM/Results/Triton/matmul/A100/tuning_time.txt", "w") as f:
        for i, level in enumerate(tuning_levels): 
            f.write(f"{level}: {results[i]}\n")
            print(f"{level}: {results[i]}")


if __name__ == "__main__": 
    main()
