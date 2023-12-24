# HEAVILY BASED ON https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py

"""
Matrix Multiplication
=====================

Using:
* Block-level matrix multiplications.
* Multi-dimensional pointer arithmetics.
* Program re-ordering for improved L2 cache hit rate.
* Automatic performance tuning

Heavily based on:
https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py
"""
import torch

import triton
from triton_matmuls import matmul_untuned, matmul_tuned


def test():
    """
        Unit Test
        ---------
        We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).
    """
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul_untuned(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


def perf(M, N, K, ms):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton-tuned-default', 'triton-tuned-advanced', "triton-tuned-max", 'triton-untuned'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton Default Tuning", "Triton Advanced Tuning", "Triton Max Tuning",
                    "Triton No Tuning"],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('orange', '-'), ('grey', '-'), ('red', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton-tuned-default':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_tuned(a, b, tuning_level="default"),
                                                     quantiles=quantiles)
    if provider == 'triton-tuned-advanced':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_tuned(a, b, tuning_level="advanced"),
                                                     quantiles=quantiles)
    if provider == "triton-tuned-max": 
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_tuned(a, b, tuning_level="max"),
                                                     quantiles=quantiles)
    if provider == 'triton-untuned':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_untuned(a, b), quantiles=quantiles)
    return perf(M, N, K, ms), perf(M, N, K, max_ms), perf(M, N, K, min_ms)


def main():
    test()
    benchmark.run(show_plots=True, print_data=True, save_path=".")


if __name__ == "__main__":
    main()
