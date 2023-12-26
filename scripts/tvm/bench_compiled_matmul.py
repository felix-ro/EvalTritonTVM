import tvm
from tvm.contrib import graph_executor

import numpy as np


# #################### CONFIGURE BEFORE RUNNING #####################
# TARGET_NAME = "nvidia/tesla-p100"
TARGET_NAME = "nvidia/nvidia-a100"
WORK_DIR = "/home/fjr38/rds/hpc-work/EvalTritonTVM/Results/TVM-MetaSchedule/matmul/A100/2250-trials/"
COMPILED_LIB_NAME = "matmul-cuda-2250.so"
M = 2048
# ###################################################################


def set_up_workload(lib: tvm.runtime.Module, device: tvm.runtime.Device):
    module = graph_executor.GraphModule(lib["default"](device))

    input_tvm = tvm.nd.array(np.random.uniform((M, M)).astype(np.float16))
    input = {"data": input_tvm}
    module.set_input(**input)
    return module


def run_measurements(module: graph_executor.GraphModule, device: tvm.runtime.Device):
    ftimer = module.module.time_evaluator("run", device, number=100, repeat=100, min_repeat_ms=500)
    benchmark_res: np.ndarray = np.array(ftimer().results)  # unit = seconds
    return benchmark_res


def perf(M, N, K, s):
    return 2 * M * N * K * 1e-12 / s


def main():
    target = tvm.target.Target(TARGET_NAME)

    lib: tvm.runtime.Module = tvm.runtime.load_module(WORK_DIR + COMPILED_LIB_NAME)
    device: tvm.runtime.Device = tvm.device(str(target), 0)

    module: graph_executor.GraphModule = set_up_workload(lib=lib, device=device)
    results = run_measurements(module=module, device=device)

    file_path = f"{WORK_DIR}benchmark-results-{M}x{M}.txt"
    with open(file_path, "w") as f:
        f.write(f"Benchmark results in TFLOPs with a {M}x{M} matrix.\n")
        for res in results:
            res_tflops = perf(M, M, M, res)
            f.write(f"{res_tflops}\n")
            print(res_tflops)


if __name__ == "__main__":
    main()
