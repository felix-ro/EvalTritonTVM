import tvm
from tvm.contrib import graph_executor

import numpy as np


# #################### CONFIGURE BEFORE RUNNING #####################
TARGET_NAME = "nvidia/tesla-p100"
COMPILED_LIB_PATH = "/home/fjr38/projects/EvalTritonTVM/Results/TVM-MetaSchedule/matmul/P100/matmul-cuda-200.so"
# ###################################################################


def set_up_workload(lib: tvm.runtime.Module, device: tvm.runtime.Device):
    module = graph_executor.GraphModule(lib["default"](device))

    input_tvm = tvm.nd.array(np.random.uniform((512, 512)).astype(np.float16))
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

    lib: tvm.runtime.Module = tvm.runtime.load_module(COMPILED_LIB_PATH)
    device: tvm.runtime.Device = tvm.device(str(target), 0)

    module: graph_executor.GraphModule = set_up_workload(lib=lib, device=device)
    e2e_results = run_measurements(module=module, device=device)
    print(perf(512, 512, 512, e2e_results[0]))


if __name__ == "__main__":
    main()
