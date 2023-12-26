import tvm
from tvm import relay

import torch
import sys

from utils import export_library, save_results
from meta_schedule_utils import tune, build

MODEL_NAME = "matmul"

# #################### CONFIGURE BEFORE RUNNING #####################
# TARGET_NAME = "llvm -num-cores 16 -mcpu=skylake"
# TARGET_NAME = "nvidia/tesla-p100"
TARGET_NAME = "nvidia/nvidia-a100"
WORK_DIR = "Results/TVM-MetaSchedule/matmul/A100/2250-trials/"
MAX_TRIALS = 2250
M = 2048  # dimensions of the square matrix
# ###################################################################


def main():
    target = tvm.target.Target(TARGET_NAME)
    build_only = False

    # Configure build
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_only = True
        global MAX_TRIALS  # sketchy
        MAX_TRIALS = 0

    # extract ScriptFunction
    tensor_a = torch.randn((M, M), device="cuda", dtype=torch.float16)
    tensor_b = torch.randn((M, M), device="cuda", dtype=torch.float16)
    scripted_func = torch.jit.trace(torch.matmul, (tensor_a, tensor_b))

    # Creating Relay Graph
    tensor_a_name = "tensor_a"
    tensor_b_name = "tensor_b"
    shape_list = [(tensor_a_name, tensor_a.shape), (tensor_b_name, tensor_b.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_func, shape_list)

    # Build/tune the model and export library
    graph_module = None
    profile_results = None
    if build_only:
        graph_module, lib = build(mod=mod, params=params, target=target)
        export_library(lib=lib, model_name=MODEL_NAME, target_name=TARGET_NAME,
                       work_dir=WORK_DIR, max_trials=MAX_TRIALS)
    else:
        graph_module, lib, profile_results = tune(mod=mod, params=params, target=target,
                                                  work_dir=WORK_DIR, max_trials=MAX_TRIALS)
        export_library(lib=lib, model_name=MODEL_NAME, target_name=TARGET_NAME,
                       work_dir=WORK_DIR, max_trials=MAX_TRIALS)

    # Benchmark the model
    dev = tvm.device(str(target), 0)
    result = graph_module.benchmark(dev)

    # Save the results
    save_results(results=result, results_name="matmul", work_dir=WORK_DIR,
                 max_trials=MAX_TRIALS, target_name=TARGET_NAME)
    if profile_results is not None:
        save_results(results=profile_results, results_name="matmul-tuning-profile", work_dir=WORK_DIR,
                     max_trials=MAX_TRIALS, target_name=TARGET_NAME)
    print(result)


if __name__ == "__main__":
    main()
