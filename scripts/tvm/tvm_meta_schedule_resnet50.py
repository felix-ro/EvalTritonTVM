import tvm
from tvm.relay import testing
import sys

from utils import export_library, save_results
from meta_schedule_utils import tune, build

MODEL_NAME = "resnet50"

# #################### CONFIGURE BEFORE RUNNING #####################
# TARGET_NAME = "llvm -num-cores 16 -mcpu=skylake"
# TARGET_NAME = "nvidia/nvidia-a100"
TARGET_NAME = "nvidia/tesla-p100"
WORK_DIR = "Results/TVM-MetaSchedule/resnet50/"
MAX_TRIALS = 10000
# ###################################################################


def main():
    target = tvm.target.Target(TARGET_NAME)
    build_only = False

    # Configure build
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_only = True
        global MAX_TRIALS  # sketchy
        MAX_TRIALS = 0

    # Selecting target and preparing logging
    target = tvm.target.Target(TARGET_NAME)
    batch_size = 1
    layout = "NHWC"
    dtype = "float32"
    image_shape = (224, 224, 3)

    mod, params = testing.resnet.get_workload(
        num_layers=50,
        batch_size=batch_size,
        layout=layout,
        dtype=dtype,
        image_shape=image_shape,
    )

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
    save_results(results=result, results_name="resnet50", work_dir=WORK_DIR,
                 max_trials=MAX_TRIALS, target_name=TARGET_NAME)
    if profile_results is not None:
        save_results(results=profile_results, results_name="resnet50-tuning-profile",
                     work_dir=WORK_DIR, max_trials=MAX_TRIALS, target_name=TARGET_NAME)
    print(result)


if __name__ == "__main__":
    main()
