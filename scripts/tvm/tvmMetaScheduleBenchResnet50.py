import torch  # IMPORT TORCH BEFORE TVM TO AVOID SYMBOL CLASH
import tvm
from tvm import relay
import sys

from utils import getImage, export_library, save_results
from meta_schedule_utils import tune, build

MODEL_NAME = "resnet50"
TARGET_NAME = "llvm -num-cores 16 -mcpu=skylake"
# TARGET_NAME = "cuda -max_threads_per_block 1024 -max_shared_memory_per_block 49152"
WORK_DIR = "Results/TVM-MetaSchedule/resnet50/"
MAX_TRIALS = 200


def main():
    model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True)
    model.eval()

    target = tvm.target.Target(TARGET_NAME)
    build_only = False

    # Configure build
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_only = True
        global MAX_TRIALS  # sketchy
        MAX_TRIALS = 0

    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    # Creating Relay Graph
    img = getImage()
    input_name = "input0"
    shape_list = [(input_name, img.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

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
