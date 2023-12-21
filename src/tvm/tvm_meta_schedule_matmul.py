import torch
import tvm
from tvm import relay, meta_schedule
from tvm.target.target import Target
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
import tvm.contrib.graph_executor as graph_executor
import sys

from utils import export_library, save_results

MODEL_NAME = "matmul"
# TARGET_NAME = "llvm -num-cores 16 -mcpu=skylake"
TARGET_NAME = "cuda -max_threads_per_block 1024 -max_shared_memory_per_block 49152"
WORK_DIR = "Results/TVM-MetaSchedule/matmul/"
MAX_TRIALS = 200


def tune(mod: tvm.IRModule, params, target: Target):
    with meta_schedule.Profiler() as profiler:
        database = meta_schedule.relay_integration.tune_relay(
            mod=mod,
            target=target,
            params=params,
            work_dir=WORK_DIR,
            max_trials_global=MAX_TRIALS,
        )
        lib: ExecutorFactoryModule = meta_schedule.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=target,
            params=params,
            backend='graph',
        )

    print(profiler.table())
    device = tvm.device(str(target), 0)
    graph_module = graph_executor.GraphModule(lib["default"](device))
    return graph_module, lib, profiler.table()


def build(mod: tvm.IRModule, params, target: Target):
    with tvm.transform.PassContext(opt_level=3):
        lib: ExecutorFactoryModule = relay.build_module.build(
                                            mod,
                                            target=target,
                                            params=params
                                        )
        dev = tvm.device(str(target), 0)
        graph_module = graph_executor.GraphModule(lib["default"](dev))
    return graph_module, lib


def main():
    target = tvm.target.Target(TARGET_NAME)
    build_only = False

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_only = True
        global MAX_TRIALS  # sketchy
        MAX_TRIALS = 0

    # extract ScriptFunction
    input_shape = [1, 3, 224, 224]
    tensor_a = torch.randn(input_shape)
    tensor_b = torch.randn(input_shape)
    scripted_func = torch.jit.trace(torch.matmul, (tensor_a, tensor_b))

    # Creating Relay Graph
    tensor_a_name = "tensor_a"
    tensor_b_name = "tensor_b"
    shape_list = [(tensor_a_name, tensor_a.shape), (tensor_b_name, tensor_b.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_func, shape_list)

    graph_module = None
    profile_results = None
    if build_only:
        graph_module, lib = build(mod=mod, params=params, target=target)
        export_library(lib=lib, model_name=MODEL_NAME, target_name=TARGET_NAME,
                       work_dir=WORK_DIR, max_trials=MAX_TRIALS)
    else:
        graph_module, lib, profile_results = tune(mod=mod, params=params, target=target)
        export_library(lib=lib, model_name=MODEL_NAME, target_name=TARGET_NAME,
                       work_dir=WORK_DIR, max_trials=MAX_TRIALS)

    dev = tvm.device(str(target), 0)
    result = graph_module.benchmark(dev)

    save_results(results=result, results_name="matmul", work_dir=WORK_DIR,
                 max_trials=MAX_TRIALS, target_name=TARGET_NAME)

    if profile_results is not None:
        save_results(results=profile_results, results_name="matmul-tuning-profile", work_dir=WORK_DIR,
                     max_trials=MAX_TRIALS, target_name=TARGET_NAME)
    print(result)


if __name__ == "__main__":
    main()
