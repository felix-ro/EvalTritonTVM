import torch  # IMPORT TORCH BEFORE TVM TO AVOID SYMBOL CLASH
import tvm
from tvm import relay, meta_schedule
from tvm.target.target import Target
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
import tvm.contrib.graph_executor as graph_executor
from typing import Tuple

MODEL_NAME = "matmul"
TARGET_NAME = "llvm -num-cores 16 -mcpu=skylake"
WORK_DIR = "Results/TVM-MetaSchedule/matmul/"

def tune(mod: tvm.IRModule, params, input_shape: Tuple[int], target: Target):
    with meta_schedule.Profiler() as profiler:
        database = meta_schedule.relay_integration.tune_relay(
            mod=mod,
            target=target,
            params=params,
            work_dir=WORK_DIR,
            max_trials_global=200,
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
    return graph_module 


def main():
    target = tvm.target.Target(TARGET_NAME)

    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 224, 224]
    input_data1 = torch.randn(input_shape)
    input_data2 = torch.randn(input_shape)
    scripted_model = torch.jit.trace(torch.matmul, (input_data1, input_data2))

    # Creating Relay Graph
    input_name1 = "input1"
    input_name2 = "input2"
    shape_list = [(input_name1, input_data1.shape), (input_name2, input_data2.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    graph_module = tune(mod=mod, params=params, input_shape=input_shape, target=target)

    # type(source_code)
    # print(source_code)
    # graph_module = build(mod=mod, params=params, input_shape=input_shape, target=target)

    dev = tvm.device(str(target), 0)
    result = graph_module.benchmark(dev)

    print("results: ")
    print(result)


if __name__ == "__main__":
    main()