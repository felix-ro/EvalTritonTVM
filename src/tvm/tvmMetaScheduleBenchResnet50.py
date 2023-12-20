import torch  # IMPORT TORCH BEFORE TVM TO AVOID SYMBOL CLASH
import tvm
from tvm import relay, meta_schedule
from tvm.target.target import Target
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
import tvm.contrib.graph_executor as graph_executor
from typing import Tuple
from utils import getImage

MODEL_NAME = "resnet50"
TARGET_NAME = "llvm -num-cores 16 -mcpu=skylake"  # "cuda"
WORK_DIR = "Results/TVM-MetaSchedule/"


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

    # must exit profiler scope
    print(profiler.table())

    # 'llvm' -> tvm.cpu(0)
    device = tvm.device(str(target), 0)
    # if (TARGET_NAME == "cuda"):
    #     source_code = lib.imported_modules[0].get_source()
    # else:
    #     source_code = lib.get_source()
    graph_module = graph_executor.GraphModule(lib["default"](device))
    return graph_module  # , source_code


def build(mod: tvm.IRModule, params, input_shape: Tuple[int], target: Target):
    with tvm.transform.PassContext(opt_level=3):
        lib: ExecutorFactoryModule = relay.build_module.build(
                                            mod,
                                            target=target,
                                            params=params
                                        )
        dev = tvm.device(str(target), 0)
        graph_module = graph_executor.GraphModule(lib["default"](dev))
    return graph_module


def main():
    model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True)
    model.eval()

    target = tvm.target.Target(TARGET_NAME)

    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    # Creating Relay Graph
    img = getImage()
    input_name = "input0"
    shape_list = [(input_name, img.shape)]
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
