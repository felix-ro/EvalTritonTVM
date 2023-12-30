import torch  # IMPORT TORCH BEFORE TVM TO AVOID SYMBOL CLASH
import numpy as np
import tvm
from tvm import relay, auto_scheduler
from tvm.relay import testing

from utils import getImage, export_library

MODEL_NAME = "resnet50"
# #################### CONFIGURE BEFORE RUNNING #####################
WORK_DIR = "Results/TVM-Ansor/resnet50/P100/"
# TARGET_NAME = "llvm -num-cores 16 -mcpu=skylake"
# TARGET_NAME = "nvidia/nvidia-a100"
TARGET_NAME = "nvidia/tesla-p100"
MAX_TRIALS = 50
# ###################################################################


def tuneAnsor(tasks, task_weights, log_file):
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=MAX_TRIALS,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


def compile(log_file, mod, target, params):
    # Compile with the history best
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            return relay.build_module.build(mod, target=target, params=params)


def createGraphExecutor(target, lib, input_shape, dtype):
    # Create graph executor
    print("Create Graph Executor...")
    dev = tvm.device(str(target), 0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("input0", data_tvm)
    return module, dev


def main():
    # model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True)
    # model.eval()

    # # We grab the TorchScripted model via tracing
    # input_shape = [1, 3, 224, 224]
    # input_data = torch.randn(input_shape)
    # scripted_model = torch.jit.trace(model, input_data).eval()

    # # Creating Relay Graph
    # img = getImage()
    # input_name = "input0"
    # shape_list = [(input_name, img.shape)]
    # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # Selecting target and preparing logging
    target = tvm.target.Target(TARGET_NAME)
    batch_size = 1
    layout = "NHWC"
    dtype = "float32"
    input_shape = (1, 224, 224, 3)
    image_shape = (224, 224, 3)
    log_file = "%s-%s-B%d-%s.json" % (MODEL_NAME, layout, batch_size, target.kind.name)

    mod, params = relay.testing.resnet.get_workload(
        num_layers=50,
        batch_size=batch_size,
        layout=layout,
        dtype=dtype,
        image_shape=image_shape,
    )

    # Extract tasks
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    # tune the model using Ansor
    tuneAnsor(tasks, task_weights, log_file)

    # compile model with history best
    lib = compile(log_file, mod, target, params)

    export_library(lib=lib, model_name=MODEL_NAME, work_dir=WORK_DIR, max_trials=MAX_TRIALS, target_name=TARGET_NAME)

    # create graph executor
    module, dev = createGraphExecutor(target, lib, input_shape, dtype)

    # Evaluate
    print("Evaluate inference time cost...")
    res = module.benchmark(dev, repeat=3, min_repeat_ms=500)
    print(res)
    fileName = "Results/TVM-Ansor/" + MODEL_NAME + "/" + MODEL_NAME + "-" + TARGET_NAME + ".txt"
    f = open(fileName, "w")
    f.write(str(res))
    f.close()


if __name__ == "__main__":
    main()
