import tvm
from tvm import relay, meta_schedule
from tvm.target.target import Target
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
import tvm.contrib.graph_executor as graph_executor


def tune(mod: tvm.IRModule, params, target: Target, work_dir: str, max_trials: int):
    with meta_schedule.Profiler() as profiler:
        database = meta_schedule.relay_integration.tune_relay(
            mod=mod,
            target=target,
            params=params,
            work_dir=work_dir,
            max_trials_global=max_trials,
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
