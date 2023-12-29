import tvm
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
from torchvision import transforms
import numpy as np
from PIL import Image


def getImage():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = tvm.contrib.download.download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))

    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    return np.expand_dims(img, 0)


def get_simplified_target_name(target_name: str):
    if "llvm" in target_name:
        return "llvm"
    else:
        return "cuda"


def export_library(lib: ExecutorFactoryModule, model_name: str, target_name: str, work_dir: str, max_trials: int):
    simplified_target_name = get_simplified_target_name(target_name=target_name)

    # export library file (.so)
    compiled_model_name = f"{model_name}-{simplified_target_name}-{max_trials}.so"
    lib.export_library(f"{work_dir}/{compiled_model_name}")
    print(f"Exported compiled library to {compiled_model_name}")


def save_results(results: any, results_name: str, work_dir: str,
                 max_trials: int, target_name: str):
    simplified_target_name = get_simplified_target_name(target_name=target_name)

    file_name = f"{work_dir}results-{results_name}-{simplified_target_name}-{max_trials}.txt"
    f = open(file_name, "w")
    result = f"Operator/Model name: {results_name}\nMax Trials: {max_trials}\n\n{results}"
    f.write(result)
    f.close
