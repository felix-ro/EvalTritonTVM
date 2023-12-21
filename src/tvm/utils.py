import tvm
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
