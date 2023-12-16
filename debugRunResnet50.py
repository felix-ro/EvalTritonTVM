import time
import os
import torch
import torch._dynamo
import torch._inductor.config

# Settings to generate output files for the generated code
os.environ["TORCH_COMPILE_DEBUG"] = "1"
torch._inductor.config.debug = True

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
opt_model = torch.compile(model, backend="inductor")

start = 0
end = 0
for i in range (0, 10):
    start = time.time()
    opt_model(torch.randn(1,3,64,64))
    end = time.time()
    print(end - start)

print("\n")

for i in range(0, 10):
    start = time.time()
    model(torch.randn(1,3,64,64))
    end = time.time()
    print(end - start)

print("\n")


os.environ["TORCH_COMPILE_DEBUG"] = "0"
torch._inductor.config.debug = False