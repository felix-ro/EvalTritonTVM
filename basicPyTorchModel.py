import time
import torch
import torch._dynamo
import torch._inductor.config

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

torch._dynamo.reset()
opt_tvm_model = torch.compile(model, backend="tvm")

start = 0
end = 0
for i in range (0, 10):
    start = time.time()
    opt_tvm_model(torch.randn(1,3,64,64))
    end = time.time()
    print(end - start)