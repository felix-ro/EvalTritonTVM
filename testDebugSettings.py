import os
import torch
import torch._dynamo
import torch._inductor.config


os.environ["TORCH_COMPILE_DEBUG"] = "1"
torch._inductor.config.debug = True

@torch._dynamo.optimize()
def addrelu(a, b):
    return torch.relu(torch.add(a, b))

addrelu(torch.randn(128, 8192), torch.randn(128, 8192))