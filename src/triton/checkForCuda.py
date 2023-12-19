import torch

print("Cuda available: " + str(torch.cuda.is_available()))
print("Cuda device count: " + str(torch.cuda.device_count()))
print("Current device number: " + str(torch.cuda.current_device()))
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
