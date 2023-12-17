import os
from timeit import Timer
import torch
import torch._dynamo
import torch._inductor.config

def timing(opt_model, model, device, reps, iters):
    print("Optimized model: ")
    for i in range(reps):
        t = Timer(lambda: opt_model(torch.randn(1,3,64,64).to(device)))
        print(t.timeit(number=iters)/iters)

    print("Unoptimized model: ")
    for i in range(reps):
        t = Timer(lambda: model(torch.randn(1,3,64,64).to(device)))
        print(t.timeit(number=iters)/iters)


def profiler(opt_model, model, device, useCuda):
    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        opt_model(torch.randn(1,3,64,64).to(device))

    print(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only= True))

    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        model(torch.randn(1,3,64,64).to(device))

    print(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only= True))


def main():
    # Settings to generate output files for the generated code
    # If this does not generate output code files run "export TORCH_COMPILE_DEBUG=1" in the terminal
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    torch._inductor.config.debug = True

    # Cuda capabilities >= 7.0 needed 
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    useCuda = False
    device = "cpu"

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.to(device)

    opt_model = torch.compile(model, backend="inductor")
    opt_model(torch.randn(1,3,64,64).to(device)) # Forces compilation bevore measuring

    timing(opt_model=opt_model, model=model, device=device, reps=10, iters=100)
    profiler(opt_model=opt_model, model=model, device=device, useCuda=useCuda)

    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    torch._inductor.config.debug = False


if __name__ == "__main__":
    main()