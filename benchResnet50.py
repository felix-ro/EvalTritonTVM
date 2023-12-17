import os
from timeit import Timer
import torch
import torch._dynamo
import torch._inductor.config

PATH = "Results/"
MODEL_NAME = "resnet50"

def timing(opt_model, model, device, reps, iters):
    fileName = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + "-optimized.txt"
    f = open(fileName, "w")
    results = ""
    print("Optimized model: ")
    for i in range(reps):
        t = Timer(lambda: opt_model(torch.randn(1,3,64,64).to(device)))
        print(t.timeit(number=iters)/iters)
        results += str(t.timeit(number=iters)/iters) + "\n"
    f.write(results)

    fileName = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + "-unoptimized.txt"
    f = open(fileName, "w")
    results = ""
    print("Unoptimized model: ")
    for i in range(reps):
        t = Timer(lambda: model(torch.randn(1,3,64,64).to(device)))
        print(t.timeit(number=iters)/iters)
        results += str(t.timeit(number=iters)/iters) + "\n"
    f.write(results)


def profiler(opt_model, model, device, useCuda):
    fileName = PATH + MODEL_NAME + "/" + "profile-" + MODEL_NAME + "-optimized.txt"
    f = open(fileName, "w")
    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        opt_model(torch.randn(1,3,64,64).to(device))

    f.write(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only= False))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only= False))

    fileName = PATH + MODEL_NAME + "/" + "profile-" + MODEL_NAME + "-unoptimized.txt"
    f = open(fileName, "w")
    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        model(torch.randn(1,3,64,64).to(device))

    f.write(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only=False))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only=False))


def main():
    # Settings to generate output files for the generated code
    # If this does not generate output code files run "export TORCH_COMPILE_DEBUG=1" in the terminal
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    torch._inductor.config.debug = True

    # Cuda capabilities >= 7.0 needed 
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # useCuda = torch.cuda.is_available()
    useCuda = False
    device = "cpu"
    
    # Load pretrained resnet50 model
    model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True)
    model.to(device)

    # Compile the model using inductor backend
    opt_model = torch.compile(model, backend="inductor")
    # Forces compilation bevore measuring
    opt_model(torch.randn(1,3,64,64).to(device)) 

    if (device == "cpu"):
        # Measure CPU time using timeit
        timing(opt_model=opt_model, model=model, device=device, reps=10, iters=100)
    else: 
        print("need to rewrite this")
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        # z = x + y
        # end.record()

        # Waits for everything to finish running
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))

    # Profile execution
    profiler(opt_model=opt_model, model=model, device=device, useCuda=useCuda)

    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    torch._inductor.config.debug = False


if __name__ == "__main__":
    main()