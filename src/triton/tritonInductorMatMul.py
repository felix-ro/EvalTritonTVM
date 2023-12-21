import torch
import torch._dynamo
import torch._inductor.config
import os

PATH = "Results/Triton/"
MODEL_NAME = "matmul"
NUM_REPS = 10
NUM_ITERS = 100


def opt_matmul(a, b):
    return torch.matmul(a, b)


def matmul(a, b):
    return torch.matmul(a, b)


def time_CPU():
    print("Measuring CPU execution time...")


def measure_cuda(func, device, tensor1, tensor2, reps, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model_results = []
    for i in range(reps):
        sum = 0
        for j in range(iters):
            start.record()
            func(tensor1.to(device), tensor2.to(device))
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            sum += start.elapsed_time(end)

        sum = sum / iters
        print(sum)
        model_results.append(sum)

    return model_results


def profiler(func, opt_func, device, tensor1, tensor2, useCuda):
    file_name = PATH + MODEL_NAME + "/" + "profile-" + MODEL_NAME + \
        "-unoptimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        func(tensor1.to(device), tensor2.to(device)).to(device)

    f.write(prof.key_averages().table(sort_by="self_cpu_time_total",
                                      top_level_events_only=False))
    print(prof.key_averages().table(sort_by="self_cpu_time_total",
                                    top_level_events_only=False))

    file_name = PATH + MODEL_NAME + "/" + "profile-" + MODEL_NAME + \
        "-optimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        opt_func(tensor1.to(device), tensor2.to(device)).to(device)

    f.write(prof.key_averages().table(sort_by="self_cpu_time_total",
                                      top_level_events_only=False))
    print(prof.key_averages().table(sort_by="self_cpu_time_total",
                                    top_level_events_only=False))


def time_CUDA(func, opt_func, device, tensor1, tensor2, reps, iters):
    print("Measuring cuda execution time...\nUnoptimized:")
    func_results = measure_cuda(func=func,
                                device=device,
                                tensor1=tensor1,
                                tensor2=tensor2,
                                reps=reps,
                                iters=iters)

    file_name = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + \
        "-unoptimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    for res in func_results:
        f.write(str(res) + "\n")
    f.close()

    print("Optimized")
    opt_func_results = measure_cuda(func=opt_func,
                                    device=device,
                                    tensor1=tensor1,
                                    tensor2=tensor2,
                                    reps=reps,
                                    iters=iters)

    file_name = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + \
        "-optimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    for res in opt_func_results:
        f.write(str(res) + "\n")
    f.close()


def main():
    print("Benchmarking Triton MatMul...")

    global PATH
    is_colab = True
    if is_colab:
        PATH = "drive/MyDrive/" + PATH

    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    torch._inductor.config.debug = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    size = (3)
    tensor1 = torch.randn(size, dtype=torch.float32)
    tensor2 = torch.randn(size, dtype=torch.float32)

    opt_matmul = torch.compile(matmul, backend="inductor")
    # warm up
    matmul(tensor1.to(device), tensor2.to(device)).to(device)
    opt_matmul(tensor1.to(device), tensor2.to(device)).to(device)


    if device == "CPU":
        time_CPU()
    else:
        time_CUDA(func=matmul,
                  opt_func=opt_matmul,
                  device=device,
                  tensor1=tensor1,
                  tensor2=tensor2,
                  reps=NUM_REPS,
                  iters=NUM_ITERS)
        
    profiler(matmul, opt_matmul, device, tensor1, tensor2, useCuda=True)

    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    torch._inductor.config.debug = False

if __name__ == "__main__":
    main()
