import torch

PATH = "Results/Triton/matmul/"
MODEL_NAME = "matmul"
NUM_REPS = 10
NUM_ITERS = 100


@torch._dynamo.optimize(backend="inductor")
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


def time_CUDA(device, tensor1, tensor2, reps, iters):
    print("Measuring cuda execution time...\nUnoptimized:")
    func_results = measure_cuda(func=matmul,
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

    opt_func_results = measure_cuda(func=opt_matmul,
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tensor1 = torch.randn(3)
    tensor2 = torch.randn(3)

    # warm up
    matmul(tensor1, tensor2)
    opt_matmul(tensor1, tensor2)

    if device == "CPU":
        time_CPU()
    else:
        time_CUDA(device=device,
                  tensor1=tensor1,
                  tensor2=tensor2,
                  reps=NUM_REPS,
                  iters=NUM_ITERS)


if __name__ == "__main__":
    main()
