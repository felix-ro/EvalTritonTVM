import os
import urllib
import torch
import torch._dynamo
import torch._inductor.config
from timeit import Timer
from PIL import Image
from torchvision import transforms

NUM_REPS = 10
NUM_ITERS = 100
PATH = "Results/Triton/"
MODEL_NAME = "resnet50"


def time_CPU(opt_model, model, device, input_batch, reps, iters):
    file_name = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + \
        "-optimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    results = ""
    print("Optimized model: ")
    for i in range(reps):
        t = Timer(lambda: opt_model(input_batch.to(device)))
        print(t.timeit(number=iters)/iters)
        results += str(t.timeit(number=iters)/iters) + "\n"
    f.write(results)

    file_name = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + \
        "-unoptimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    results = ""
    print("Unoptimized model: ")
    for i in range(reps):
        t = Timer(lambda: model(input_batch.to(device)))
        print(t.timeit(number=iters)/iters)
        results += str(t.timeit(number=iters)/iters) + "\n"
    f.write(results)


def profiler(opt_model, model, device, input_batch, useCuda):
    file_name = PATH + MODEL_NAME + "/" + "profile-" + MODEL_NAME + \
        "-unoptimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        model(input_batch.to(device))

    f.write(prof.key_averages().table(sort_by="self_cpu_time_total",
                                      top_level_events_only=False))
    print(prof.key_averages().table(sort_by="self_cpu_time_total",
                                    top_level_events_only=False))

    file_name = PATH + MODEL_NAME + "/" + "profile-" + MODEL_NAME + \
        "-optimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        opt_model(input_batch.to(device))

    f.write(prof.key_averages().table(sort_by="self_cpu_time_total",
                                      top_level_events_only=False))
    print(prof.key_averages().table(sort_by="self_cpu_time_total",
                                    top_level_events_only=False))


def measure_cuda(model, device, input_batch, reps, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model_results = []
    for i in range(reps):
        sum = 0
        for j in range(iters):
            start.record()
            model(input_batch.to(device))
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            sum += start.elapsed_time(end)  # records in milliseconds

        sum = sum / iters
        print(str(sum) + " ms")
        model_results.append(sum)

    return model_results


def time_cuda(opt_model, model, device, input_batch, reps, iters):
    print("Unoptimized Model:")
    model_results = measure_cuda(
        model=model,
        device=device,
        input_batch=input_batch,
        reps=reps,
        iters=iters
    )

    print("\nOptimized Model")
    opt_model_results = measure_cuda(
        model=opt_model,
        device=device,
        input_batch=input_batch,
        reps=reps,
        iters=iters
    )

    file_name = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + \
        "-optimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    for res in opt_model_results:
        f.write(str(res) + " ms\n")
    f.close()

    file_name = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + \
        "-unoptimized-" + str(device) + ".txt"
    f = open(file_name, "w")
    for res in model_results:
        f.write(str(res) + " ms\n")
    f.close()


def get_aritficial_data(shape, dtype):
    return torch.rand(shape, dtype=dtype)


def get_test_image():
    url, file_name = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, file_name)
    except Exception:
        urllib.request.urlretrieve(url, file_name)

    input_image = Image.open(file_name)
    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


def main():
    # Settings to generate output files for the generated code
    # If this does not generate output code files run
    # "export TORCH_COMPILE_DEBUG=1" in the terminal
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    torch._inductor.config.debug = True

    # Cuda capabilities >= 7.0 needed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    # useCuda = False
    # device = "cpu"

    # Load pretrained resnet50 model
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           MODEL_NAME,
                           pretrained=True)
    model.to(device)

    # input_batch = get_test_image()
    input_batch = get_aritficial_data(shape=(1, 3, 224, 224), dtype=torch.float32)

    # Compile the model using inductor backend
    opt_model = torch.compile(model, backend="inductor", mode="max-autotune")
    opt_model.to(device)

    # Warm up and force opt_model compilation
    opt_model(input_batch.to(device))
    model(input_batch.to(device))

    if (device == "cpu"):
        # Measure CPU time using timeit
        time_CPU(opt_model=opt_model,
                 model=model,
                 device=device,
                 input_batch=input_batch,
                 reps=NUM_REPS,
                 iters=NUM_ITERS)
    else:
        time_cuda(opt_model=opt_model,
                  model=model,
                  device=device,
                  input_batch=input_batch,
                  reps=NUM_REPS,
                  iters=NUM_ITERS)

    # Profile execution
    profiler(opt_model=opt_model,
             model=model,
             device=device,
             input_batch=input_batch,
             useCuda=use_cuda)

    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    torch._inductor.config.debug = False


if __name__ == "__main__":
    main()
