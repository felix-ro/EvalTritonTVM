import os
from timeit import Timer
import torch
import torch._dynamo
import torch._inductor.config
import urllib
from PIL import Image
from torchvision import transforms

NUM_REPS = 10
NUM_ITERS = 100
PATH = "Results/Triton/"
MODEL_NAME = "resnet50"

def timing(opt_model, model, device, input_batch, reps, iters):
    fileName = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + "-optimized-" + str(device) + ".txt"
    f = open(fileName, "w")
    results = ""
    print("Optimized model: ")
    for i in range(reps):
        t = Timer(lambda: opt_model(input_batch.to(device)))
        print(t.timeit(number=iters)/iters)
        results += str(t.timeit(number=iters)/iters) + "\n"
    f.write(results)

    fileName = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + "-unoptimized-" + str(device) + ".txt"
    f = open(fileName, "w")
    results = ""
    print("Unoptimized model: ")
    for i in range(reps):
        t = Timer(lambda: model(input_batch.to(device)))
        print(t.timeit(number=iters)/iters)
        results += str(t.timeit(number=iters)/iters) + "\n"
    f.write(results)


def profiler(opt_model, model, device, input_batch, useCuda):
    fileName = PATH + MODEL_NAME + "/" + "profile-" + MODEL_NAME + "-optimized-" + str(device) + ".txt"
    f = open(fileName, "w")
    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        opt_model(input_batch.to(device))

    f.write(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only= False))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only= False))

    fileName = PATH + MODEL_NAME + "/" + "profile-" + MODEL_NAME + "-unoptimized-" + str(device) + ".txt"
    f = open(fileName, "w")
    with torch.autograd.profiler.profile(use_cuda=useCuda) as prof:
        model(input_batch.to(device))

    f.write(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only=False))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", top_level_events_only=False))

def measureCuda(model, device, input_batch, reps, iters):
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
        sum += start.elapsed_time(end)
      
      sum = sum / iters
      print(sum)
      model_results.append(sum)
    
    return model_results

def timingCuda(opt_model, model, device, input_batch, reps, iters):
    model_results = measureCuda(model, device, input_batch, reps, iters)
    opt_model_results = measureCuda(opt_model, device, input_batch, reps, iters)

    fileName = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + "-optimized-" + str(device) + ".txt"
    f = open(fileName, "w")
    for res in opt_model_results: 
      f.write(str(res) + "\n")
    f.close()

    fileName = PATH + MODEL_NAME + "/" + "results-" + MODEL_NAME + "-unoptimized-" + str(device) + ".txt"
    f = open(fileName, "w")
    for res in model_results: 
      f.write(str(res) + "\n")
    f.close()


def main():
    global PATH
    is_colab = True
    if is_colab: 
      PATH = "drive/MyDrive/" + PATH

    # Settings to generate output files for the generated code
    # If this does not generate output code files run "export TORCH_COMPILE_DEBUG=1" in the terminal
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    torch._inductor.config.debug = True

    # Cuda capabilities >= 7.0 needed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    useCuda = torch.cuda.is_available()
    # useCuda = False
    # device = "cpu"

    # Load pretrained resnet50 model
    model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True)
    model.to(device)

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)


    input_image = Image.open(filename)
    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    

    # Compile the model using inductor backend
    opt_model = torch.compile(model, backend="inductor")
    # Forces compilation bevore measuring
    opt_model(input_batch.to(device))


    if (device == "cpu"):
        # Measure CPU time using timeit
        timing(opt_model=opt_model, model=model, device=device, input_batch=input_batch, reps=NUM_REPS, iters=NUM_ITERS)
    else:
        timingCuda(opt_model=opt_model, model=model, device=device, input_batch=input_batch, reps=NUM_REPS, iters=NUM_ITERS)


    # Profile execution
    profiler(opt_model=opt_model, model=model, device=device, input_batch=input_batch, useCuda=useCuda)

    #os.environ["TORCH_COMPILE_DEBUG"] = "0"
    #torch._inductor.config.debug = False


if __name__ == "__main__":
    main()