# Weird results
- Experienced crashes with dot products larger than 3 in length 
- Matmuls of any kind triggered aten backend and not triton compilation 

[See this for more:](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747/2)
```TorchInductor generates nearly all of its kernels automatically from scratch based on its IR.

The two exceptions are matmul/conv where it has a template with auto-generated epilogue fusions. In the current numbers these are disabled by config, and we are just using aten. I’m expecting another 10-20% speedup from enabling this.

There is also a small list of kernels we haven’t implemented yet and are using aten fallbacks: TorchInductor missing ops tracker · Issue #93757 · pytorch/pytorch · GitHub 69
but eventually we wan’t to codegen everything.``````