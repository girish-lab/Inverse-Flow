# Cuda implementation for inverse convolution

For now only the boilerplate code is added. Inverse has to be implemented in the inv_conv_cuda_kernel.cu file (inside function inv_conv_cuda_inverse_kernel).

notes: https://pytorch.org/docs/master/notes/extending.html

https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix 

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

## Setup
1. download the ninja build binary (https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip) and put it under misc/bin (under home folder) 
2. install pytorch lts (1.8.2), with cuda 10.2
3. source the env.sh file
4. in the inv_conv_cuda directory, run python test_cuda_kernel.py
5. from inv_conv_cuda import inverse and forward and dL/dx (x=input) , dL/dw (w =kernel)

## References
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- https://developer.nvidia.com/blog/even-easier-introduction-cuda/
- https://github.com/pytorch/extension-cpp


Fortunately for us, ATen provides accessors that are created with a single dynamic check that a Tensor is the type and number of dimensions. Accessors then expose an API for accessing the Tensor elements efficiently without having to convert to a single pointer:

        torch::Tensor foo = torch::rand({12, 12});

        // assert foo is 2-dimensional and holds floats.
        auto foo_a = foo.accessor<float,2>();
        float trace = 0;

        for(int i = 0; i < foo_a.size(0); i++) {
        // use the accessor foo_a to get tensor data.
        trace += foo_a[i][i];
        }

# C++/CUDA Extensions in PyTorch

An example of writing a C++ extension for PyTorch. See
[here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.

There are a few "sights" you can metaphorically visit in this repository:

- Inspect the C++ and CUDA extensions in the `cpp/` and `cuda/` folders,
- Build C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and executing `python setup.py install`,
- JIT-compile C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and calling `python jit.py`, which will JIT-compile the extension and load it,
- Benchmark Python vs. C++ vs. CUDA by running `python benchmark.py {py, cpp, cuda} [--cuda]`,
- Run gradient checks on the code by running `python grad_check.py {py, cpp, cuda} [--cuda]`.
- Run output checks on the code by running `python check.py {forward, backward} [--cuda]`.

## Authors

[Peter Goldsborough](https://github.com/goldsborough), Soumith Chintala, and [Sandeep Nagar](github.com/naagar)