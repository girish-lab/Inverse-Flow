#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_cuda_inverse_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> kernel,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output, int k, int d) {
  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    const auto m  = output.size(0); // assuming height and width to be the same
    const auto n  = output.size(1);
    // NOTE: For some reason kernel.size(0) doesnot work correctly here

    // Compute the index from the dth diagonal assigned to the thread
    int h, i, j;
    h = tid % m; // batch index encoded in lower indices modulo batchsize
    tid = (tid - h) / m; // remaining part of tid has the index along the diagonal
    if (d <= n) {
        i = d - 1 - tid;
        j = tid;
    } 
    else {
        j = (d - n)  + tid;
        i = n - 1 - tid;
    }
    // compute entry of the output in the diagonal d assigned to this thread
    output[h][i][j] = input[h][i][j];
    
    for (int qi = 0; qi < k; qi++) {
        for (int qj = 0; qj < k; qj++) {
            if ( i-(k-1)+ qi >=0 && j-(k-1)+ qj >=0 && !((qi == k-1) && (qj==k-1))) {
            // if (i - qi >= 0 && j - qj >= 0  && i - qi < n && j -qj < n) { //
                // printf("inverse%g %d, %d\n", kernel[qi][qj], qi, qj);
                // printf("%d %d %d %d %d %d %d %d\n", h, i, j, qi, qj, i-(k-1)+qi, j-(k-1)+qj, output[h][i-(k-1)+qi][j-(k-1)+qj]);
                output[h][i][j] -= kernel[qi][qj]*output[h][i-(k-1)+qi][j-(k-1)+qj];
            }
            // if (i-(k-1)+ qi < 0 || j-(k-1)+ qj < 0) {
            //     // output[h][i][j] -= kernel[qi][qj]*input[h][i-(k-1)+qi][j-(k-1)+qj];
            //     printf("inverse%g %d, %d\n", kernel[qi][qj], qi, qj);
            // }
        }
    }
}
} // namespace

std::vector<torch::Tensor> inv_conv_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output) {

  // assuming batch size 1 for initial implementation
  // all the tensors above are 2D only

  const auto m = output.size(0);
  const auto n = output.size(1); // assuming height and width to be the same
  const auto k = kernel.size(0); // assuming square kernel

//   printf("m, n, k %d %d %d", m , n,k);

  for (int d = 1; d <= 2 * n - 1; d++) { // Iterating over diagonal index

    int max_threads = 1024;

    // all elements of the dth diagonal computed in parallel
    int threads = d;
    if (d > n) {
      threads = 2 * n - d;
    }
    threads = threads * m;
    // printf("m, n, k d, %d %d %d %d", m , n, k, d);
    
    const int blocks = (max_threads + threads) / max_threads; // use multiple blocks if 2d-1 > threads

    AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_inverse_cuda", ([&] {
        inv_conv_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            kernel.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(), k, d);
    }));

    // synchronize all threads
    cudaDeviceSynchronize();

  }

  return {output};
} // cinc = inv_conv


// dL_dW, 2D //
namespace {
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_dL_dw_cuda_inverse_kernel1(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> kernel,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> M, \
    int n, int k, int d) {
    // const auto n = input.size(1); // assuming height and width to be the same
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    int  a, b; // h,
    if (d <= k) {
        a = d - 1 - tid;
        b = tid;
    } 
    else {
        a = k - 1 - tid;
        b = (d - k) + tid;
    }
    // printf(" n, k, d, a, b  %d %d %d %d %d \n", n, k, d, a, b);
    if (!((a == k-1 && b == k-1))){
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i-a>=0 && j-b>=0) {// apply_conv_window
                // printf("M[a,b,i,j] M[a,b,i,j] M[a,b,i,j] != 0 %d %d \n", a, b);
                float temp = 0;
                for (int iq = 0; iq < k; iq++) {
                    for (int jq = 0; jq < k; jq++) {
                    // printf("i, j, iq, jq %d %d %d %d \n", i, j, iq, jq);
                    if (i-iq >= 0 && j-jq >= 0) {
                        temp += M[a][b][i-iq][j-jq] * kernel[k-1-iq][k-1-jq]; //
                    }
                    }
                }
                M[a][b][i][j] =  -( temp + input[i-a][j-b]);
                printf(" temp, input[i-a][j-b], M %d %d %g %g %g\n", i-a, j-b, temp, input[i-a][j-b], M[a][b][i][j]);
                // printf("input[c_in][i-a][j-b] %g \n", input[c_in][i-a][j-b] ); 
            }
        }
      }
    }
}
} // namespace- end

      // cuda kernel-2 //
namespace { 
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_dL_dw_cuda_inverse_kernel2(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> loss,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> M,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output, \
    int n, int k, int d) {
  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    // const auto n = loss.size(1); // assuming height and width to be the same
    const auto h = loss.size(0); // number of channels
    // NOTE: For some reason kernel.size(0), M.size(0) doesnot work correctly here
    // Compute the index from the dth diagonal assigned to the thread
    int  a, b; // h,
    // h = tid % k; // batch index encoded in lower indices modulo batchsize
    // tid = (tid - h) / k; // remaining part of tid has the index along the diagonal
    // tid = tid/k;
    // pixel index 
    if (d <= k) {
        a = d - 1 - tid;
        b = tid;
    } 
    else {
        a = k - 1 - tid;
        b = (d - k)  + tid;
    }
    // printf("kernel2, n, k, d, a, b  %d %d %d %d %d \n", n, k, d, a, b); // 3, 2, range 1 - 2*k-1
    // printf("a, b %d %d \n", a, b);
    // output[a][b] = (loss * M[a][b]).sum();
    // compute entry of the output in the diagonal d assigned to this thread
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          output[a][b] += loss[i][j] * M[a][b][i][j];
          // printf(" a, b, i, j, output[h][a][b] %d %d %d %d %g \n", a, b, i, j,\
           output[h][a][b], loss[h][i][j]);
            
        }
    }
  }

} // namespace-2 end

std::vector<torch::Tensor> inv_conv_dL_dw_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor loss,
    torch::Tensor M,
    torch::Tensor output) {
  // const auto c_in = input.size(0);
  const auto n = input.size(1); // assuming height and width to be the same
  const auto k = kernel.size(1); // assuming square kernel
//   printf(" n, k,  %d %d \n", n, k);
  for (int d = 1; d <= 2*k-1; d++) { // Iterating over diagonal index
    int max_threads = 1024;
    int threads = d;
    if (d > k) {
      threads = 2 * k - d;
    }
    // threads = threads*k_c; // * k_c if channels > 1
    const int blocks = (max_threads + threads)/max_threads; // use multiple blocks if 2d-1 > threads
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "inv_conv_dL_dw_cuda_inverse", ([&] {
      inv_conv_dL_dw_cuda_inverse_kernel1<scalar_t><<<blocks, threads>>>(
          input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          kernel.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          M.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
          n, k, d);
    }));
    // synchronize all threads
    cudaDeviceSynchronize();
  }
  for (int d = 1; d <= 2*k-1; d++) { // Iterating over diagonal index

    int max_threads = 1024;
    int threads = d;
    if (d > k) {
      threads = 2 * k - d;
    }
    // threads = threads*k_c; // * k_c if channels > 1
    const int blocks = (max_threads + threads)/max_threads; // use multiple blocks if 2d-1 > threads

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "inv_conv_dL_dw_cuda_inverse", ([&] {
      inv_conv_dL_dw_cuda_inverse_kernel2<scalar_t><<<blocks, threads>>>(
          loss.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          M.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
          output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(), 
          n, k, d);
    }));
    // synchronize all threads
    cudaDeviceSynchronize();
  }
  return {output};
}



// dL_dx, 3D
// cuda kernel1
namespace {
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_dL_dx_cuda_inverse_kernel1(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> M,  int d, int count) {
  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    const auto m  = M.size(0); // assuming height and width to be the same
    const auto n  = M.size(1);
    const auto k  = kernel.size(0);
    // NOTE: For some reason kernel.size(0) doesnot work correctly here

    // Compute the index from the dth diagonal assigned to the thread
    int h, ip, jp;
    h = tid % m; // batch index encoded in lower indices modulo batchsize
    tid = (tid - h) / m; // remaining part of tid has the index along the diagonal

    if (d <= n) {
        ip = d - 1 - tid;
        jp = tid;
    } 
    else {
        ip = n - 1 - tid;
        jp = (d - n)  + tid;
    }

    // compute entry of the output in the diagonal d assigned to this thread
    if (ip == 0 && jp == 0) {
      M[h][ip][jp] = 1.0;
      printf(" d, ip, jp, h, M[ip][jp] %d %d %d %d %f \n", d, ip, jp, h , M[h][ip][jp]);
      // count += 1;             
      // printf("count %d \n", count);
    }
    else  {
      // int temp_const = 0;
      for (int iq = 0; iq < k; iq++) {
          for (int jq = 0; jq < k; jq++) {
            if ( ip - iq >= 0 && jp - jq >= 0) {
            // # if __CUDA_ARCH__ >= 200
            //   printf("%d %d %d %d %d %d \n",i,j,a,b,n,k);
            // #endif
              M[h][ip][jp] -=  (kernel[k-1-iq][k-1-jq] * M[h][ip-iq][jp-jq]);
              // printf(" d, ip, jp, h, M[ip][jp] %d %d %d %d %f \n", d, ip, jp, h , M[h][ip][jp]);
              // count += 1;
              // printf("count %d \n", count);
            }
          }
      }
      // M[h][ip][jp] = -temp_const;
      // printf(" d, ip, jp, h, M[ip][jp] %d%d %d %d %f \n", d, ip, jp, h , M[h][ip][jp]);
    }
  }      
} // end namespace

// cuda kernel-2
namespace {
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_dL_dx_cuda_inverse_kernel2(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> M,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output, int d) {
  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    const auto m  = output.size(0); // challens
    const auto n  = output.size(1); // assuming height and width to be the same
    // NOTE: For some reason kernel.size(0) doesnot work correctly here

    // Compute the index from the dth diagonal assigned to the thread
    int h, ip, jp;
    h = tid % m; // batch index encoded in lower indices modulo batchsize
    tid = (tid - h) / m; // remaining part of tid has the index along the diagonal

    if (d <= n) {
        ip = d - 1 - tid;
        jp = tid;
    } 
    else {
        ip = n - 1 - tid;
        jp = (d - n)  + tid;
    }
    // compute entry of the output in the diagonal d assigned to this thread
    for (int iq = 0; iq < n; iq++) {
        for (int jq = 0; jq < n; jq++) {
          if ( iq <= ip && jq <= jp ) {
              output[h][ip][jp] += input[h][iq][jq] * M[h][ip-iq][jp-jq];
              // count += 1;
              // printf("count %d \n", count);
          }
        }
    }
    // printf("d, i, j, output[i][j]  %d %d %d %f \n", d, ip, jp , output[h][ip][jp]);
  }
} // end namespace

// // // ////////////////////////////////////////////////
std::vector<torch::Tensor> inv_conv_dL_dx_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor M,
    torch::Tensor output) {

  // assuming batch size 1 for initial implementation
  // all the tensors above are 2D only
  const auto m = output.size(0);
  const auto n = output.size(1); // assuming height and width to be the same
  const auto k = kernel.size(0); // assuming square kernel

  // printf("m, n, k %d %d %d \n", m, n, k);
  int count = 0;
  for (int d = 1; d <= 2*n-1; d++) { // Iterating over diagonal index

    int max_threads = 1024;

    // all elements of the dth diagonal computed in parallel
    int threads = d;

    if (d > n) {
      threads = 2*n-d;
    }
    threads = threads*m;
    // printf("m, n, k, d: %d %d %d %d \n", m , n, k, d);
    
    const int blocks = (max_threads + threads) / max_threads; // use multiple blocks if 2d-1 > threads

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "inv_conv_dL_dx_cuda_inverse", ([&] {
      inv_conv_dL_dx_cuda_inverse_kernel1<scalar_t><<<blocks, threads>>>(
          input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          kernel.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          M.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), d, count);
    }));
    // synchronize all threads
    cudaDeviceSynchronize();
  }
  
  for (int d = 1; d <= 2*n-1; d++) { // Iterating over diagonal index
    // compute the output
    int max_threads = 1024;
    int threads = d;
    if (d > n) {
      threads = 2*n-d;
    }
    threads = threads*m;
    // printf("m, n, k, d: %d %d %d %d \n", m , n, k, d);

    const int blocks = (max_threads +threads)/max_threads; // use multiple blocks if 2d-1 > threads
    AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_dL_dx_cuda_inverse", ([&] {
      inv_conv_dL_dx_cuda_inverse_kernel2<scalar_t><<<blocks, threads>>>(
          input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          M.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), d);
    }));
    // synchronize all threads
    cudaDeviceSynchronize();
  }
  return {output};
}
// cinc = inv_conv_dL_dx

// Standard convolution forward pass
namespace {
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_fwd_cuda_inverse_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output, int k, int d) {
  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    const auto m  = output.size(0); // assuming height and width to be the same
    const auto n  = output.size(1);
    // NOTE: For some reason kernel.size(0) doesnot work correctly here

    // Compute the index from the dth diagonal assigned to the thread
    int h, i, j;
    h = tid % m; // batch index encoded in lower indices modulo batchsize
    tid = (tid - h) / m; // remaining part of tid has the index along the diagonal
    
    if (d <= n) {
        i = d - 1 - tid;
        j = tid;
    }
    else {
        j = (d - n)  + tid;
        i = n - 1 - tid;
    }
    // compute entry of the output in the diagonal d assigned to this thread
    for (int qi = 0; qi < k; qi++) {
        for (int qj = 0; qj < k; qj++) {
          if (i-(k-1)+ qi >=0 && j-(k-1)+ qj >=0) {
            output[h][i][j] += kernel[qi][qj]*input[h][i-(k-1)+qi][j-(k-1)+qj];
            // printf("output, input, kernel  results %g %g %g %g \n", output[h][i][j], input[h][i-(k-1)+qi][j-(k-1)+qj], kernel[qi][qj]);
          }
        }
    }
    // output[h][i][j] = result;
            
    }
} // namespace

std::vector<torch::Tensor> inv_conv_fwd_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output) {

  // assuming batch size 1 for initial implementation
  // all the tensors above are 2D only

  const auto m = output.size(0);
  const auto n = output.size(1); // assuming height and width to be the same
  const auto k = kernel.size(0); // assuming square kernel

  // printf("%d %d", n,k);
  // printf("input %g \n", input); 6.95297e-310
  // printf("output %g \n", output); 6.95297e-310
  // printf("kernel %g \n", kernel); 6.95297e-310

  for (int d = 1; d <= 2*n-1; d++) { // Iterating over diagonal index

    int max_threads = 1024;

    // all elements of the dth diagonal computed in parallel
    int threads = d;
    if (d > n) {
      threads = 2*n-d;
    }
    threads = threads*m;
    
    const int blocks = (max_threads +threads)/max_threads; // use multiple blocks if 2d-1 > threads

    AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_fwd_cuda_inverse", ([&] {
      inv_conv_fwd_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
          input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          kernel.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), k, d);
    }));

    // synchronize all threads
    cudaDeviceSynchronize();
  }

  return {output};
}
// cinc = inv_conv_fwd