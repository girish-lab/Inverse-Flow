#include <torch/extension.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// cuda kernel1
namespace {

// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_dy_kernel1(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> M,  
    int c,
    int d,
    const int relevant_threads,
    const int order_stride ) 
    {
  
    const auto b = threadIdx.x;
    const auto tid = blockIdx.x; 
    const auto order = blockIdx.y;  // Current Order
    const auto B = M.size(0);  
    const auto C = M.size(1);
    const auto H = M.size(2);
    const auto W = M.size(3); 

    // NOTE: For some reason kernel.size(0) doesnot work correctly here
    const auto K_H = kernel.size(2);
    const auto K_W = kernel.size(3);

    int h, w;
    // pixel position in the diagonal
    if (d <= H) {
        h = d - 1 - tid;
        w = tid;
    } 
    else {
        w = (d - H)  + tid;
        h = H - 1 - tid;
    }
    
    // // compute entry of the output in the diagonal d assigned to this thread
    // apply_conv_window()
    if (h == 0 && w == 0) {
      M[b][c + order * order_stride][h][w] = 1.0;
    }
    else  {
      for (int k_h = 0; k_h < K_H; k_h++) {
        if (h - k_h < 0) break;
        for (int k_w = 0; k_w < K_W; k_w++) {
            if (w - k_w < 0) break;
            for (int k_c = 0; k_c < order_stride; k_c++) {
                if (k_h == 0 && k_w == 0) {
                    if (k_c == c) continue;
                }
                // output[b][c + order * order_stride][h][w] += input[b][k_c + order * order_stride][h - k_h][w - k_w] \
                // * kernel[c + order * order_stride][k_c][K_H - k_h - 1][K_W - k_w - 1];
                // M[h][ip][jp] -=  (kernel[k-1-iq][k-1-jq] * M[h][ip-iq][jp-jq]);
                M[b][c + order * order_stride][h][w] -=  
                (kernel[c + order * order_stride][k_c][K_H-1-k_h][K_W-1-k_w] * M[b][c + order * order_stride][h-k_h][w-k_w]);
                // printf("k_h, k_w, k_c, M[][][][]: %d %d %d %g \n", k_h, k_w, (c + order * order_stride), M[b][c + order * order_stride][h][w]);
            }
          }
        } 
      }
  }      
} // end namespace

// cuda kernel-2
namespace {
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_dy_kernel2(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> M,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
    int c,
    int d,
    const int relevant_threads,
    const int order_stride ) 
    {
  
    const auto b = threadIdx.x;
    const auto tid = blockIdx.x; 
    const auto order = blockIdx.y;  // Current Order
    const auto B = M.size(0);  
    const auto C = M.size(1);
    const auto H = M.size(2);
    const auto W = M.size(3); 

    // NOTE: For some reason kernel.size(0) doesnot work correctly here
   

    int n = H;
    int h, w;
    
    if (d <= n) {
        h = d - 1 - tid;
        w = tid;
    } 
    else {
        w = (d - n)  + tid;
        h = n - 1 - tid;
    }
    // output[b][c + order * order_stride][h][w] = input[b][c + order * order_stride][k_h][k_w] 
    //             * M[b][c + order * order_stride][h-k_h][w-k_w];
    // // compute entry of the output in the diagonal d assigned to this thread
    for (int k_h = 0; k_h < H; k_h++) {
        if (h - k_h < 0) break;
        for (int k_w = 0; k_w < W; k_w++) {
            if (w - k_w < 0) break;
          
            output[b][c + order * order_stride][h][w] += input[b][c + order * order_stride][k_h][k_w] 
            * M[b][c + order * order_stride][h-k_h][w-k_w];

            // printf("M[%d][%d][%d][%d] %g \n", b, (c+ order * order_stride), h- k_h, w- k_w, M[b][c+ order * order_stride][h-k_h][w-k_w]);

            // }
          }
        } 
  }
} // end namespace

// // // ////////////////////////////////////////////////
std::vector<torch::Tensor> inv_conv_dy(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor M,
    torch::Tensor output) {

  // assuming batch size 1 for initial implementation
  const auto B = output.size(0);
  const auto C = output.size(1);
  const auto H = output.size(2);
  const auto W = output.size(3); // assuming height and width to be the same

  const auto n = H < W ? H : W; //samller dimension  -- min(H, W)
  const auto m = H > W ? H : W; //larger dimension   -- max(H, W)

  dim3 threads(1024, 1, 1);         // Fix the number of threads
  dim3 blocks(B, 4, 1);             // Fix the number of grids = Batch_Size

  const auto order_stride = (int) C / 4;

  // kernel-1
  for (int d = 1; d <= H + W -1; d++) { // Iterating over diagonal index
    for (int c = 0; c < (int) C / 4; c++) {

        // all elements of the dth diagonal computed in parallel
        int relevant_threads = d;         // Since we have fixed the threads, not all threads are going to be useful
        if (d > n) {
            if (d <= m) {
                relevant_threads = n;
            } else {  // equivalent to if (d > m) 
                relevant_threads = m + n - d;  // equivalent to 2 * n - d when n == m;
            }
        }
        // threads = relevant_threads*m;dim3 threads(B, 1, 1);         // Fix the number of threads

        dim3 threads(B, 1, 1);         // Fix the number of threads
        dim3 blocks(relevant_threads, 4, 1); 
        
        // const int blocks = (max_threads + relevant_threads)/max_threads; // use multiple blocks if 2d-1 > threads

        AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_dL_dx_cuda_inverse", ([&] {
          inv_conv_dy_kernel1<scalar_t><<<blocks, threads>>>(
              input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
              kernel.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
              M.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
              c,
              d,
              relevant_threads,
              order_stride
            );
        }));

      // synchronize all threads
      cudaDeviceSynchronize();
    }
  }
  
  // kernel-2
  for (int d = 1; d <= H + W -1; d++) { // Iterating over diagonal index
    for (int c = 0; c < (int) C / 4; c++) {

        // all elements of the dth diagonal computed in parallel
        int relevant_threads = d;  // Since we have fixed the threads, not all threads are going to be useful
        if (d > n) {
            if (d <= m) {
                relevant_threads = n;
            } else {  // equivalent to if (d > m) 
                relevant_threads = m + n - d;  // equivalent to 2 * n - d when n == m;
            }
        }
        // threads = relevant_threads*m;dim3 threads(B, 1, 1);         // Fix the number of threads

        dim3 threads(B, 1, 1);         // Fix the number of threads
        dim3 blocks(relevant_threads, 4, 1); 
        
        // const int blocks = (max_threads + relevant_threads)/max_threads; // use multiple blocks if 2d-1 > threads

        AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_dL_dx_cuda_inverse", ([&] {
          inv_conv_dy_kernel2<scalar_t><<<blocks, threads>>>(
              input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
              M.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
              output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
              c,
              d,
              relevant_threads,
              order_stride
            );
        }));

      // synchronize all threads
      cudaDeviceSynchronize();
    }
  }
  
  return {output};
}
// cinc = inv_conv_fwd