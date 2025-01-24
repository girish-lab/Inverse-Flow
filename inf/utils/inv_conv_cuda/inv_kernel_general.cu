#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

 
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_cuda_inverse_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> kernel,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output, 
    const int c,
    const int d,
    const int relevant_threads,
    const int order_stride
    )
{
  
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    
    const auto b = threadIdx.x; // batch index
    const auto tid = blockIdx.x;  // thread id
    const auto order = blockIdx.y;  // Current Order, channel index

    const auto B = output.size(0);  
    const auto C = output.size(1);
    const auto H = output.size(2);
    const auto W = output.size(3); 

    // NOTE: For some reason kernel.size(0) doesnot work correctly here
    const auto K_H = kernel.size(2);
    const auto K_W = kernel.size(3);
    // printf("kernel size %d %d %d %d\n", kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3));
    // Compute the index from the dth diagonal assigned to the thread
    int h, w;
    if (d <= H) {
        h = d - 1 - tid;
        w = tid;
    } 
    else {
        w = (d - H)  + tid;
        h = H - 1 - tid;
    }
    // printf(" cuda_cinc B, C, H, W, h, w, k_h, k_w : %d  %d  %d  %d  %d  %d  %d  %d \n", b, c, H, W, h, w, K_H, K_W);
    // printf("order %d \n", order);

    // compute entry of the output in the diagonal d assigned to this thread
    output[b][c + order * order_stride][h][w] = input[b][c + order * order_stride][h][w];
    for (int k_h = 0; k_h < K_H; k_h++) {
        if (h - k_h < 0) break;
        for (int k_w = 0; k_w < K_W; k_w++) {
            if (w - k_w < 0) break;
            for (int k_c = 0; k_c < order_stride; k_c++) {
                if (k_h == 0 && k_w == 0) {
                    if (k_c == c) continue;
                }
                output[b][c + order * order_stride][h][w] -= output[b][c + order * order_stride][h - k_h][w - k_w] \
                * kernel[c + order * order_stride][k_c][K_H - k_h - 1][K_W - k_w - 1]; 
            }
        }
    }


}

} // namespace

std::vector<torch::Tensor> inv_conv_cuda_inverse(
    torch::Tensor input,   //B, C, H, W
    torch::Tensor kernel,  // 4, C, C, K_H, K_W
    torch::Tensor output)  // B, C, H, W
{

  // assuming batch size 1 for initial implementation
  // all the tensors above are 2D only

  const auto B = output.size(0);
  const auto C = output.size(1);
  const auto H = output.size(2);
  const auto W = output.size(3);

  const auto K_H = kernel.size(2);
  const auto K_W = kernel.size(3);

  // printf("%d %d", n,k);
  const auto n = H < W ? H : W; //samller dimension  -- min(H, W)
  const auto m = H > W ? H : W; //larger dimension   -- max(H, W)

  dim3 threads(1024, 1, 1);         // Fix the number of threads
  dim3 blocks(B, 4, 1);             // Fix the number of grids = Batch_Size

  const auto order_stride = (int) C / 4;
  for (int d = 1; d <= H + W - 1; d++) { // Iterating over diagonal index
    for (int c = 0; c < (int) C / 4 ; c++) {
      // const int threads = 1024;
      
      // all elements of the dth diagonal computed in parallel
      int relevant_threads = d;         // Since we have fixed the threads, not all threads are going to be useful
      if (d > n) {
          if (d <= m) {
              relevant_threads = n;
          } else {  // equivalent to if (d > m) 
              relevant_threads = m + n - d;  // equivalent to 2 * n - d when n == m;
          }
      }
      dim3 threads(B, 1, 1);         // Fix the number of threads
      dim3 blocks(relevant_threads, 4, 1); 
      AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_inverse_cuda", ([&] {
          inv_conv_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
              input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>(),
              kernel.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>(),
              output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>(),
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
// cinc = inv_conv


// Standard convolution forward pass


// Compute the index from the dth diagonal assigned to the thread
    // int h, i, j;
    // h = tid % m; // batch index encoded in lower indices modulo batchsize
    // tid = (tid - h) / m; // remaining part of tid has the index along the diagonal
    // if (d <= n) {
    //     i = d - 1 - tid;
    //     j = tid;
    // } 
    // else {
    //     j = (d - n)  + tid;
    //     i = n - 1 - tid;
    // }

    // compute entry of the output in the diagonal d assigned to this thread
    // output[h][i][j] = input[h][i][j];
    
    // for (int qi = 0; qi < k; qi++) {
    //     for (int qj = 0; qj < k; qj++) {
    //         if ( i-(k-1)+ qi >=0 && j-(k-1)+ qj >=0 && !((qi == k-1) && (qj==k-1))) {
    //         // if (i - qi >= 0 && j - qj >= 0  && i - qi < n && j -qj < n) { //
    //             // printf("inverse%g %d, %d\n", kernel[qi][qj], qi, qj);
    //             // printf("%d %d %d %d %d %d %d %d\n", h, i, j, qi, qj, i-(k-1)+qi, j-(k-1)+qj, output[h][i-(k-1)+qi][j-(k-1)+qj]);
    //             output[h][i][j] -= kernel[qi][qj]*output[h][i-(k-1)+qi][j-(k-1)+qj];
    //         }
    //         // if (i-(k-1)+ qi < 0 || j-(k-1)+ qj < 0) {
    //         //     // output[h][i][j] -= kernel[qi][qj]*input[h][i-(k-1)+qi][j-(k-1)+qj];
    //         //     printf("inverse%g %d, %d\n", kernel[qi][qj], qi, qj);
    //         // }
    //     }
    // }

//   printf("m, n, k %d %d %d", m , n,k);

  // for (int d = 1; d <= 2 * n - 1; d++) { // Iterating over diagonal index

  //   int max_threads = 1024;

  //   // all elements of the dth diagonal computed in parallel
  //   int threads = d;
  //   if (d > n) {
  //     threads = 2 * n - d;
  //   }
  //   threads = threads * m;
  //   // printf("m, n, k d, %d %d %d %d", m , n, k, d);
    
  //   const int blocks = (max_threads + threads) / max_threads; // use multiple blocks if 2d-1 > threads

  //   AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_inverse_cuda", ([&] {
  //       inv_conv_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
  //           input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
  //           kernel.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
  //           output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(), k, d);
  //   }));

  //   // synchronize all threads
  //   cudaDeviceSynchronize();

  // }