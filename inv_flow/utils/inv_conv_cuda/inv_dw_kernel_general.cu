#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// cuda kernel1 for inv_conv_dw
namespace { 
// Implementation for batch size = 1, channels = 1, assuming height and width to be the same for input and kenrel 
template <typename scalar_t>
__global__ void inv_conv_dw_kernel1(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t,6,torch::RestrictPtrTraits,size_t> M,
    int c,
    int d,
    int K,
    const int relevant_threads,
    const int order_stride )  
    {
      

    const auto b = threadIdx.x;
    const auto tid = blockIdx.x; 
    const auto order = blockIdx.y;  // Current Order
    const auto B = input.size(0);  
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3); 

    // NOTE: For some reason kernel.size(0) doesnot work correctly here
    // const auto K_C_out = kernel.size(0);
    // const auto K_C_in = kernel.size(1);
    // const auto K_H = kernel.size(2);
    // const auto K_W = kernel.size(3);

    // int h, w;
    // int n =  H < W ? H : W; //samller dimension  -- min(H, W)

    int k_h, k_w;
    // pixel position in the diagonal
    if (d <= K) {
        k_h = d - 1 - tid;
        k_w = tid;
    } 
    else {
        k_w = (d - K)  + tid;
        k_h = K - 1 - tid;
    }

    if (!((k_h == K-1 && k_w == K-1))){
      for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
          if (i-k_h>=0 && j-k_w>=0) {// apply_conv_window
            // printf("M[a,b,i,j] M[a,b,i,j] M[a,b,i,j] != 0 %d %d \n", a, b);
            float temp = 0;
            for (int k_c = 0; k_c < order_stride; k_c++) {
              if (i == 0 && j == 0) {
                // M[b][c + order * order_stride][a][b][i][j] = 0.0;
                if (k_c == c) continue;
              }
              for (int iq = 0; iq < K; iq++) {
                for (int jq = 0; jq < K; jq++) {
                  // printf("i, j, iq, jq %d %d %d %d \n", i, j, iq, jq);
                  if (i-iq >= 0 && j-jq >= 0) {
                      temp += M[b][c + order * order_stride][k_h][k_w][i-iq][j-jq] 
                               * kernel[c + order * order_stride][k_c][K-1-iq][K-1-jq]; //
                  }
                }
              }
            }
            M[b][c + order * order_stride][k_h][k_w][i][j] = 
             -(temp + input[b][c + order * order_stride][i-k_h][j-k_w]);
            
          }
        }
      }
    }
    
}
} // end namespace-1

      // cuda kernel-2 //
namespace { 
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_dw_kernel2(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> loss,
    const torch::PackedTensorAccessor<scalar_t,6,torch::RestrictPtrTraits,size_t> M,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output, 
    int K,
    int c,
    int d,
    const int relevant_threads,
    const int order_stride )
    {
  
    const auto b = threadIdx.x;
    const auto tid = blockIdx.x; 
    const auto order = blockIdx.y;  // Current Order
    const auto B = loss.size(0);  
    const auto C = loss.size(1);
    const auto H = loss.size(2);
    const auto W = loss.size(3); 

    // NOTE: For some reason kernel.size(0) doesnot work correctly here
    // const auto K_H = kernel.size(2);
    // const auto K_W = kernel.size(3);

    // int h, w;
    // int n =  H < W ? H : W; //samller dimension  -- min(H, W)

    int k_h, k_w;
    // pixel position in the diagonal
    if (d <= K) {
        k_h = d - 1 - tid;
        k_w = tid;
    } 
    else {
        k_w = (d - K)  + tid;
        k_h = K - 1 - tid;
    }
    // for (int i = 0)


    // output[a][b] = (loss * M[a][b]).sum();
    // compute entry of the output in the diagonal d assigned to this thread
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
          for (int k_c = 0; k_c < order_stride; k_c++) {
            if (i == 0 && j == 0) {
            //   // M[b][c + order * order_stride][a][b][i][j] = 0.0;
              if (k_c == c) continue;
            }
            output[b][c + order * order_stride][k_h][k_w] += 
                   loss[b][c + order * order_stride][i][j] * M[b][c + order * order_stride][k_h][k_w][i][j];
          }
          // output[a][b] += loss[h][i][j] * M[a][b][i][j];
          // printf(" a, b, i, j, output[h][a][b] %d %d %d %d %g \n", a, b, i, j,\
           output[h][a][b], loss[h][i][j]);
            
        }
    }
  }

} // namespace-2

/////////////////////////////////////
std::vector<torch::Tensor> inv_conv_dw(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor loss,
    torch::Tensor M,
    torch::Tensor output) {

  const auto B = output.size(0);
  const auto C = output.size(1);
  const auto H = output.size(2);
  const auto W = output.size(3); // assuming height and width to be the same

  const auto c_out = kernel.size(0);
  const auto c_in = kernel.size(1);
  const auto k_h = kernel.size(2); // assuming square kernel
  const auto k_w = kernel.size(3); // assuming square kernel

  const auto n = H < W ? H : W; //samller dimension  -- min(H, W)
  const auto m = H > W ? H : W; //larger dimension   -- max(H, W)

  dim3 threads(1024, 1, 1);         // Fix the number of threads
  dim3 blocks(B, 4, 1);             // Fix the number of grids = Batch_Size

  const auto order_stride = (int) C / 4;

  // kernel-1
  for (int d = 1; d <= k_h + k_w - 1; d++) { // Iterating over diagonal index
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
          inv_conv_dw_kernel1<scalar_t><<<blocks, threads>>>(
              input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
              kernel.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
              M.packed_accessor<scalar_t,6,torch::RestrictPtrTraits,size_t>(), 
              c,
              d,
              k_h,
              relevant_threads,
              order_stride
            );
        }));

      // synchronize all threads
      cudaDeviceSynchronize();
    }
  }

  
  // kernel-2
  for (int d = 1; d <= 2*k_h-1; d++) { // Iterating over diagonal index
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
      
      // const int blocks = (max_threads + threads)/max_threads; // use multiple blocks if 2d-1 > threads

      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "inv_conv_dL_dw_cuda_inverse", ([&] {
        inv_conv_dw_kernel2<scalar_t><<<blocks, threads>>>(
            loss.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            M.packed_accessor<scalar_t,6,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
            k_w,
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
  
  return {output, M};
}
// cinc = inv_conv_fwd

// trash code


// const auto c = input.size(0); // number of channels
  // const auto k_h = kernel.size(1); // assuming square kernel
  // const auto k_w = kernel.size(0); // assuming square kernel
  // // const auto k = k_h; // assuming square kernel
  // printf("initial c, k_h, k_w %d %d %d  \n", c, k_h, k_w); // 3, 2
  // // printf("initial---- m, n, k %d %d %d \n", m, n, k); // 1, 3, 2
// for (int d = 1; d <= 2*k_h-1; d++) { // Iterating over diagonal index
  //   int max_threads = 1024;
  //   // all elements of the dth diagonal computed in parallel
  //   int threads = d;
  //   if (d > k_h) {
  //     threads = 2*k_h-d;
  //   }
  //   // threads = threads; // * k_c if channels > 1;
  //   const int blocks = (max_threads + threads) / max_threads; //use multiple blocks if 2d-1 > threads

  //   AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_dL_dw_cuda_inverse", ([&] {
  //     inv_conv_dw_kernel1<scalar_t><<<blocks, threads>>>(
  //         input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
  //         kernel.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
  //         M.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), d);
  //   }));
  //   // synchronize all threads
  //   cudaDeviceSynchronize();
  // }
  // // float  M = torch::zeros({k, k, m, n}, torch::CUDA(torch::kFloat));
// int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    // // const auto m  = input.size(0); // assuming height and width to be the same
    // const auto k_h = M.size(0); // number of channels
    // // const auto n = input.size(1); // assuming height and width to be the same
    // const auto k_w = M.size(2); // assuming height and width to be the same
    // const auto n  = M.size(2); // assuming square kernel
    // const auto m  = M.size(3); // assuming square kernel
    // const auto c_in  = input.size(0);
    // printf("kernel1, ----- %d %d %d %d %d \n", c_in, n, m, k_h, k_w, d); // 2, 0, 2, 0, 2
    // const auto n  = input.size(1);
    // printf("n %d \n", n);
    // const auto k  = kernel.size(0);
    // NOTE: For some reason kernel.size(0) doesnot work correctly here

    // Compute the index from the dth diagonal assigned to the thread
    // int  a, b; // h,
    // // h = tid % k; // batch index encoded in lower indices modulo batchsize
    // tid =  tid / k_h; //(tid - h) / k; // remaining part of tid has the index along the diagonal
    // if (d <= k_h) {
    //     a = d - 1 - tid;
    //     b = tid;
    // }
    // else {
    //     a = k_h - 1 - tid;
    //     b = (d - k_h)  + tid;
    // }
    // // compute entry of the output in the diagonal d assigned to this thread
    // if (a == k_h-1 && b == k_h-1) {
    //     // printf("M[a,b,i,j] = 0 \n");
    //     for (int i = 0; i < m; i++) {
    //       for (int j = 0; j < m; j++) {
    //         M[a][b][i][j] = 0.0;
    //       }
    //     }
    // }
    // else {
    //   for (int i = 0; i < m; i++) {
    //       for (int j = 0; j < m; j++) {
    //       float temp = 0;
    //       //  apply_conv_window
    //         for (int iq = 0; iq < k_h; iq++) {
    //             for (int jq = 0; jq < k_h; jq++) {
    //               if (i-iq >= 0 && j-jq >= 0) {
    //                 temp += M[a][b][i-iq][j-jq] * kernel[k_h-1-iq][k_h-1-jq]; //
    //               }
    //             }
    //         }
    //       printf("temp %g \n", temp);
    //       M[a][b][i][j] -=  temp + input[c_in][i-a][j-b];
    //       }
    //   } 
    // }