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
} // cinc = inv_conv







// kernel for fwd pass
namespace {
// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void inv_conv_fwd_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
    int c,
    int d,
    const int relevant_threads,
    const int order_stride
    ) {
  
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    // const auto m  = output.size(0); // assuming height and width to be the same
    // const auto n  = output.size(1);
    const auto b = threadIdx.x;
    const auto tid = blockIdx.x; 
    const auto order = blockIdx.y;  // Current Order
    const auto B = output.size(0);  
    const auto C = output.size(1);
    const auto H = output.size(2);
    const auto W = output.size(3); 

    // NOTE: For some reason kernel.size(0) doesnot work correctly here
    const auto K_H = kernel.size(2);
    const auto K_W = kernel.size(3);
    int n = H;

    int h, w;
    // h = tid % m; // batch index encoded in lower indices modulo batchsize
    // tid = (tid - h) / m; // remaining part of tid has the index along the diagonal
    if (d <= n) {
        h = d - 1 - tid;
        w = tid;
    } 
    else {
        w = (d - n)  + tid;
        h = n - 1 - tid;
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
                output[b][c + order * order_stride][h][w] += input[b][k_c + order * order_stride][h - k_h][w - k_w] \
                * kernel[c + order * order_stride][k_c][K_H - k_h - 1][K_W - k_w - 1];
                // output[h][i][j] += kernel[k_h][k_w]*input[h][i-(k-1)+qi][j-(k-1)+qj]; 
                // printf("output, input, kernel  results %g %g %g %g \n", output[b][c + order * order_stride][h][w], input[b][k_c + order * order_stride][h - k_h][w - k_w], kernel[c + order * order_stride][k_c][K_H - k_h - 1][K_W - k_w - 1]);
            }
        }
    } 

    }
} // namespace

std::vector<torch::Tensor> inv_conv_fwd_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output) {

  // assuming batch size 1 for initial implementation
  // all the tensors above are 2D only
  const auto B = output.size(0);
  const auto C = output.size(1);
  const auto H = output.size(2);
  const auto W = output.size(3); // assuming height and width to be the same
  // const auto k = kernel.size(0); // assuming square kernel

  // printf("%d %d", n,k);
  const auto n = H < W ? H : W; //samller dimension  -- min(H, W)
  const auto m = H > W ? H : W; //larger dimension   -- max(H, W)

  dim3 threads(1024, 1, 1);         // Fix the number of threads
  dim3 blocks(B, 4, 1);             // Fix the number of grids = Batch_Size

  const auto order_stride = (int) C / 4;

  for (int d = 1; d <= H + W -1; d++) { // Iterating over diagonal index
    for (int c = 0; c < (int) C / 4; c++) {
        // int max_threads = 1024;

        // all elements of the dth diagonal computed in parallel
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

        AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_fwd_cuda_inverse", ([&] {
          inv_conv_fwd_kernel<scalar_t><<<blocks, threads>>>(
              input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
              kernel.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
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
} // cinc = inv_conv_fwd


// cuda kernel1 for dy
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

// cuda kernel-2 for inv_conv_dy
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
} // cinc = inv_conv_dy








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
  
  return {output};
} // cinc = inv_conv_dw







// //////////              TRASH TRASH TRASH                   ///////////

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