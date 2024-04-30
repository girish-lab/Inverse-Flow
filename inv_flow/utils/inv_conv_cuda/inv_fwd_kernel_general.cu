#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
}
// cinc = inv_conv_fwd
// #include <torch/extension.h>

// #include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// #include <vector>

// namespace {

 
// // Implementation for batch size = 1, channels = 1
// template <typename scalar_t>
// __global__ void inv_conv_fwd_cuda_inverse_kernel(
//     const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
//     const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> kernel,
//     torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output, int k, int d) {
  
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
//     const auto m  = output.size(0); // assuming height and width to be the same
//     const auto n  = output.size(1);
//     // NOTE: For some reason kernel.size(0) doesnot work correctly here

//     // Compute the index from the dth diagonal assigned to the thread
//     int h, i, j;
//     h = tid % m; // batch index encoded in lower indices modulo batchsize
//     tid = (tid - h) / m; // remaining part of tid has the index along the diagonal
//     if (d <= n) {
//         i = d - 1 - tid;
//         j = tid;
//     }
//     else {
//         j = (d - n)  + tid;
//         i = n - 1 - tid;
//     }
//     // compute entry of the output in the diagonal d assigned to this thread
//     for (int qi = 0; qi < k; qi++) {
//         for (int qj = 0; qj < k; qj++) {
//             if ( i-(k-1) + qi >0 && j-(k-1)+ qj >0 && !((qi==k-1) && (qj==k-1))) {
//               printf("%d %d %d %d %d %d %d\n", h, i, j, qi, qj, i-(k-1)+qi, j-(k-1)+qj);
//               output[h][i][j] = 0;
//               // output[h][i][j] += kernel[qi][qj] * input[h][i-(k-1)+qi][j-(k-1)+qj];
//             }
//         }
//     }
//     // output[h][1][1] = 100;
// }

// } // namespace

// std::vector<torch::Tensor> inv_conv_fwd_cuda_inverse(
//     torch::Tensor input,
//     torch::Tensor kernel,
//     torch::Tensor output) {

//   // assuming batch size 1 for initial implementation
//   // all the tensors above are 2D only

//   const auto m = output.size(0);
//   const auto n = output.size(1); // assuming height and width to be the same
//   const auto k = kernel.size(0); // assuming square kernel

//   // printf("%d %d", n,k);

//   for (int d = 1; d <= 2*n-1; d++) { // Iterating over diagonal index

//     int max_threads = 1024;

//     // all elements of the dth diagonal computed in parallel
//     int threads = d;
//     if (d > n) {
//       threads = 2*n-d;
//     }
//     threads = threads*m;
    
//     const int blocks = (max_threads +threads)/max_threads; // use multiple blocks if 2d-1 > threads

//     AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_fwd_cuda_inverse", ([&] {
//       inv_conv_fwd_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
//           input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
//           kernel.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
//           output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), k, d);
//     }));

//     // synchronize all threads
//     cudaDeviceSynchronize();

//   }

//   return {output};
// }
// cinc = inv_conv_fwd
// #include <torch/extension.h>

// #include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// #include <vector>

// namespace {

 
// // Implementation for batch size = 1, channels = 1
// template <typename scalar_t>
// __global__ void inv_conv_fwd_cuda_inverse_kernel(
//     const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> input,
//     const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> kernel,
//     torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output, int k, int d) {
  
//     int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread id
//     const auto m = output.size(0); // assuming height and width to be the same
//     const auto n = output.size(1);

//     int h, i, j;
//     h = tid % m; // batch index encoded in lower indices modulo batchsize
//     tid = (tid - h) / m; // remaining part of tid has the index along the diagonal
//     // i = tid / n;
//     // j = tid % n;
//     if (d <= n) {
//         i = d - 1 - tid;
//         j = tid;
//     } 
//     else {
//         j = (d - n)  + tid;
//         i = n - 1 - tid;
//     }

    
//     scalar_t result = 0.0;
//     for (int a = 0; a < k; a++) {
//         for (int b = 0; b < k; b++) {
//             if ( i-(k-1)+a >=0 && j-(k-1)+b >=0 && !((a==k-1) && (b==k-1))) { 
//                result += input[h][i-(k-1)+a][j-(k-1)+b] * kernel[a][b];
//             }
//         }
//     }
//     output[h][i][j] = result;
    
// }

// } // namespace

// std::vector<torch::Tensor> inv_conv_fwd_cuda_inverse(
//     torch::Tensor input,
//     torch::Tensor kernel,
//     torch::Tensor output) {

//   // assuming batch size 1 for initial implementation
//   // all the tensors above are 2D only

//   const auto m = output.size(0);
//   const auto n = output.size(1); // assuming height and width to be the same
//   const auto k = kernel.size(0); // assuming square kernel

//   // printf("%d %d", n,k);

//   for (int d = 1; d <= 2*n-1; d++) { // Iterating over diagonal index

//     int max_threads = 1024;
//     // all elements of the dth diagonal computed in parallel
//     int threads = d;
//     if (d > n) {
//       threads = 2 * n - d;
//     }
//     threads = threads * m;
//     const int blocks = (max_threads + threads) / max_threads; // use multiple blocks if 2d-1 > threads

//     AT_DISPATCH_FLOATING_TYPES(input.type(), "inv_conv_fwd_cuda_inverse", ([&] {
//       inv_conv_fwd_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
//           input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
//           kernel.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
//           output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), k, d);
//     }));

//     // synchronize all threads
//     cudaDeviceSynchronize();

//   }

//   return {output};
// }
// cinc = inv_conv_fwd

// #include <torch/extension.h>

// #include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// #include <vector>

// namespace {

// template <typename scalar_t>
// __global__ void conv_cuda_forward_kernel(
//     const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> input,
//     const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> kernel,
//     torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output,
//     int k) {

//     int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread id
//     const auto m = output.size(0); // assuming height and width to be the same
//     const auto n = output.size(1);

//     int h, i, j;
//     h = tid % m; // batch index encoded in lower indices modulo batchsize
//     tid = (tid - h) / m; // remaining part of tid has the index along the diagonal
//     i = tid / n;
//     j = tid % n;

//     if (i < n && j < n) {
//         scalar_t result = 0.0;
//         for (int a = 0; a < k; a++) {
//             for (int b = 0; b < k; b++) {
//                 if (i + a < n && j + b < n) {
//                     result += input[h][i + a][j + b] * kernel[a][b];
//                 }
//             }
//         }
//         output[h][i][j] = result;
//     }
// }

// } // namespace

// std::vector<torch::Tensor> conv_cuda_forward(
//     torch::Tensor input,
//     torch::Tensor kernel,
//     torch::Tensor output) {

//     const auto m = output.size(0);
//     const auto n = output.size(1); // assuming height and width to be the same
//     const auto k = kernel.size(0); // assuming square kernel

//     int max_threads = 1024;
//     int threads = m * n;
//     const int blocks = (max_threads + threads - 1) / max_threads;

//     AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_forward_cuda", ([&] {
//         conv_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
//             input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
//             kernel.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
//             output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(), k);
//     }));

//     cudaDeviceSynchronize();

//     return {output};
// }
// if (i - qi >= 0 && j - qj >= 0  && i - qi < n && j -qj < n) { //
            // printf("inverse%g %d, %d\n", kernel[qi][qj], qi, qj);
            // printf("%d %d %d %d %d %d %d %d\n", h, i, j, qi, qj, i-(k-1)+qi, j-(k-1)+qj, output[h][i-(k-1)+qi][j-(k-1)+qj]);
            // count++;
            // printf("h %d\n", h);
            // printf("qi ,qj ,d  i, j, count:- %d %d %d %d %d %d\n", qi, qj, d, i, j, count);
            // printf("  d,   i, j,   qi, qj:-  %d %d %d %d %d\n", d, i, j, qi, qj);
            // if (i-(k-1) + qi >=0 && j-(k-1) + qj >=0) { //&& !((qi==k-1) && (qj==k-1))
            //   output[h][i][j] += kernel[qi][qj]*input[h][i- (k-1) + qi][j- (k-1) + qj];
            //   // printf("\n fwd %g %d, %d\n", kernel[qi][qj], qi, qj);
            //   printf("\n                   d,   i, j,   qi, qj:-  %d %d %d %d %d\n", d, i, j, qi, qj);
            //   printf("              \n");
            // }
            // if ( i-qi >= 0 && j-qj >= 0 ) {
            //     printf("output[i][j] %g \n", output[h][i][j]);
            //     output[h][i][j] += input[h][i-(k-1)-qi][j-(k-1)-qj] * kernel[k-1 - qi][k-1 -qj];
            //     printf("here h, d, i, j, qi, qj, val: %d %d %d %d %d %d", h,  d,  i, j, qi, qj);
            //     printf(" %g %g %g \n", input[h][i-qi][j-qj], kernel[k-1 - qi][k-1 -qj], input[h][i - qi][j-qj] * kernel[k-1 - qi][k-1 -qj]);
            // }
              // # if __CUDA_ARCH__ >= 200
              //   printf("%d %d %d %d %d %d \n",i,j,a,b,n,k);
              // #endif
              // printf(" %d %d %d %d %d %d\n", i, j, qi, qj, i-(k-1)+qi, j-(k-1)+qj);
              // printf("here\n");
              // printf("%g\n", output[h][i][j]);
              // printf("%g %d, %d\n", kernel[qi][qj], qi, qj);
              // printf("%g\n", output[h][i-(k-1) + qi][j- (k-1) +qj]);
              // printf("qi_qj %d %d\n", qi, qj);

            
            // if (i-(k-1)+ qi < 0 || j-(k-1)+ qj < 0) {
            //     // output[h][i][j] -= kernel[qi][qj]*input[h][i-(k-1)+qi][j-(k-1)+qj];
            //     printf("fwd %g %d, %d\n", kernel[qi][qj], qi, qj);
            // }
            // else {
            //     //
            //     if (i-(k-1)+ qi >=0 && j-(k-1)+ qj >=0 && (qi==k-1 && qj==k-1)) {
            //         output[h][i][j] += input[h][qi][qj];
            //         printf("here qi=k-1, qj=k-1,\n");
            //     }
            // }






// # impoirtant
// # multiple kernelin one cu file
// #include <torch/extension.h>

// #include <vector>

// // s'(z) = (1 - s(z)) * s(z)
// torch::Tensor d_sigmoid(torch::Tensor z) {
//   auto s = torch::sigmoid(z);
//   return (1 - s) * s;
// }

// // tanh'(z) = 1 - tanh^2(z)
// torch::Tensor d_tanh(torch::Tensor z) {
//   return 1 - z.tanh().pow(2);
// }

// // elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
// torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
//   auto e = z.exp();
//   auto mask = (alpha * (e - 1)) < 0;
//   return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
// }

// std::vector<torch::Tensor> lltm_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias,
//     torch::Tensor old_h,
//     torch::Tensor old_cell) {
//   auto X = torch::cat({old_h, input}, /*dim=*/1);

//   auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
//   auto gates = gate_weights.chunk(3, /*dim=*/1);

//   auto input_gate = torch::sigmoid(gates[0]);
//   auto output_gate = torch::sigmoid(gates[1]);
//   auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

//   auto new_cell = old_cell + candidate_cell * input_gate;
//   auto new_h = torch::tanh(new_cell) * output_gate;

//   return {new_h,
//           new_cell,
//           input_gate,
//           output_gate,
//           candidate_cell,
//           X,
//           gate_weights};
// }

// std::vector<torch::Tensor> lltm_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gate_weights,
//     torch::Tensor weights) {
//   auto d_output_gate = torch::tanh(new_cell) * grad_h;
//   auto d_tanh_new_cell = output_gate * grad_h;
//   auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

//   auto d_old_cell = d_new_cell;
//   auto d_candidate_cell = input_gate * d_new_cell;
//   auto d_input_gate = candidate_cell * d_new_cell;

//   auto gates = gate_weights.chunk(3, /*dim=*/1);
//   d_input_gate *= d_sigmoid(gates[0]);
//   d_output_gate *= d_sigmoid(gates[1]);
//   d_candidate_cell *= d_elu(gates[2]);

//   auto d_gates =
//       torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

//   auto d_weights = d_gates.t().mm(X);
//   auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

//   auto d_X = d_gates.mm(weights);
//   const auto state_size = grad_h.size(1);
//   auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
//   auto d_input = d_X.slice(/*dim=*/1, state_size);

//   return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &lltm_forward, "LLTM forward");
//   m.def("backward", &lltm_backward, "LLTM backward");
// }