#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullptr
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  float l_sum = 0;
  float l_sum_square = 0;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_sum_square += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }
  
  // Step 2
  // Block reduce sum and sum square
  float mean_dim = (float)hidden_size * 4.f;
  float reduce_val[2] = {l_sum, l_sum_square};
  blockReduce<ReduceType::kSum, 2>(reduce_val);

  //  Write shared
  __shared__ float s_means;
  __shared__ float s_inv_stds;

  if(threadIdx.x == 0) {
    s_means = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = s_means;
    }
    float s_var = reduce_val[1] / mean_dim - s_means * s_means + LN_EPSILON;
    vars[blockIdx.x] = s_var;
    s_inv_stds = rsqrtf(s_var);
    
  }
  __syncthreads(); 

  // Step 3
  float4 *ln_res_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);

  for(uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
      float4 res;
      float4 val = inp_f4[idx];
      float4 l_scale = scale_f4[idx]; 
      float4 l_bias = bias_f4[idx];    
      
      res.x = l_scale.x * ((val.x - s_means) * s_inv_stds) + l_bias.x;
      res.y = l_scale.y * ((val.y - s_means) * s_inv_stds) + l_bias.y;
      res.z = l_scale.z * ((val.z - s_means) * s_inv_stds) + l_bias.z;
      res.w = l_scale.w * ((val.w - s_means) * s_inv_stds) + l_bias.w;
      ln_res_f4[idx] = res;
  }
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backward kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  int idx_in_hori = blockIdx.x;
  int tx = threadIdx.x, ty = threadIdx.y;
  int steps = (rows + TILE_DIM - 1) / TILE_DIM; // Compute stride steps in vertical.
  int curr_col = idx_in_hori * TILE_DIM + tx; // Every thread compute partial of a column.

  // Step 1
  float l_sum_dout = 0;
  float l_sum_dout_mul_xhat = 0;
  // Looping rows to compute local sum                                     
  for(int st = 0; st < steps; st++){
    int curr_row = ty + TILE_DIM * st;

    if(curr_row < rows && curr_col < width) {
      // Reading in row-major 
      float dout = (float)out_grad[curr_row * width + curr_col];
      l_sum_dout += dout;
      
      float xhat;
      if(means) {
        float input = (float)inp[curr_row * width + curr_col];
        xhat = (input- (float)means[curr_row]) * rsqrtf((float)vars[curr_row] + LN_EPSILON);
      } else {
        float output = (float)inp[curr_row * width + curr_col];
        xhat = __fdividef(output - (float)betta[curr_col], (float)gamma[curr_col]);
      }

      l_sum_dout_mul_xhat += dout * xhat;
    }  
  }

  // Step 2
  if(curr_col < width) {
    // Store partial sum in shared memory in col-major way.
    betta_buffer[tx][ty] = l_sum_dout;
    gamma_buffer[tx][ty] = l_sum_dout_mul_xhat;
  } else {
    betta_buffer[tx][ty] = 0;
    gamma_buffer[tx][ty] = 0;
  }
  __syncthreads(); 

  // Step 3
  float sum_dout = betta_buffer[ty][tx];
  float sum_dout_mul_xhat = gamma_buffer[ty][tx];

  #pragma unroll
  for(int i = TILE_DIM / 2; i > 0; i >>= 1) {  
    // Reduce sum in warp
    sum_dout += g.shfl_down(sum_dout, i);
    sum_dout_mul_xhat += g.shfl_down(sum_dout_mul_xhat, i);
  }

  // Step 4
  if(g.thread_rank() == 0) {
    // Write back to global, lan0 0 for wrting tx col
    int col_idx = idx_in_hori * TILE_DIM + ty;
    if (col_idx < width) {
      betta_grad[col_idx] = (T) sum_dout;
      gamma_grad[col_idx] = (T) sum_dout_mul_xhat;
    }
  }
  /// END ASSIGN4_2_2
}

/**
@brief: helper fucntion for ker_ln_bw_dinp
*/
__device__ __forceinline__ float4 compute_dxhat_f4(float4 out_grad, float4 gamma) {
  return make_float4(out_grad.x * gamma.x, out_grad.y * gamma.y, out_grad.z * gamma.z, 
                     out_grad.w * gamma.w);
}

__device__ __forceinline__ float4 compute_xhat_f4_from_input(float4 input, float mean, float var){
  return make_float4((input.x - mean) * rsqrtf(var + LN_EPSILON), (input.y - mean) * rsqrtf(var + LN_EPSILON),
                     (input.z - mean) * rsqrtf(var + LN_EPSILON), (input.w - mean) * rsqrtf(var + LN_EPSILON));
}

__device__ __forceinline__ float4 compute_xhat_f4_from_output(float4 output, float4 gamma, float4 betta) {
  return make_float4(__fdividef(output.x - betta.x, gamma.x), __fdividef(output.y - betta.y, gamma.y),
                     __fdividef(output.z - betta.z, gamma.z), __fdividef(output.w - betta.w, gamma.w));
}

/**
@brief: ker_ln_bw_dinp
Layer norm backward kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + blockIdx.x * hidden_dim;
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  const float4 *betta_f4 = reinterpret_cast<const float4 *>(betta);
  const float *mean = (means != nullptr) ?
                       means + blockIdx.x :
                       nullptr;
  
  const float *var =  vars + blockIdx.x;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_dim;

  int tx = threadIdx.x;
  // Step 1
  float4 dxhat = {0};

  if(tx < hidden_dim) {
    float4 l_out_grad = out_grad_f4[tx];
    float4 l_gamma = gamma_f4[tx];
    dxhat = compute_dxhat_f4(l_out_grad, l_gamma);
  }
  
  // Step 2
  float4 xhat = {0};

  if(tx < hidden_dim) {
    if(mean) {
      float4 l_input = inp_f4[tx];
      float m = *mean;
      float v = *var;

      xhat = compute_xhat_f4_from_input(l_input, m, v);
    } else {
      float4 l_output = inp_f4[tx];
      float4 l_betta = betta_f4[tx];
      float4 l_gamma = gamma_f4[tx];

      xhat = compute_xhat_f4_from_output(l_output, l_gamma, l_betta);
    }
  }

  // Step 3
  float l_sum_dxhat = 0;
  float l_sum_dxhat_mul_xhat = 0;

  if(tx < hidden_dim) {
    l_sum_dxhat += (dxhat.x + dxhat.y + dxhat.z + dxhat.w);
    l_sum_dxhat_mul_xhat += (dxhat.x * xhat.x + 
                             dxhat.y * xhat.y + 
                             dxhat.z * xhat.z + 
                             dxhat.w * xhat.w);
  }
  
  
  __shared__ float s_sum_dxhat;
  __shared__ float s_sum_dxhat_mul_xhat;
  
  blockReduce<ReduceType::kSum, 1>(&l_sum_dxhat);
  blockReduce<ReduceType::kSum, 1>(&l_sum_dxhat_mul_xhat);
  
  if(threadIdx.x == 0){
    s_sum_dxhat = l_sum_dxhat;
    s_sum_dxhat_mul_xhat = l_sum_dxhat_mul_xhat;
  }
  __syncthreads();
 
  // Step 4
  if(tx < hidden_dim) {
    float4 res;
    float v = *var;
    res.x = (dxhat.x - __fdividef(s_sum_dxhat + xhat.x * s_sum_dxhat_mul_xhat, hidden_dim << 2)) * 
           rsqrtf(v + LN_EPSILON);
    res.y = (dxhat.y - __fdividef(s_sum_dxhat + xhat.y * s_sum_dxhat_mul_xhat, hidden_dim << 2)) * 
           rsqrtf(v + LN_EPSILON);
    res.z = (dxhat.z - __fdividef(s_sum_dxhat + xhat.z * s_sum_dxhat_mul_xhat, hidden_dim << 2)) * 
           rsqrtf(v + LN_EPSILON);
    res.w = (dxhat.w - __fdividef(s_sum_dxhat + xhat.w * s_sum_dxhat_mul_xhat, hidden_dim << 2)) * 
           rsqrtf(v + LN_EPSILON);

    float4 * inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + blockIdx.x * hidden_dim;
    inp_grad_f4[tx] = res;
  }
  
  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Create CUDA events for timing
  // cudaEvent_t start, stop, start_dinp, stop_dinp;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventCreate(&start_dinp);
  // cudaEventCreate(&stop_dinp);
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  dim3 grid_dim((hidden_dim + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  
  //cudaEventRecord(start, stream_1);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);
  //cudaEventRecord(stop, stream_1);
  //cudaStreamSynchronize(stream_1);
  
  //float gamma_beta_ms = 0;
  //cudaEventElapsedTime(&gamma_beta_ms, start, stop);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  
  //cudaEventRecord(start_dinp, stream_2);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);
  //cudaEventRecord(stop_dinp, stream_2);
  //cudaStreamSynchronize(stream_2);
  
  //float dinp_ms = 0;
  //cudaEventElapsedTime(&dinp_ms, start_dinp, stop_dinp);
  
  // printf("[Timing] Gamma/Beta kernel: %.3f ms, Dinp kernel: %.3f ms, Total: %.3f ms\n",
  //        gamma_beta_ms, dinp_ms, gamma_beta_ms + dinp_ms);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
  
  // // Destroy CUDA events
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
  // cudaEventDestroy(start_dinp);
  // cudaEventDestroy(stop_dinp);
}}
}}

