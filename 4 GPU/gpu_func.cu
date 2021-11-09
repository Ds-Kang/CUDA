
#include <cuda_runtime.h>
#include "gpu_func.h"
#include "timer.h"
#include "util.h"

#define CHECK_CUDA(err) \
  do { \
    cudaError_t CHECK_CUDA_err = (err); \
    if (CHECK_CUDA_err != cudaSuccess) { \
      printf("[%s:%d] CUDA error %d (%s)\n", __FILE__, __LINE__, CHECK_CUDA_err, cudaGetErrorString(CHECK_CUDA_err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)
  
  // __global__ void gpu_Conv2d(float* input_col, float* weight, float* bias, float* output, int OO, int RSC, int K, bool has_bias);
  // __global__ void gpu_to_col(float* input, float* weight, float* input_col, int stride, int pad, int dilation, int C, int H, int W, int K, int R, int S, int OH, int OW);
  // __global__ void gpu_to_col_trans(float* input, float* weight, float* input_col, int stride, int pad, int C, int H, int W, int R, int S, int OH, int OW);
  // __global__ void gpu_alloc_out(float* input, float* output, int C, int H, int W);
  // __global__ void gpu_UnnormalizeAB(float* input, float* output, int H, int W, int C);
  // __global__ void gpu_Upsample(float* input, float* output, float scale_factor, int C, int H, int W, int OH, int OW);
  // __global__ void gpu_Softmax(float* input, float* output, int C, int H, int W);
  // __global__ void gpu_BatchNorm2d(float* input, float* weight, float* bias, float* running_mean, float* running_var, float* output, const float eps, int C, int H, int W);
  // __global__ void gpu_ReLU(float* inout, int H, int W, int C);
  // __global__ void gpu_NormalizeL(float* input, float* output, int H, int W);


__global__ void gpu_change_dim(float* input, int R, int S, int C, int K){
  int r = blockDim.x * blockIdx.x + threadIdx.x;
  int s = blockDim.y * blockIdx.y + threadIdx.y;
  extern __shared__ float tmp[];
  if (r >= R || s >= S) return;

  
  for (int c=0;c<C;++c){
    for (int k=0;k<K;++k){
      tmp[r*S*K*C+s*K*C+c*K+k]= input[k*R*S*C+c*R*S+r*S+s];
    }
  }
  for (int c=0;c<C;++c){
    for (int k=0;k<K;++k){
      input[r*S*K*C+s*K*C+c*K+k]=tmp[r*S*K*C+s*K*C+c*K+k];
    }
  }
}

__global__ void gpu_NormalizeL(float* input, float* output, int H, int W) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  if (h >= H || w >= W) return;

  output[h * W + w] = (input[h * W + w] - 50) / 100;
}


__global__ void gpu_Conv2d(float* input_col, float* weight, float* bias, float* output, int OO, int RSC, int K, bool has_bias) {

  int oo = blockDim.x * blockIdx.x + threadIdx.x;
  int k = blockDim.y * blockIdx.y + threadIdx.y;

  if (oo >= OO || k >= K) return;

  float s= has_bias ? bias[k] : 0;
  for (int rsc=0; rsc<RSC; rsc++){
    s+=input_col[oo*RSC+rsc]*weight[rsc*K+k];
  }
  output[oo*K+k]=s;
}

__global__ void gpu_to_col(float* input, float* weight, float* input_col, int stride, int pad, int dilation, int C, int H, int W, int K, int R, int S, int OH, int OW){

  int oh = blockDim.x * blockIdx.x + threadIdx.x;
  int ow = blockDim.y * blockIdx.y + threadIdx.y;

  if (oh >= OH || ow >= OW) return;

  for (int r = 0; r < R; ++r) {
    int h = oh * stride - pad + dilation*r;
    for (int s = 0; s < S; ++s) {
      int w = ow * stride - pad + dilation*s;
      for (int c = 0; c < C; ++c) {
        if (h < 0 || h >= H || w < 0 || w >= W) input_col[oh*OW*R*S*C + ow*R*S*C+ r*S*C+ s*C + c] = 0;
        else input_col[oh*OW*R*S*C + ow*R*S*C+ r*S*C+ s*C + c] = input[h * C * W + w * C + c];
      }
    }
  }
}



__global__ void gpu_ReLU(float* inout, int H, int W, int C) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  for (int c = 0; c < C; ++c) {
    int idx = h * W * C + w * C + c;
    inout[idx] = inout[idx] > 0 ? inout[idx] : 0;
  }
}

__global__ void gpu_BatchNorm2d(float* input, float* weight, float* bias, float* running_mean, float* running_var, float* output, int C, int H, int W, const float eps) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  if (h >= H || w >= W) return;

  for (int c = 0; c < C; ++c) {
    int idx = h * W * C + w * C + c;
    output[idx] = (input[idx] - running_mean[c]) / sqrtf(running_var[c] + eps) * weight[c] + bias[c];
  }

}

__global__ void gpu_to_col_trans(float* input, float* weight, float* input_col, int stride, int pad, int C, int H, int W, int R, int S, int OH, int OW){

  int oh = blockDim.x * blockIdx.x + threadIdx.x;
  int ow = blockDim.y * blockIdx.y + threadIdx.y;

  if (oh >= OH || ow >= OW) return;

  for (int r = 0; r < R; ++r) {
    int h = (oh + pad - r) / stride;
    for (int s = 0; s < S; ++s) {
      int w = (ow + pad - s) / stride;
      for (int c = 0; c < C; ++c) {
        if ((h < 0 || h >= H || w < 0 || w >= W) || ((ow + pad - s) % stride != 0) || ((oh + pad - r) % stride != 0)) input_col[oh*OW*R*S*C + ow*R*S*C+ r*S*C+ s*C + c] = 0;
        else input_col[oh*OW*R*S*C + ow*R*S*C+ r*S*C+ s*C + c] = input[h * C * W + w * C + c];
      }
    }
  }
}

__global__ void gpu_Softmax(float* input, float* output, int C, int H, int W) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  if (h >= H || w >= W) return;

  float sum = 0;
  for (int c = 0; c < C; ++c) {
    sum += expf(input[h * W * C + w * C + c]);
  }
  for (int c = 0; c < C; ++c) {
    output[h * W * C + w * C + c] = expf(input[h * W * C + w * C + c]) / sum;
  }  
}
__global__ void gpu_Upsample(float* input, float* output, float scale_factor, int C, int H, int W, int OH, int OW) {
  
  int oh = blockDim.x * blockIdx.x + threadIdx.x;
  int ow = blockDim.y * blockIdx.y + threadIdx.y;
  if (oh >= OH || ow >= OW) return;

  for (int c = 0; c < C; ++c) {
  float h = (oh + 0.5) / scale_factor - 0.5;
  float w = (ow + 0.5) / scale_factor - 0.5;
  int h0 = floorf(h), w0 = floorf(w);
  int h1 = h0 + 1, w1 = w0 + 1;
  float h_offset = h - h0, w_offset = w - w0;
  float w00 = (1 - h_offset) * (1 - w_offset);
  float w01 = (1 - h_offset) * w_offset;
  float w10 = h_offset * (1 - w_offset);
  float w11 = h_offset * w_offset;
  h0 = h0 < 0 ? 0 : (h0 > H - 1 ? H - 1 : h0);
  h1 = h1 < 0 ? 0 : (h1 > H - 1 ? H - 1 : h1);
  w0 = w0 < 0 ? 0 : (w0 > W - 1 ? W - 1 : w0);
  w1 = w1 < 0 ? 0 : (w1 > W - 1 ? W - 1 : w1);
  output[oh * OW * C + ow * C + c] = w00 * input[h0 * W * C+ w0 * C + c]
                                        + w01 * input[h0 * W * C+ w1 * C + c]
                                        + w10 * input[h1 * W * C+ w0 * C + c]
                                        + w11 * input[h1 * W * C+ w1 * C + c];

  }
}

__global__ void gpu_UnnormalizeAB(float* input, float* output, int H, int W, int C) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  if (h >= H || w >= W) return;

  for (int c = 0; c < C; ++c) {
    output[h * W * C + w * C + c] = input[h * W * C + w * C + c] * 110;
  }
}


__global__ void gpu_alloc_out(float* input, float* output, int C, int H, int W) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  if (h >= H || w >= W) return;

  for (int c = 0; c < C; ++c) {
    output[c * H * W + h * W + w] = input[h * W * C + w * C + c];
  }
}

void gpu_malloc(float** adr, int size){
  CHECK_CUDA(cudaMalloc(adr, size));
}

void gpu_free(float** adr){
  CHECK_CUDA(cudaFree(*adr));
}

void gpu_memcpy(float** in, float** out, int size, bool to_gpu){
  if (to_gpu) CHECK_CUDA(cudaMemcpy(*in, *out, size, cudaMemcpyHostToDevice));
  else CHECK_CUDA(cudaMemcpy(*in, *out, size, cudaMemcpyDeviceToHost));
}

void gpu_sync(){
  CHECK_CUDA(cudaDeviceSynchronize());
}


/*
 * Normalize L channel.
 * Formula: y = (x - 50) / 100
 */

 void NormalizeL(Tensor input, Tensor output, float* gpu_input, float* gpu_output) {
  int H = input.shape[0], W = input.shape[1];
  CHECK_ERROR(input.shape[2] == 1 && output.shape[2] == 1 && output.shape[0] == H && output.shape[1] == W, "Size mismatch");

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);

  gpu_NormalizeL<<<gridDim, blockDim>>>(gpu_input, gpu_output, H, W);
  
  CHECK_CUDA(cudaDeviceSynchronize());
}



/*
 * Convolution
 * input shape = (C, H, W)
 * weight shape = (K, C, R, S)
 * bias shape = (K)
 * output shape = (K, OH, OW)
 * -->
 * input shape = (H, W, C)
 * weight shape = (R, S, C, K)
 * bias shape = (K)
 * output shape = (OH, OW, K)
 * 
 *   where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 *         OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 */






void Conv2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_output, float* gpu_input_col, bool has_bias) {  
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  int K = weight.shape[3], R = weight.shape[0], S = weight.shape[1];
  int OH = output.shape[0], OW = output.shape[1];
  CHECK_ERROR(OH == (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1, "Output height mismatch");
  CHECK_ERROR(OW == (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1, "Output width mismatch");
  CHECK_ERROR(weight.shape[2] == C && (!has_bias || bias.shape[0] == K) && output.shape[2] == K, "Channel size mismatch");
  
  {
    dim3 gridDim(OH, OW, 1);
    dim3 blockDim(1, 1, 1);
    gpu_to_col<<<gridDim, blockDim>>>(gpu_input, gpu_weight, gpu_input_col, stride, pad, dilation, C, H, W, K, R, S, OH, OW);
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  {
    if (K%32==0){
      dim3 gridDim(K/32, (OH*OW)/32, 1);
      dim3 blockDim(32, 32, 1);
      
      gpu_Conv2d<<<gridDim, blockDim>>>(gpu_input_col, gpu_weight, gpu_bias, gpu_output, OH*OW, R*S*C, K, has_bias);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    else{
      dim3 gridDim(K, OH*OW, 1);
      dim3 blockDim(1, 1, 1);
      gpu_Conv2d<<<gridDim, blockDim>>>(gpu_input_col, gpu_weight, gpu_bias, gpu_output, OH*OW, R*S*C, K, has_bias);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
  
  }
}


/*
 * ReLU
 * Formula: y = max(x, 0)
 */


void ReLU(Tensor inout, float* gpu_inout) {
  int C = inout.shape[2], H = inout.shape[0], W = inout.shape[1];

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);

  gpu_ReLU<<<gridDim, blockDim>>>(gpu_inout, H, W, C);
  
  CHECK_CUDA(cudaDeviceSynchronize());
}

/*
 * Batch Normaliztion
 * input shape = (C, H, W)
 * weight shape = (C)
 * bias shape = (C)
 * running_mean shape = (C)
 * running_var shape = (C)
 * output shape = (C, H, W)
 */

void BatchNorm2d(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_running_mean, float* gpu_running_var, float* gpu_output, const float eps) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];

  CHECK_ERROR(weight.shape[0] == C && bias.shape[0] == C && running_mean.shape[0] == C && running_var.shape[0] == C, "Shape mismatch");
  CHECK_ERROR(output.shape[2] == C && output.shape[0] == H && output.shape[1] == W, "Shape mismatch");

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);

  gpu_BatchNorm2d<<<gridDim, blockDim>>>(gpu_input, gpu_weight, gpu_bias, gpu_running_mean, gpu_running_var, gpu_output, C, H, W, eps);
  
  gpu_sync();
}

/*
 * Transposed convolution
 * input shape = (C, H, W)
 * weight shape = (C, K, R, S) -> (R, S, C, K), {512, 256, 4, 4} -> {4, 4, 256, 512}
 * bias shape = (K)
 * output shape = (K, OH, OW)
 *   where OH = (H - 1) * stride - 2 * pad + R
 *         OW = (W - 1) * stride - 2 * pad + S
 */




void ConvTranspose2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_output, float* gpu_input_col) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  int K = weight.shape[3], R = weight.shape[0], S = weight.shape[1];
  int OH = output.shape[0], OW = output.shape[1];

  CHECK_ERROR(OH == (H - 1) * stride - 2 * pad + R, "Output height mismatch");
  CHECK_ERROR(OW == (W - 1) * stride - 2 * pad + S, "Output width mismatch");
  CHECK_ERROR(weight.shape[2] == C && bias.shape[0] == K && output.shape[2] == K, "Channel size mismatch");
  {
    dim3 gridDim(OH, OW, 1);
    dim3 blockDim(1, 1, 1);

    gpu_to_col_trans<<<gridDim, blockDim>>>(gpu_input, gpu_weight, gpu_input_col, stride, pad, C, H, W, R, S, OH, OW);
    gpu_sync();
  }
  {
    dim3 gridDim(K/32, (OH*OW)/32, 1);
    dim3 blockDim(32, 32, 1);
  
    gpu_Conv2d<<<gridDim, blockDim>>>(gpu_input_col, gpu_weight, gpu_bias, gpu_output, OH*OW, R*S*C, K, 1);
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

/*
 * Softmax
 * Formula: y = e^x / sum(e^x)
 */

void Softmax(Tensor input, Tensor output, float* gpu_input, float* gpu_output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  CHECK_ERROR(output.shape[2] == C && output.shape[0] == H && output.shape[1] == W, "shape mismatch");

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);

  gpu_Softmax<<<gridDim, blockDim>>>(gpu_input, gpu_output, C, H, W);
  
  gpu_sync();

}

/*
 * Bilinear interpolation
 * input shape = (C, H, W)
 * output shape = (C, floor(H * scale_factor), floor(W * scale_factor))
 */

void Upsample(Tensor input, Tensor output, float scale_factor, float* gpu_input, float* gpu_output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  int OH = output.shape[0], OW = output.shape[1];
  CHECK_ERROR(output.shape[2] == C && OH == floorf(H * scale_factor) && OW == floorf(W * scale_factor), "shape mismatch");

  dim3 gridDim(OH, OW, 1);
  dim3 blockDim(1, 1, 1);

  gpu_Upsample<<<gridDim, blockDim>>>(gpu_input, gpu_output, scale_factor, C, H, W, OH, OW);

}

/*
 * Unnormalize A and B channel
 * Formula: y = x * 110
 */
 
void UnnormalizeAB(Tensor input, Tensor output,float* gpu_input,float* gpu_output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  CHECK_ERROR(output.shape[2] == C && output.shape[0] == H && output.shape[1] == W, "shape mismatch");

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);

  gpu_UnnormalizeAB<<<gridDim, blockDim>>>(gpu_input, gpu_output, H, W, C);
  
  gpu_sync();
  
}


void alloc_out(Tensor input, Tensor output, float* gpu_input, float* gpu_output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);
  gpu_alloc_out<<<gridDim, blockDim>>>(gpu_input, gpu_output, C, H, W);
  
  gpu_sync();

}