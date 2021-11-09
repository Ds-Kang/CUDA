#include "colorizer.h"

#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"
#include "timer.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(err) \
  do { \
    cudaError_t CHECK_CUDA_err = (err); \
    if (CHECK_CUDA_err != cudaSuccess) { \
      printf("[%s:%d] CUDA error %d (%s)\n", __FILE__, __LINE__, CHECK_CUDA_err, cudaGetErrorString(CHECK_CUDA_err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)


/*
 * Declarations
 */

struct Tensor {
  // Pointer to data
  float* buf;
  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape  = {2, 3}
  int shape[4];
};

extern int mpi_rank, mpi_size;



// Feature maps
static Tensor fm_normalize_l,
              fm1_0, fm1_1, fm1_2,
              fm2_0, fm2_1, fm2_2,
              fm3_0, fm3_1, fm3_2, fm3_3,
              fm4_0, fm4_1, fm4_2, fm4_3,
              fm5_0, fm5_1, fm5_2, fm5_3,
              fm6_0, fm6_1, fm6_2, fm6_3,
              fm7_0, fm7_1, fm7_2, fm7_3,
              fm8_0, fm8_1, fm8_2, fm8_3,
              fm_softmax, fm_model_out, fm_upsample4, pre_AB;

static float  *gpu_in, *gpu_out,
              *gpu_fm_normalize_l,
              *gpu_fm1_0, *gpu_fm1_1, *gpu_fm1_2,
              *gpu_fm2_0, *gpu_fm2_1, *gpu_fm2_2,
              *gpu_fm3_0, *gpu_fm3_1, *gpu_fm3_2, *gpu_fm3_3,
              *gpu_fm4_0, *gpu_fm4_1, *gpu_fm4_2, *gpu_fm4_3,
              *gpu_fm5_0, *gpu_fm5_1, *gpu_fm5_2, *gpu_fm5_3,
              *gpu_fm6_0, *gpu_fm6_1, *gpu_fm6_2, *gpu_fm6_3,
              *gpu_fm7_0, *gpu_fm7_1, *gpu_fm7_2, *gpu_fm7_3,
              *gpu_fm8_0, *gpu_fm8_1, *gpu_fm8_2, *gpu_fm8_3,
              *gpu_fm_softmax, *gpu_fm_model_out, *gpu_fm_upsample4, *gpu_pre_AB, *gpu_null,
              
              *gpu_model1_0_weight, *gpu_model1_0_bias, *gpu_model1_2_weight,
              *gpu_model1_2_bias,  *gpu_model1_4_weight,  *gpu_model1_4_bias,
              *gpu_model2_0_weight, *gpu_model2_0_bias, *gpu_model2_2_weight,
              *gpu_model2_2_bias, *gpu_model2_4_weight, *gpu_model2_4_bias,
              *gpu_model3_0_weight, *gpu_model3_0_bias, *gpu_model3_2_weight, *gpu_model3_2_bias,
              *gpu_model3_4_weight, *gpu_model3_4_bias, *gpu_model3_6_weight, *gpu_model3_6_bias,
              *gpu_model4_0_weight, *gpu_model4_0_bias, *gpu_model4_2_weight, *gpu_model4_2_bias,
              *gpu_model4_4_weight, *gpu_model4_4_bias, *gpu_model4_6_weight, *gpu_model4_6_bias,
              *gpu_model5_0_weight, *gpu_model5_0_bias, *gpu_model5_2_weight, *gpu_model5_2_bias,
              *gpu_model5_4_weight, *gpu_model5_4_bias, *gpu_model5_6_weight, *gpu_model5_6_bias,
              *gpu_model6_0_weight, *gpu_model6_0_bias, *gpu_model6_2_weight, *gpu_model6_2_bias,
              *gpu_model6_4_weight, *gpu_model6_4_bias, *gpu_model6_6_weight, *gpu_model6_6_bias,
              *gpu_model7_0_weight, *gpu_model7_0_bias, *gpu_model7_2_weight, *gpu_model7_2_bias,
              *gpu_model7_4_weight, *gpu_model7_4_bias, *gpu_model7_6_weight, *gpu_model7_6_bias,
              *gpu_model8_0_weight, *gpu_model8_0_bias, *gpu_model8_2_weight, *gpu_model8_2_bias,
              *gpu_model8_4_weight, *gpu_model8_4_bias, *gpu_model8_6_weight, *gpu_model8_6_bias,
              *gpu_model_out_weight, *gpu_model1_4_running_mean, *gpu_model1_4_running_var,
              *gpu_model2_4_running_mean, *gpu_model2_4_running_var, *gpu_model3_6_running_mean,
              *gpu_model3_6_running_var, *gpu_model4_6_running_mean, *gpu_model4_6_running_var,
              *gpu_model5_6_running_mean, *gpu_model5_6_running_var, *gpu_model6_6_running_mean,
              *gpu_model6_6_running_var, *gpu_model7_6_running_mean, *gpu_model7_6_running_var,
              
              *gpu_3_3_1_256_256, *gpu_3_3_64_128_128, *gpu_3_3_128_64_64, *gpu_3_3_256_64_64,
              *gpu_3_3_256_32_32, *gpu_3_3_512_32_32, *gpu_4_4_512_64_64, *gpu_1_1_256_64_64, *gpu_1_1_313_64_64;


// Layers
static Tensor Make3DTensor(int H, int W, int C);
static void DumpTensor(const char* filename, Tensor input, int dim);
__global__ void gpu_Conv2d(float* input_col, float* weight, float* bias, float* output, int OO, int RSC, int K, bool has_bias);
__global__ void gpu_to_col(float* input, float* weight, float* input_col, int stride, int pad, int dilation, int C, int H, int W, int K, int R, int S, int OH, int OW);
void Conv2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_output, float* gpu_input_col, bool has_bias = true);
__global__ void gpu_to_col_trans(float* input, float* weight, float* input_col, int stride, int pad, int C, int H, int W, int R, int S, int OH, int OW);
static void ConvTranspose2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_output, float* gpu_input_col);

static void change_dim(Tensor input);
static void change_dim_trans(Tensor input);

static void alloc_out(Tensor input, Tensor output, float* gpu_input, float* gpu_output);
static void Upsample(Tensor input, Tensor output, float scale_factor, float* gpu_input, float* gpu_output);
static void UnnormalizeAB(Tensor input, Tensor output,float* gpu_input,float* gpu_weight);
static void Softmax(Tensor input, Tensor output, float* gpu_input, float* gpu_output);
static void BatchNorm2d(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_running_mean, float* gpu_running_var, float* gpu_output, const float eps = 1e-5);
static void ReLU(Tensor inout, float* gpu_inout);
static void NormalizeL(Tensor input, Tensor output, float* gpu_input, float* gpu_output);

__global__ void gpu_alloc_out(float* input, float* output, int C, int H, int W);
__global__ void gpu_UnnormalizeAB(float* input, float* output, int H, int W, int C);
__global__ void gpu_Upsample(float* input, float* output, float scale_factor, int C, int H, int W, int OH, int OW);
__global__ void gpu_Softmax(float* input, float* output, int C, int H, int W);
__global__ void gpu_BatchNorm2d(float* input, float* weight, float* bias, float* running_mean, float* running_var, float* output, const float eps, int C, int H, int W);
__global__ void gpu_ReLU(float* inout, int H, int W, int C);
__global__ void gpu_Conv2d_(float* A, float* B, float* bias, float* C, int M, int K, int N, bool has_bias);
__global__ void gpu_NormalizeL(float* input, float* output, int H, int W);


// Public APIs
void ColorizerInit();
void Colorize(float* input, float* network, float* output, int N);
void ColorizerFinalize();

/*
 * Definitions
 */

void ColorizerInit() {
  /*
   * You can do input-independent jobs here.
   * e.g., Getting OpenCL Platform, allocating feature maps, ...
   * Execution time of this function is not measured, so do as much as possible!
   */
  fm_normalize_l = Make3DTensor(256, 256, 1);
  fm1_0 = Make3DTensor(256, 256, 64);
  fm1_1 = Make3DTensor(128, 128, 64);
  fm1_2 = Make3DTensor(128, 128, 64);
  fm2_0 = Make3DTensor(128, 128, 128);
  fm2_1 = Make3DTensor(64, 64, 128);
  fm2_2 = Make3DTensor(64, 64, 128);
  fm3_0 = Make3DTensor(64, 64, 256);
  fm3_1 = Make3DTensor(64, 64, 256);
  fm3_2 = Make3DTensor(32, 32, 256);
  fm3_3 = Make3DTensor(32, 32, 256);
  fm4_0 = Make3DTensor(32, 32, 512);
  fm4_1 = Make3DTensor(32, 32, 512);
  fm4_2 = Make3DTensor(32, 32, 512);
  fm4_3 = Make3DTensor(32, 32, 512);
  fm5_0 = Make3DTensor(32, 32, 512);
  fm5_1 = Make3DTensor(32, 32, 512);
  fm5_2 = Make3DTensor(32, 32, 512);
  fm5_3 = Make3DTensor(32, 32, 512);
  fm6_0 = Make3DTensor(32, 32, 512);
  fm6_1 = Make3DTensor(32, 32, 512);
  fm6_2 = Make3DTensor(32, 32, 512);
  fm6_3 = Make3DTensor(32, 32, 512);
  fm7_0 = Make3DTensor(32, 32, 512);
  fm7_1 = Make3DTensor(32, 32, 512);
  fm7_2 = Make3DTensor(32, 32, 512);
  fm7_3 = Make3DTensor(32, 32, 512);
  fm8_0 = Make3DTensor(64, 64, 256);
  fm8_1 = Make3DTensor(64, 64, 256);
  fm8_2 = Make3DTensor(64, 64, 256);
  fm8_3 = Make3DTensor(64, 64, 313);
  fm_softmax = Make3DTensor(64, 64, 313);
  fm_model_out = Make3DTensor(64, 64, 2);
  fm_upsample4 = Make3DTensor(256, 256, 2);
  pre_AB = Make3DTensor(256, 256, 2);


  CHECK_CUDA(cudaMalloc(&gpu_in, 256*256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_out, 256*256*2* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm_normalize_l, 256*256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm1_0, 256*256*64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm1_1, 128*128*64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm1_2, 128*128*64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm2_0, 128*128*128*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm2_1, 64*64*128* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm2_2, 64*64*128* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm3_0, 64*64*256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm3_1, 64*64*256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm3_2, 32*32*256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm3_3, 32*32*256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm4_0, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm4_1, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm4_2, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm4_3, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm5_0, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm5_1, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm5_2, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm5_3, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm6_0, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm6_1, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm6_2, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm6_3, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm7_0, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm7_1, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm7_2, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm7_3, 32*32*512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm8_0, 64*64*256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm8_1, 64*64*256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm8_2, 64*64*256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm8_3, 64*64*313* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm_softmax, 64*64*313* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm_model_out, 64*64*2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_fm_upsample4, 256*256*2* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_pre_AB, 256*256*2* sizeof(float)));

  CHECK_CUDA(cudaMalloc(&gpu_null, 2*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model1_0_bias, 64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model1_2_bias, 64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model1_4_bias, 64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model1_4_weight, 64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model2_0_bias, 128* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model2_2_bias, 128* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model2_4_weight, 128* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model2_4_bias, 128* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_0_bias, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_2_bias, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_4_bias, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_6_weight, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_6_bias, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_0_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_2_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_4_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_6_weight, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_6_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_0_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_2_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_4_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_6_weight, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_6_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_0_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_2_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_4_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_6_weight, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_6_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_0_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_2_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_4_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_6_weight, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_6_bias, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model8_0_bias, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model8_2_bias, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model8_4_bias, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model8_6_bias, 313* sizeof(float)));
  
  CHECK_CUDA(cudaMalloc(&gpu_model1_4_running_mean, 64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model1_4_running_var, 64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model2_4_running_mean, 128* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model2_4_running_var, 128* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_6_running_mean, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_6_running_var, 256* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_6_running_mean, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_6_running_var, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_6_running_mean, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_6_running_var, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_6_running_mean, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_6_running_var, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_6_running_mean, 512* sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_6_running_var, 512* sizeof(float)));
         
  CHECK_CUDA(cudaMalloc(&gpu_model1_0_weight, 3*3*1*64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model1_2_weight, 3*3*64*64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model2_0_weight, 3*3*64*128*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model2_2_weight, 3*3*128*128*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_0_weight, 3*3*128*256*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_2_weight, 3*3*256*256*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model3_4_weight, 3*3*256*256*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_0_weight, 3*3*256*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_2_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model4_4_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_0_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_2_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model5_4_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_0_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_2_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model6_4_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_0_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_2_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model7_4_weight, 3*3*512*512*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model8_0_weight, 4*4*512*256*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model8_2_weight, 3*3*256*256*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model8_4_weight, 3*3*256*256*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model8_6_weight, 1*1*256*313*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_model_out_weight, 1*1*313*2*sizeof(float)));


  CHECK_CUDA(cudaMalloc(&gpu_3_3_1_256_256, 3*3*1*256*256*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_3_3_64_128_128, 3*3*64*128*128*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_3_3_128_64_64, 3*3*64*64*128*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_3_3_256_64_64, 3*3*256*64*64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_3_3_256_32_32, 3*3*256*32*32*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_3_3_512_32_32, 3*3*512*32*32*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_4_4_512_64_64, 4*4*512*64*64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_1_1_256_64_64, 1*1*256*64*64*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_1_1_313_64_64, 313*64*64*sizeof(float)));

}


void Colorize(float* input, float* network, float* output, int N) {
  /*
   * !!!! CAUTION !!!!
   * Like your previous MPI homework, all inputs (input, network, output, and even N)
   * are only given to rank 0 process. You should manually:
   *   1. allocate buffers on rank >0 processes
   *   2. send inputs from rank 0 to others
   *   3. gather outputs from others to rank 0
   */
  int H = 256, W = 256;

  // Split network into parameters
  float* offset = network;
  Tensor model1_0_weight{offset, {3, 3, 1, 64}}; offset += 576;
  Tensor model1_0_bias{offset, {64}}; offset += 64;
  Tensor model1_2_weight{offset, {3, 3, 64, 64}}; offset += 36864;
  Tensor model1_2_bias{offset, {64}}; offset += 64;
  Tensor model1_4_weight{offset, {64}}; offset += 64;
  Tensor model1_4_bias{offset, {64}}; offset += 64;
  Tensor model2_0_weight{offset, {3, 3, 64, 128}}; offset += 73728;
  Tensor model2_0_bias{offset, {128}}; offset += 128;
  Tensor model2_2_weight{offset, {3, 3, 128, 128}}; offset += 147456;
  Tensor model2_2_bias{offset, {128}}; offset += 128;
  Tensor model2_4_weight{offset, {128}}; offset += 128;
  Tensor model2_4_bias{offset, {128}}; offset += 128;
  Tensor model3_0_weight{offset, {3, 3, 128, 256}}; offset += 294912;
  Tensor model3_0_bias{offset, {256}}; offset += 256;
  Tensor model3_2_weight{offset, {3, 3, 256, 256}}; offset += 589824;
  Tensor model3_2_bias{offset, {256}}; offset += 256;
  Tensor model3_4_weight{offset, {3, 3, 256, 256}}; offset += 589824;
  Tensor model3_4_bias{offset, {256}}; offset += 256;
  Tensor model3_6_weight{offset, {256}}; offset += 256;
  Tensor model3_6_bias{offset, {256}}; offset += 256;
  Tensor model4_0_weight{offset, {3, 3, 256, 512}}; offset += 1179648;
  Tensor model4_0_bias{offset, {512}}; offset += 512;
  Tensor model4_2_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model4_2_bias{offset, {512}}; offset += 512;
  Tensor model4_4_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model4_4_bias{offset, {512}}; offset += 512;
  Tensor model4_6_weight{offset, {512}}; offset += 512;
  Tensor model4_6_bias{offset, {512}}; offset += 512;
  Tensor model5_0_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model5_0_bias{offset, {512}}; offset += 512;
  Tensor model5_2_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model5_2_bias{offset, {512}}; offset += 512;
  Tensor model5_4_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model5_4_bias{offset, {512}}; offset += 512;
  Tensor model5_6_weight{offset, {512}}; offset += 512;
  Tensor model5_6_bias{offset, {512}}; offset += 512;
  Tensor model6_0_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model6_0_bias{offset, {512}}; offset += 512;
  Tensor model6_2_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model6_2_bias{offset, {512}}; offset += 512;
  Tensor model6_4_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model6_4_bias{offset, {512}}; offset += 512;
  Tensor model6_6_weight{offset, {512}}; offset += 512;
  Tensor model6_6_bias{offset, {512}}; offset += 512;
  Tensor model7_0_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model7_0_bias{offset, {512}}; offset += 512;
  Tensor model7_2_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model7_2_bias{offset, {512}}; offset += 512;
  Tensor model7_4_weight{offset, {3, 3, 512, 512}}; offset += 2359296;
  Tensor model7_4_bias{offset, {512}}; offset += 512;
  Tensor model7_6_weight{offset, {512}}; offset += 512;
  Tensor model7_6_bias{offset, {512}}; offset += 512;
  Tensor model8_0_weight{offset, {4, 4, 512, 256}}; offset += 2097152;
  Tensor model8_0_bias{offset, {256}}; offset += 256;
  Tensor model8_2_weight{offset, {3, 3, 256, 256}}; offset += 589824;
  Tensor model8_2_bias{offset, {256}}; offset += 256;
  Tensor model8_4_weight{offset, {3, 3, 256, 256}}; offset += 589824;
  Tensor model8_4_bias{offset, {256}}; offset += 256;
  Tensor model8_6_weight{offset, {1, 1, 256, 313}}; offset += 80128;
  Tensor model8_6_bias{offset, {313}}; offset += 313;
  Tensor model_out_weight{offset, {1, 1, 313, 2}}; offset += 626;
  Tensor model1_4_running_mean{offset, {64}}; offset += 64;
  Tensor model1_4_running_var{offset, {64}}; offset += 64;
  Tensor model2_4_running_mean{offset, {128}}; offset += 128;
  Tensor model2_4_running_var{offset, {128}}; offset += 128;
  Tensor model3_6_running_mean{offset, {256}}; offset += 256;
  Tensor model3_6_running_var{offset, {256}}; offset += 256;
  Tensor model4_6_running_mean{offset, {512}}; offset += 512;
  Tensor model4_6_running_var{offset, {512}}; offset += 512;
  Tensor model5_6_running_mean{offset, {512}}; offset += 512;
  Tensor model5_6_running_var{offset, {512}}; offset += 512;
  Tensor model6_6_running_mean{offset, {512}}; offset += 512;
  Tensor model6_6_running_var{offset, {512}}; offset += 512;
  Tensor model7_6_running_mean{offset, {512}}; offset += 512;
  Tensor model7_6_running_var{offset, {512}}; offset += 512;
  timer_reset(1); timer_start(1);
  change_dim(model1_0_weight);
  change_dim(model1_2_weight);
  change_dim(model1_4_weight);
  change_dim(model2_0_weight);
  change_dim(model2_2_weight);
  change_dim(model2_4_weight);
  change_dim(model3_0_weight);
  change_dim(model3_2_weight);
  change_dim(model3_4_weight);
  change_dim(model3_6_weight);
  change_dim(model4_0_weight);
  change_dim(model4_2_weight);
  change_dim(model4_4_weight);
  change_dim(model4_6_weight);
  change_dim(model5_0_weight);
  change_dim(model5_2_weight);
  change_dim(model5_4_weight);
  change_dim(model5_6_weight);
  change_dim(model6_0_weight);
  change_dim(model6_2_weight);
  change_dim(model6_4_weight);
  change_dim(model6_6_weight);
  change_dim(model7_0_weight);
  change_dim(model7_2_weight);
  change_dim(model7_4_weight);
  change_dim(model7_6_weight);
  change_dim_trans(model8_0_weight);
  change_dim(model8_2_weight);
  change_dim(model8_4_weight);
  change_dim(model8_6_weight);
  change_dim(model_out_weight);
  PRINTF_WITH_RANK("change dim done! (%f s)", timer_read(1));


  timer_reset(1); timer_start(1);
  // CHECK_CUDA(cudaMemcpy(gpu_fm_normalize_l, fm_normalize_l.buf, 256*256* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm1_0, fm1_0.buf, 256*256*64*sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm1_1, fm1_1.buf, 128*128*64*sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm1_2, fm1_2.buf, 128*128*64*sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm2_0, fm2_0.buf, 128*128*128*sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm2_1, fm2_1.buf, 64*64*128* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm2_2, fm2_2.buf, 64*64*128* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm3_0, fm3_0.buf, 64*64*256* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm3_1, fm3_1.buf, 64*64*256* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm3_2, fm3_2.buf, 32*32*256* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm3_3, fm3_3.buf, 32*32*256* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm4_0, fm4_0.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm4_1, fm4_1.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm4_2, fm4_2.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm4_3, fm4_3.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm5_0, fm5_0.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm5_1, fm5_1.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm5_2, fm5_2.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm5_3, fm5_3.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm6_0, fm6_0.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm6_1, fm6_1.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm6_2, fm6_2.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm6_3, fm6_3.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm7_0, fm7_0.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm7_1, fm7_1.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm7_2, fm7_2.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm7_3, fm7_3.buf, 32*32*512* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm8_0, fm8_0.buf, 64*64*256* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm8_1, fm8_1.buf, 64*64*256* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm8_2, fm8_2.buf, 64*64*256* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm8_3, fm8_3.buf, 64*64*313* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm_softmax, fm_softmax.buf, 64*64*313* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm_model_out, fm_model_out.buf, 64*64*2 * sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_pre_AB, pre_AB.buf, 256*256*2* sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_fm_upsample4, fm_upsample4.buf, 256*256*2* sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(gpu_model1_0_bias, model1_0_bias.buf, 64*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model1_2_bias, model1_2_bias.buf, 64*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model1_4_weight, model1_4_weight.buf, 64*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model1_4_bias, model1_4_bias.buf, 64*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model2_0_bias, model2_0_bias.buf, 128* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model2_2_bias, model2_2_bias.buf, 128* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model2_4_weight, model2_4_weight.buf, 128* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model2_4_bias, model2_4_bias.buf, 128* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_0_bias, model3_0_bias.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_2_bias, model3_2_bias.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_4_bias, model3_4_bias.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_6_weight, model3_6_weight.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_6_bias, model3_6_bias.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_0_bias, model4_0_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_2_bias, model4_2_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_4_bias, model4_4_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_6_weight, model4_6_weight.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_6_bias, model4_6_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_0_bias, model5_0_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_2_bias, model5_2_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_4_bias, model5_4_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_6_weight, model5_6_weight.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_6_bias, model5_6_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_0_bias, model6_0_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_2_bias, model6_2_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_4_bias, model6_4_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_6_weight, model6_6_weight.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_6_bias, model6_6_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_0_bias, model7_0_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_2_bias, model7_2_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_4_bias, model7_4_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_6_weight, model7_6_weight.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_6_bias, model7_6_bias.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model8_0_bias, model8_0_bias.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model8_2_bias, model8_2_bias.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model8_4_bias, model8_4_bias.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model8_6_bias, model8_6_bias.buf, 313* sizeof(float), cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMemcpy(gpu_model1_4_running_mean, model1_4_running_mean.buf, 64*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model1_4_running_var, model1_4_running_var.buf, 64*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model2_4_running_mean, model2_4_running_mean.buf, 128* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model2_4_running_var, model2_4_running_var.buf, 128* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_6_running_mean, model3_6_running_mean.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_6_running_var, model3_6_running_var.buf, 256* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_6_running_mean, model4_6_running_mean.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_6_running_var, model4_6_running_var.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_6_running_mean, model5_6_running_mean.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_6_running_var, model5_6_running_var.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_6_running_mean, model6_6_running_mean.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_6_running_var, model6_6_running_var.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_6_running_mean, model7_6_running_mean.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_6_running_var, model7_6_running_var.buf, 512* sizeof(float), cudaMemcpyHostToDevice));
         
  CHECK_CUDA(cudaMemcpy(gpu_model1_0_weight, model1_0_weight.buf, 3*3*1*64*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model1_2_weight, model1_2_weight.buf, 3*3*64*64*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model2_0_weight, model2_0_weight.buf, 3*3*64*128*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model2_2_weight, model2_2_weight.buf, 3*3*128*128*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_0_weight, model3_0_weight.buf, 3*3*128*256*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_2_weight, model3_2_weight.buf, 3*3*256*256*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model3_4_weight, model3_4_weight.buf, 3*3*256*256*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_0_weight, model4_0_weight.buf, 3*3*256*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_2_weight, model4_2_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model4_4_weight, model4_4_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_0_weight, model5_0_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_2_weight, model5_2_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model5_4_weight, model5_4_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_0_weight, model6_0_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_2_weight, model6_2_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model6_4_weight, model6_4_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_0_weight, model7_0_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_2_weight, model7_2_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model7_4_weight, model7_4_weight.buf, 3*3*512*512*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model8_0_weight, model8_0_weight.buf, 4*4*512*256*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model8_2_weight, model8_2_weight.buf, 3*3*256*256*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model8_4_weight, model8_4_weight.buf, 3*3*256*256*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model8_6_weight, model8_6_weight.buf, 1*1*256*313*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_model_out_weight, model_out_weight.buf, 1*1*313*2*sizeof(float), cudaMemcpyHostToDevice));

  PRINTF_WITH_RANK("gpu cpy done! (%f s)", timer_read(1));

  for (int i = 0; i < N; ++i) {
    // Let's process i-th image

    // Find i-th image in input buffer
    Tensor image_L{input + i * H * W, {H, W, 1}};
    CHECK_CUDA(cudaMemcpy(gpu_in, image_L.buf, W*H* sizeof(float), cudaMemcpyHostToDevice));

    // Fine location to write i-th result in output buffer
    Tensor image_AB{output + i * 2 * H * W, {2, H, W}};

    // NormalizeL
    NormalizeL(image_L, fm_normalize_l, gpu_in, gpu_fm_normalize_l);


    /*
     * Block 1
     * Comments may help you debug.
     */

    timer_reset(1); timer_start(1);

    Conv2d(fm_normalize_l, model1_0_weight, model1_0_bias, fm1_0, 1, 1, 1, gpu_fm_normalize_l, gpu_model1_0_weight, gpu_model1_0_bias, gpu_fm1_0, gpu_3_3_1_256_256);
    ReLU(fm1_0,gpu_fm1_0);
    Conv2d(fm1_0, model1_2_weight, model1_2_bias, fm1_1, 2, 1, 1, gpu_fm1_0, gpu_model1_2_weight, gpu_model1_2_bias, gpu_fm1_1, gpu_3_3_64_128_128);
    ReLU(fm1_1, gpu_fm1_1);
    BatchNorm2d(fm1_1, model1_4_weight, model1_4_bias, model1_4_running_mean, model1_4_running_var, fm1_2, gpu_fm1_1, gpu_model1_4_weight, gpu_model1_4_bias, gpu_model1_4_running_mean, gpu_model1_4_running_var, gpu_fm1_2);
    PRINTF_WITH_RANK("Block 1 done! (%f s)", timer_read(1));

    /*
     * Block 2
     */

    timer_reset(1); timer_start(1);
    
    Conv2d(fm1_2, model2_0_weight, model2_0_bias, fm2_0, 1, 1, 1, gpu_fm1_2, gpu_model2_0_weight, gpu_model2_0_bias, gpu_fm2_0, gpu_3_3_64_128_128);
    ReLU(fm2_0, gpu_fm2_0);
    Conv2d(fm2_0, model2_2_weight, model2_2_bias, fm2_1, 2, 1, 1, gpu_fm2_0, gpu_model2_2_weight, gpu_model2_2_bias, gpu_fm2_1, gpu_3_3_128_64_64);
    ReLU(fm2_1, gpu_fm2_1);
    BatchNorm2d(fm2_1, model2_4_weight, model2_4_bias, model2_4_running_mean, model2_4_running_var, fm2_2, gpu_fm2_1, gpu_model2_4_weight, gpu_model2_4_bias, gpu_model2_4_running_mean, gpu_model2_4_running_var, gpu_fm2_2);
    PRINTF_WITH_RANK("Block 2 done! (%f s)", timer_read(1));
    //DumpTensor("fm2_2.txt", fm2_2, 3);

    /*
     * Block 3
     */

    timer_reset(1); timer_start(1);
    Conv2d(fm2_2, model3_0_weight, model3_0_bias, fm3_0, 1, 1, 1, gpu_fm2_2, gpu_model3_0_weight, gpu_model3_0_bias, gpu_fm3_0, gpu_3_3_128_64_64);
    ReLU(fm3_0,gpu_fm3_0);
    Conv2d(fm3_0, model3_2_weight, model3_2_bias, fm3_1, 1, 1, 1, gpu_fm3_0, gpu_model3_2_weight, gpu_model3_2_bias, gpu_fm3_1, gpu_3_3_256_64_64);
    ReLU(fm3_1, gpu_fm3_1);
    Conv2d(fm3_1, model3_4_weight, model3_4_bias, fm3_2, 2, 1, 1, gpu_fm3_1, gpu_model3_4_weight, gpu_model3_4_bias, gpu_fm3_2, gpu_3_3_256_32_32);
    ReLU(fm3_2, gpu_fm3_2);
    BatchNorm2d(fm3_2, model3_6_weight, model3_6_bias, model3_6_running_mean, model3_6_running_var, fm3_3, gpu_fm3_2, gpu_model3_6_weight, gpu_model3_6_bias, gpu_model3_6_running_mean, gpu_model3_6_running_var, gpu_fm3_3);
    PRINTF_WITH_RANK("Block 3 done! (%f s)", timer_read(1));
    //DumpTensor("fm3_3.txt", fm3_3, 3);

    /*
     * Block 4
     */

    timer_reset(1); timer_start(1);
    Conv2d(fm3_3, model4_0_weight, model4_0_bias, fm4_0, 1, 1, 1, gpu_fm3_3, gpu_model4_0_weight, gpu_model4_0_bias, gpu_fm4_0, gpu_3_3_256_32_32);
    ReLU(fm4_0, gpu_fm4_0);
    Conv2d(fm4_0, model4_2_weight, model4_2_bias, fm4_1, 1, 1, 1, gpu_fm4_0, gpu_model4_2_weight, gpu_model4_2_bias, gpu_fm4_1, gpu_3_3_256_32_32);
    ReLU(fm4_1, gpu_fm4_1);
    Conv2d(fm4_1, model4_4_weight, model4_4_bias, fm4_2, 1, 1, 1, gpu_fm4_1, gpu_model4_4_weight, gpu_model4_4_bias, gpu_fm4_2, gpu_3_3_512_32_32);
    ReLU(fm4_2, gpu_fm4_2);
    BatchNorm2d(fm4_2, model4_6_weight, model4_6_bias, model4_6_running_mean, model4_6_running_var, fm4_3, gpu_fm4_2, gpu_model4_6_weight, gpu_model4_6_bias, gpu_model4_6_running_mean, gpu_model4_6_running_var, gpu_fm4_3);
    PRINTF_WITH_RANK("Block 4 done! (%f s)", timer_read(1));
    //DumpTensor("fm4_3.txt", fm4_3, 3);

    /*
     * Block 5
     */

    timer_reset(1); timer_start(1);
    Conv2d(fm4_3, model5_0_weight, model5_0_bias, fm5_0, 1, 2, 2, gpu_fm4_3, gpu_model5_0_weight, gpu_model5_0_bias, gpu_fm5_0, gpu_3_3_512_32_32);
    ReLU(fm5_0, gpu_fm5_0);
    Conv2d(fm5_0, model5_2_weight, model5_2_bias, fm5_1, 1, 2, 2, gpu_fm5_0, gpu_model5_2_weight, gpu_model5_2_bias, gpu_fm5_1, gpu_3_3_512_32_32);
    ReLU(fm5_1, gpu_fm5_1);
    Conv2d(fm5_1, model5_4_weight, model5_4_bias, fm5_2, 1, 2, 2, gpu_fm5_1, gpu_model5_4_weight, gpu_model5_4_bias, gpu_fm5_2, gpu_3_3_512_32_32);
    ReLU(fm5_2, gpu_fm5_2);
    BatchNorm2d(fm5_2, model5_6_weight, model5_6_bias, model5_6_running_mean, model5_6_running_var, fm5_3,  gpu_fm5_2, gpu_model5_6_weight, gpu_model5_6_bias, gpu_model5_6_running_mean, gpu_model5_6_running_var, gpu_fm5_3);
    PRINTF_WITH_RANK("Block 5 done! (%f s)", timer_read(1));
    //DumpTensor("fm5_3.txt", fm5_3, 3);

    /*
     * Block 6
     */
    timer_reset(1); timer_start(1);
    Conv2d(fm5_3, model6_0_weight, model6_0_bias, fm6_0, 1, 2, 2, gpu_fm5_3, gpu_model6_0_weight, gpu_model6_0_bias, gpu_fm6_0, gpu_3_3_512_32_32);
    ReLU(fm6_0, gpu_fm6_0);
    Conv2d(fm6_0, model6_2_weight, model6_2_bias, fm6_1, 1, 2, 2, gpu_fm6_0, gpu_model6_2_weight, gpu_model6_2_bias, gpu_fm6_1, gpu_3_3_512_32_32);
    ReLU(fm6_1, gpu_fm6_1);
    Conv2d(fm6_1, model6_4_weight, model6_4_bias, fm6_2, 1, 2, 2, gpu_fm6_1, gpu_model6_4_weight, gpu_model6_4_bias, gpu_fm6_2, gpu_3_3_512_32_32);
    ReLU(fm6_2, gpu_fm6_2);
    BatchNorm2d(fm6_2, model6_6_weight, model6_6_bias, model6_6_running_mean, model6_6_running_var, fm6_3, gpu_fm6_2, gpu_model6_6_weight, gpu_model6_6_bias, gpu_model6_6_running_mean, gpu_model6_6_running_var, gpu_fm6_3);
    PRINTF_WITH_RANK("Block 6 done! (%f s)", timer_read(1));
    //DumpTensor("fm6_3.txt", fm6_3, 3);

    /*
     * Block 7
     */
    timer_reset(1); timer_start(1);
    Conv2d(fm6_3, model7_0_weight, model7_0_bias, fm7_0, 1, 1, 1, gpu_fm6_3, gpu_model7_0_weight, gpu_model7_0_bias, gpu_fm7_0, gpu_3_3_512_32_32);
    ReLU(fm7_0, gpu_fm7_0);
    Conv2d(fm7_0, model7_2_weight, model7_2_bias, fm7_1, 1, 1, 1, gpu_fm7_0, gpu_model7_2_weight, gpu_model7_2_bias, gpu_fm7_1, gpu_3_3_512_32_32);
    ReLU(fm7_1, gpu_fm7_1);
    Conv2d(fm7_1, model7_4_weight, model7_4_bias, fm7_2, 1, 1, 1, gpu_fm7_1, gpu_model7_4_weight, gpu_model7_4_bias, gpu_fm7_2, gpu_3_3_512_32_32);
    ReLU(fm7_2, gpu_fm7_2);
    BatchNorm2d(fm7_2, model7_6_weight, model7_6_bias, model7_6_running_mean, model7_6_running_var, fm7_3, gpu_fm7_2, gpu_model7_6_weight, gpu_model7_6_bias, gpu_model7_6_running_mean, gpu_model7_6_running_var, gpu_fm7_3);
    PRINTF_WITH_RANK("Block 7 done! (%f s)", timer_read(1));

    /*
     * Block 8
     */
    timer_reset(1); timer_start(1);
    ConvTranspose2d(fm7_3, model8_0_weight, model8_0_bias, fm8_0, 2, 1, gpu_fm7_3, gpu_model8_0_weight, gpu_model8_0_bias, gpu_fm8_0, gpu_4_4_512_64_64);
    ReLU(fm8_0, gpu_fm8_0);
    Conv2d(fm8_0, model8_2_weight, model8_2_bias, fm8_1, 1, 1, 1, gpu_fm8_0, gpu_model8_2_weight, gpu_model8_2_bias, gpu_fm8_1, gpu_3_3_256_64_64);
    ReLU(fm8_1, gpu_fm8_1);
    Conv2d(fm8_1, model8_4_weight, model8_4_bias, fm8_2, 1, 1, 1, gpu_fm8_1, gpu_model8_4_weight, gpu_model8_4_bias, gpu_fm8_2, gpu_3_3_256_64_64);
    ReLU(fm8_2, gpu_fm8_2);
    Conv2d(fm8_2, model8_6_weight, model8_6_bias, fm8_3, 1, 0, 1, gpu_fm8_2, gpu_model8_6_weight, gpu_model8_6_bias, gpu_fm8_3, gpu_1_1_256_64_64);
    PRINTF_WITH_RANK("Block 8 done! (%f s)", timer_read(1));

    /*
     * Wrap-up block
     */
    timer_reset(1); timer_start(1);
    Softmax(fm8_3, fm_softmax, gpu_fm8_3, gpu_fm_softmax);
    Conv2d(fm_softmax, model_out_weight, {}, fm_model_out, 1, 0, 1, gpu_fm_softmax, gpu_model_out_weight, gpu_null, gpu_fm_model_out, gpu_1_1_313_64_64, false);
    Upsample(fm_model_out, fm_upsample4, 4, gpu_fm_model_out, gpu_fm_upsample4);
    UnnormalizeAB(fm_upsample4, pre_AB, gpu_fm_upsample4, gpu_pre_AB);
    alloc_out(pre_AB, image_AB, gpu_pre_AB, gpu_out);
    PRINTF_WITH_RANK("Block output done! (%f s)", timer_read(1));
    CHECK_CUDA(cudaMemcpy(image_AB.buf, gpu_out, W*H*2* sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void ColorizerFinalize() {
  // Free buffers we allocated in ColorizerInit
  free(fm_normalize_l.buf);
  free(fm1_0.buf);
  free(fm1_1.buf);
  free(fm1_2.buf);
  free(fm2_0.buf);
  free(fm2_1.buf);
  free(fm2_2.buf);
  free(fm3_0.buf);
  free(fm3_1.buf);
  free(fm3_2.buf);
  free(fm3_3.buf);
  free(fm4_0.buf);
  free(fm4_1.buf);
  free(fm4_2.buf);
  free(fm4_3.buf);
  free(fm5_0.buf);
  free(fm5_1.buf);
  free(fm5_2.buf);
  free(fm5_3.buf);
  free(fm6_0.buf);
  free(fm6_1.buf);
  free(fm6_2.buf);
  free(fm6_3.buf);
  free(fm7_0.buf);
  free(fm7_1.buf);
  free(fm7_2.buf);
  free(fm7_3.buf);
  free(fm8_0.buf);
  free(fm8_1.buf);
  free(fm8_2.buf);
  free(fm8_3.buf);
  free(fm_softmax.buf);
  free(fm_model_out.buf);
  free(fm_upsample4.buf);
  

  CHECK_CUDA(cudaFree(gpu_in));
  CHECK_CUDA(cudaFree(gpu_out));
  CHECK_CUDA(cudaFree(gpu_fm_normalize_l));
  CHECK_CUDA(cudaFree(gpu_fm1_0));
  CHECK_CUDA(cudaFree(gpu_fm1_1));
  CHECK_CUDA(cudaFree(gpu_fm1_2));
  CHECK_CUDA(cudaFree(gpu_fm2_0));
  CHECK_CUDA(cudaFree(gpu_fm2_1));
  CHECK_CUDA(cudaFree(gpu_fm2_2));
  CHECK_CUDA(cudaFree(gpu_fm3_0));
  CHECK_CUDA(cudaFree(gpu_fm3_1));
  CHECK_CUDA(cudaFree(gpu_fm3_2));
  CHECK_CUDA(cudaFree(gpu_fm3_3));
  CHECK_CUDA(cudaFree(gpu_fm4_0));
  CHECK_CUDA(cudaFree(gpu_fm4_1));
  CHECK_CUDA(cudaFree(gpu_fm4_2));
  CHECK_CUDA(cudaFree(gpu_fm4_3));
  CHECK_CUDA(cudaFree(gpu_fm5_0));
  CHECK_CUDA(cudaFree(gpu_fm5_1));
  CHECK_CUDA(cudaFree(gpu_fm5_2));
  CHECK_CUDA(cudaFree(gpu_fm5_3));
  CHECK_CUDA(cudaFree(gpu_fm6_0));
  CHECK_CUDA(cudaFree(gpu_fm6_1));
  CHECK_CUDA(cudaFree(gpu_fm6_2));
  CHECK_CUDA(cudaFree(gpu_fm6_3));
  CHECK_CUDA(cudaFree(gpu_fm7_0));
  CHECK_CUDA(cudaFree(gpu_fm7_1));
  CHECK_CUDA(cudaFree(gpu_fm7_2));
  CHECK_CUDA(cudaFree(gpu_fm7_3));
  CHECK_CUDA(cudaFree(gpu_fm8_0));
  CHECK_CUDA(cudaFree(gpu_fm8_1));
  CHECK_CUDA(cudaFree(gpu_fm8_2));
  CHECK_CUDA(cudaFree(gpu_fm8_3));
  CHECK_CUDA(cudaFree(gpu_fm_softmax));
  CHECK_CUDA(cudaFree(gpu_fm_model_out));
  CHECK_CUDA(cudaFree(gpu_pre_AB));
  CHECK_CUDA(cudaFree(gpu_fm_upsample4));

  CHECK_CUDA(cudaFree(gpu_null));

  CHECK_CUDA(cudaFree(gpu_model1_0_bias));
  CHECK_CUDA(cudaFree(gpu_model1_2_bias));
  CHECK_CUDA(cudaFree(gpu_model1_4_bias));
  CHECK_CUDA(cudaFree(gpu_model1_4_weight));
  CHECK_CUDA(cudaFree(gpu_model2_0_bias));
  CHECK_CUDA(cudaFree(gpu_model2_2_bias));
  CHECK_CUDA(cudaFree(gpu_model2_4_weight));
  CHECK_CUDA(cudaFree(gpu_model2_4_bias));
  CHECK_CUDA(cudaFree(gpu_model3_0_bias));
  CHECK_CUDA(cudaFree(gpu_model3_2_bias));
  CHECK_CUDA(cudaFree(gpu_model3_4_bias));
  CHECK_CUDA(cudaFree(gpu_model3_6_weight));
  CHECK_CUDA(cudaFree(gpu_model3_6_bias));
  CHECK_CUDA(cudaFree(gpu_model4_0_bias));
  CHECK_CUDA(cudaFree(gpu_model4_2_bias));
  CHECK_CUDA(cudaFree(gpu_model4_4_bias));
  CHECK_CUDA(cudaFree(gpu_model4_6_weight));
  CHECK_CUDA(cudaFree(gpu_model4_6_bias));
  CHECK_CUDA(cudaFree(gpu_model5_0_bias));
  CHECK_CUDA(cudaFree(gpu_model5_2_bias));
  CHECK_CUDA(cudaFree(gpu_model5_4_bias));
  CHECK_CUDA(cudaFree(gpu_model5_6_weight));
  CHECK_CUDA(cudaFree(gpu_model5_6_bias));
  CHECK_CUDA(cudaFree(gpu_model6_0_bias));
  CHECK_CUDA(cudaFree(gpu_model6_2_bias));
  CHECK_CUDA(cudaFree(gpu_model6_4_bias));
  CHECK_CUDA(cudaFree(gpu_model6_6_weight));
  CHECK_CUDA(cudaFree(gpu_model6_6_bias));
  CHECK_CUDA(cudaFree(gpu_model7_0_bias));
  CHECK_CUDA(cudaFree(gpu_model7_2_bias));
  CHECK_CUDA(cudaFree(gpu_model7_4_bias));
  CHECK_CUDA(cudaFree(gpu_model7_6_weight));
  CHECK_CUDA(cudaFree(gpu_model7_6_bias));
  CHECK_CUDA(cudaFree(gpu_model8_0_bias));
  CHECK_CUDA(cudaFree(gpu_model8_2_bias));
  CHECK_CUDA(cudaFree(gpu_model8_4_bias));
  CHECK_CUDA(cudaFree(gpu_model8_6_bias));
  
  CHECK_CUDA(cudaFree(gpu_model1_4_running_mean));
  CHECK_CUDA(cudaFree(gpu_model1_4_running_var));
  CHECK_CUDA(cudaFree(gpu_model2_4_running_mean));
  CHECK_CUDA(cudaFree(gpu_model2_4_running_var));
  CHECK_CUDA(cudaFree(gpu_model3_6_running_mean));
  CHECK_CUDA(cudaFree(gpu_model3_6_running_var));
  CHECK_CUDA(cudaFree(gpu_model4_6_running_mean));
  CHECK_CUDA(cudaFree(gpu_model4_6_running_var));
  CHECK_CUDA(cudaFree(gpu_model5_6_running_mean));
  CHECK_CUDA(cudaFree(gpu_model5_6_running_var));
  CHECK_CUDA(cudaFree(gpu_model6_6_running_mean));
  CHECK_CUDA(cudaFree(gpu_model6_6_running_var));
  CHECK_CUDA(cudaFree(gpu_model7_6_running_mean));
  CHECK_CUDA(cudaFree(gpu_model7_6_running_var));
         
  CHECK_CUDA(cudaFree(gpu_model1_0_weight));
  CHECK_CUDA(cudaFree(gpu_model1_2_weight));
  CHECK_CUDA(cudaFree(gpu_model2_0_weight));
  CHECK_CUDA(cudaFree(gpu_model2_2_weight));
  CHECK_CUDA(cudaFree(gpu_model3_0_weight));
  CHECK_CUDA(cudaFree(gpu_model3_2_weight));
  CHECK_CUDA(cudaFree(gpu_model3_4_weight));
  CHECK_CUDA(cudaFree(gpu_model4_0_weight));
  CHECK_CUDA(cudaFree(gpu_model4_2_weight));
  CHECK_CUDA(cudaFree(gpu_model4_4_weight));
  CHECK_CUDA(cudaFree(gpu_model5_0_weight));
  CHECK_CUDA(cudaFree(gpu_model5_2_weight));
  CHECK_CUDA(cudaFree(gpu_model5_4_weight));
  CHECK_CUDA(cudaFree(gpu_model6_0_weight));
  CHECK_CUDA(cudaFree(gpu_model6_2_weight));
  CHECK_CUDA(cudaFree(gpu_model6_4_weight));
  CHECK_CUDA(cudaFree(gpu_model7_0_weight));
  CHECK_CUDA(cudaFree(gpu_model7_2_weight));
  CHECK_CUDA(cudaFree(gpu_model7_4_weight));
  CHECK_CUDA(cudaFree(gpu_model8_0_weight));
  CHECK_CUDA(cudaFree(gpu_model8_2_weight));
  CHECK_CUDA(cudaFree(gpu_model8_4_weight));
  CHECK_CUDA(cudaFree(gpu_model8_6_weight));
  CHECK_CUDA(cudaFree(gpu_model_out_weight));


  CHECK_CUDA(cudaFree(gpu_3_3_1_256_256));
  CHECK_CUDA(cudaFree(gpu_3_3_64_128_128));
  CHECK_CUDA(cudaFree(gpu_3_3_128_64_64));
  CHECK_CUDA(cudaFree(gpu_3_3_256_64_64));
  CHECK_CUDA(cudaFree(gpu_3_3_256_32_32));
  CHECK_CUDA(cudaFree(gpu_3_3_512_32_32));
  CHECK_CUDA(cudaFree(gpu_4_4_512_64_64));
  CHECK_CUDA(cudaFree(gpu_1_1_256_64_64));
  CHECK_CUDA(cudaFree(gpu_1_1_313_64_64));


}

/*
 * Make a new 3D Tensor. Caller is responsible to free its buf.
 */
Tensor Make3DTensor(int H, int W, int C) {
  return Tensor{(float*)malloc(C * H * W * sizeof(float)), {H, W, C}};
}

// (K, C, R, S) --> (R, S, C, K)
void change_dim(Tensor input){
  int R = input.shape[0], S = input.shape[1], K = input.shape[3], C = input.shape[2];
  float* tmp = (float*)malloc(sizeof(float) * K * R * S * C);
  
  #pragma omp parallel
  {
    #pragma omp for
    for (int r=0;r<R;++r){
      for (int s=0;s<S;++s){
        for (int c=0;c<C;++c){
          for (int k=0;k<K;++k){
            tmp[r*S*K*C+s*K*C+c*K+k]= input.buf[k*R*S*C+c*R*S+r*S+s];
          }
        }
      }
    }
    #pragma omp for
    for (int r=0;r<R;++r){
      for (int s=0;s<S;++s){
        for (int c=0;c<C;++c){
          for (int k=0;k<K;++k){
            input.buf[r*S*K*C+s*K*C+c*K+k]=tmp[r*S*K*C+s*K*C+c*K+k];
          }
        }
      }
    }
  }
  free(tmp);
}

// (K, C, R, S) --> (R, S, K, C)
void change_dim_trans(Tensor input){
  int R = input.shape[0], S = input.shape[1], K = input.shape[2], C = input.shape[3];
  
  float* tmp = (float*)malloc(sizeof(float) * K * R * S * C);

  #pragma omp parallel
  {
    #pragma omp for     
    for (int r=0;r<R;++r){
      for (int s=0;s<S;++s){
        for (int k=0;k<K;++k){
          for (int c=0;c<C;++c){
            tmp[r*S*K*C+s*K*C+k*C+c]= input.buf[k*R*S*C+c*R*S+r*S+s];
          }
        }
      }
    }
    #pragma omp for
    for (int r=0;r<R;++r){
      for (int s=0;s<S;++s){
        for (int k=0;k<K;++k){
          for (int c=0;c<C;++c){
            input.buf[r*S*K*C+s*K*C+k*C+c]=tmp[r*S*K*C+s*K*C+k*C+c];
          }
        }
      }
    }
  }
  free(tmp);
}


/*
 * Dump all contents of tensor to file. May help your debugging.
 */


void DumpTensor(const char* filename, Tensor input, int dim) {
  FILE* f = fopen(filename, "w");
  if (dim == 3) {
    int C = input.shape[2], H = input.shape[0], W = input.shape[1];
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          fprintf(f, "[%d,%d,%d]=%f\n", c, h, w, input.buf[h * W * C + w*C+c]);
        }
      }
    }
  } else {
    CHECK_ERROR(false, "unexpected dimension");
  }
  fclose(f);
}



/*
 * Normalize L channel.
 * Formula: y = (x - 50) / 100
 */

__global__ void gpu_NormalizeL(float* input, float* output, int H, int W) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  if (h >= H || w >= W) return;

  output[h * W + w] = (input[h * W + w] - 50) / 100;
}

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

__global__ void gpu_Conv2d(float* input_col, float* weight, float* bias, float* output, int OO, int RSC, int K, bool has_bias) {

  int k = blockDim.x * blockIdx.x + threadIdx.x;
  int oo = blockDim.y * blockIdx.y + threadIdx.y;

  if (oo >= OO || k >= K) return;

  float s= has_bias ? bias[k] : 0;
  for (int rsc=0; rsc<RSC; rsc++){
    s+=input_col[oo*RSC+rsc]*weight[rsc*K+k];
  }
  output[oo*K+k]=s;
}

#define TILE 16
__global__ void gpu_Conv2d_(float* A, float* B, float* bias, float* C, int M, int K, int N, bool has_bias) {
  __shared__ float tileA[TILE][TILE]; 
  __shared__ float tileB[TILE][TILE]; 

  int gn = blockDim.x * blockIdx.x + threadIdx.x;
  int gm = blockDim.y * blockIdx.y + threadIdx.y;

  int ln=threadIdx.x;
  int lm=threadIdx.y;

  if(gn>=N || gm>=M) return;

  float s= has_bias ? bias[N] : 0;
  for (int k=0; k<K; K=k++){
    if(gm<M&&k+ln<K){
      tileA[lm][ln]=A[gm*K+k+ln];
    }
    else{
      tileA[lm][ln]=0;
    }
    if (k+lm<K && gn<N){
      tileB[lm][ln]=B[(k+lm)*N+gn];
    }
    else{
      tileB[lm][ln]=0;
    }
    __syncthreads();
    for (int i=0;i<TILE;++i){
      s+=tileA[lm][i]*tileB[i][ln];
    }
    __syncthreads();
  }
  if(gm<M&&gn<N){
    C[gm*N+gn]=s;
  }
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

__global__ void gpu_ReLU(float* inout, int H, int W, int C) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  for (int c = 0; c < C; ++c) {
    int idx = h * W * C + w * C + c;
    inout[idx] = inout[idx] > 0 ? inout[idx] : 0;
  }
}

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
__global__ void gpu_BatchNorm2d(float* input, float* weight, float* bias, float* running_mean, float* running_var, float* output, int C, int H, int W, const float eps) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  if (h >= H || w >= W) return;

  for (int c = 0; c < C; ++c) {
    int idx = h * W * C + w * C + c;
    output[idx] = (input[idx] - running_mean[c]) / sqrtf(running_var[c] + eps) * weight[c] + bias[c];
  }

}

void BatchNorm2d(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_running_mean, float* gpu_running_var, float* gpu_output, const float eps) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];

  CHECK_ERROR(weight.shape[0] == C && bias.shape[0] == C && running_mean.shape[0] == C && running_var.shape[0] == C, "Shape mismatch");
  CHECK_ERROR(output.shape[2] == C && output.shape[0] == H && output.shape[1] == W, "Shape mismatch");

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);

  gpu_BatchNorm2d<<<gridDim, blockDim>>>(gpu_input, gpu_weight, gpu_bias, gpu_running_mean, gpu_running_var, gpu_output, C, H, W, eps);
  
  CHECK_CUDA(cudaDeviceSynchronize());
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
    CHECK_CUDA(cudaDeviceSynchronize());
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

void Softmax(Tensor input, Tensor output, float* gpu_input, float* gpu_output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  CHECK_ERROR(output.shape[2] == C && output.shape[0] == H && output.shape[1] == W, "shape mismatch");

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);

  gpu_Softmax<<<gridDim, blockDim>>>(gpu_input, gpu_output, C, H, W);
  
  CHECK_CUDA(cudaDeviceSynchronize());

}

/*
 * Bilinear interpolation
 * input shape = (C, H, W)
 * output shape = (C, floor(H * scale_factor), floor(W * scale_factor))
 */
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
 
__global__ void gpu_UnnormalizeAB(float* input, float* output, int H, int W, int C) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  if (h >= H || w >= W) return;

  for (int c = 0; c < C; ++c) {
    output[h * W * C + w * C + c] = input[h * W * C + w * C + c] * 110;
  }
}

void UnnormalizeAB(Tensor input, Tensor output,float* gpu_input,float* gpu_output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  CHECK_ERROR(output.shape[2] == C && output.shape[0] == H && output.shape[1] == W, "shape mismatch");

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);

  gpu_UnnormalizeAB<<<gridDim, blockDim>>>(gpu_input, gpu_output, H, W, C);
  
  CHECK_CUDA(cudaDeviceSynchronize());
  
}

__global__ void gpu_alloc_out(float* input, float* output, int C, int H, int W) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  if (h >= H || w >= W) return;

  for (int c = 0; c < C; ++c) {
    output[c * H * W + h * W + w] = input[h * W * C + w * C + c];
  }
}


void alloc_out(Tensor input, Tensor output, float* gpu_input, float* gpu_output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];

  dim3 gridDim(H, W, 1);
  dim3 blockDim(1, 1, 1);
  gpu_alloc_out<<<gridDim, blockDim>>>(gpu_input, gpu_output, C, H, W);
  
  CHECK_CUDA(cudaDeviceSynchronize());

}