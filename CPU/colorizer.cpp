

#include "colorizer.h"
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"
#include "timer.h"



/*
 * Declarations
 */

struct Tensor {
  // Pointer to data
  float* buf;
  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
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

// Layers
static Tensor Make3DTensor(int H, int W, int C);
static void DumpTensor(const char* filename, Tensor input, int dim);
static void NormalizeL(Tensor input, Tensor output);
static void Conv2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, bool has_bias = true);
static void ReLU(Tensor inout);
static void BatchNorm2d(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, const float eps = 1e-5);
static void ConvTranspose2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad);
static void Softmax(Tensor input, Tensor output);
static void Upsample(Tensor input, Tensor output, float scale_factor);
static void UnnormalizeAB(Tensor input, Tensor output);
static void change_dim(Tensor input);
static void change_dim_trans(Tensor input);
static void alloc_out(Tensor input, Tensor output);

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
  //timer_reset(1); timer_start(1);
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
  //PRINTF_WITH_RANK("change dim done! (%f s)", timer_read(1));

  for (int i = 0; i < N; ++i) {
    // Let's process i-th image

    // Find i-th image in input buffer
    Tensor image_L{input + i * H * W, {H, W, 1}};

    // Fine location to write i-th result in output buffer
    Tensor image_AB{output + i * 2 * H * W, {2, H, W}};
    

    // NormalizeL
    NormalizeL(image_L, fm_normalize_l);
    /*
     * Block 1
     * Comments may help you debug.
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fm_normalize_l, model1_0_weight, model1_0_bias, fm1_0, 1, 1, 1);
    ReLU(fm1_0);
    Conv2d(fm1_0, model1_2_weight, model1_2_bias, fm1_1, 2, 1, 1);
    ReLU(fm1_1);
    BatchNorm2d(fm1_1, model1_4_weight, model1_4_bias, model1_4_running_mean, model1_4_running_var, fm1_2);
    //PRINTF_WITH_RANK("Block 1 done! (%f s)", timer_read(1));
    //DumpTensor("fm1_2.txt", fm1_2, 3);

    /*
     * Block 2
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fm1_2, model2_0_weight, model2_0_bias, fm2_0, 1, 1, 1);
    ReLU(fm2_0);
    Conv2d(fm2_0, model2_2_weight, model2_2_bias, fm2_1, 2, 1, 1);
    ReLU(fm2_1);
    BatchNorm2d(fm2_1, model2_4_weight, model2_4_bias, model2_4_running_mean, model2_4_running_var, fm2_2);
    //PRINTF_WITH_RANK("Block 2 done! (%f s)", timer_read(1));
    //DumpTensor("fm2_2.txt", fm2_2, 3);

    /*
     * Block 3
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fm2_2, model3_0_weight, model3_0_bias, fm3_0, 1, 1, 1);
    ReLU(fm3_0);
    Conv2d(fm3_0, model3_2_weight, model3_2_bias, fm3_1, 1, 1, 1);
    ReLU(fm3_1);
    Conv2d(fm3_1, model3_4_weight, model3_4_bias, fm3_2, 2, 1, 1);
    ReLU(fm3_2);
    BatchNorm2d(fm3_2, model3_6_weight, model3_6_bias, model3_6_running_mean, model3_6_running_var, fm3_3);
    //PRINTF_WITH_RANK("Block 3 done! (%f s)", timer_read(1));
    //DumpTensor("fm3_3.txt", fm3_3, 3);

    /*
     * Block 4
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fm3_3, model4_0_weight, model4_0_bias, fm4_0, 1, 1, 1);
    ReLU(fm4_0);
    Conv2d(fm4_0, model4_2_weight, model4_2_bias, fm4_1, 1, 1, 1);
    ReLU(fm4_1);
    Conv2d(fm4_1, model4_4_weight, model4_4_bias, fm4_2, 1, 1, 1);
    ReLU(fm4_2);
    BatchNorm2d(fm4_2, model4_6_weight, model4_6_bias, model4_6_running_mean, model4_6_running_var, fm4_3);
    //PRINTF_WITH_RANK("Block 4 done! (%f s)", timer_read(1));
    //DumpTensor("fm4_3.txt", fm4_3, 3);

    /*
     * Block 5
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fm4_3, model5_0_weight, model5_0_bias, fm5_0, 1, 2, 2);
    ReLU(fm5_0);
    Conv2d(fm5_0, model5_2_weight, model5_2_bias, fm5_1, 1, 2, 2);
    ReLU(fm5_1);
    Conv2d(fm5_1, model5_4_weight, model5_4_bias, fm5_2, 1, 2, 2);
    ReLU(fm5_2);
    BatchNorm2d(fm5_2, model5_6_weight, model5_6_bias, model5_6_running_mean, model5_6_running_var, fm5_3);
    //PRINTF_WITH_RANK("Block 5 done! (%f s)", timer_read(1));
    //DumpTensor("fm5_3.txt", fm5_3, 3);

    /*
     * Block 6
     */
    //timer_reset(1); timer_start(1);
    Conv2d(fm5_3, model6_0_weight, model6_0_bias, fm6_0, 1, 2, 2);
    ReLU(fm6_0);
    Conv2d(fm6_0, model6_2_weight, model6_2_bias, fm6_1, 1, 2, 2);
    ReLU(fm6_1);
    Conv2d(fm6_1, model6_4_weight, model6_4_bias, fm6_2, 1, 2, 2);
    ReLU(fm6_2);
    BatchNorm2d(fm6_2, model6_6_weight, model6_6_bias, model6_6_running_mean, model6_6_running_var, fm6_3);
    //PRINTF_WITH_RANK("Block 6 done! (%f s)", timer_read(1));
    //DumpTensor("fm6_3.txt", fm6_3, 3);

    /*
     * Block 7
     */
    //timer_reset(1); timer_start(1);
    Conv2d(fm6_3, model7_0_weight, model7_0_bias, fm7_0, 1, 1, 1);
    ReLU(fm7_0);
    Conv2d(fm7_0, model7_2_weight, model7_2_bias, fm7_1, 1, 1, 1);
    ReLU(fm7_1);
    Conv2d(fm7_1, model7_4_weight, model7_4_bias, fm7_2, 1, 1, 1);
    ReLU(fm7_2);
    BatchNorm2d(fm7_2, model7_6_weight, model7_6_bias, model7_6_running_mean, model7_6_running_var, fm7_3);
    //PRINTF_WITH_RANK("Block 7 done! (%f s)", timer_read(1));
    //DumpTensor("fm7_3.txt", fm7_3, 3);

    /*
     * Block 8
     */
    //timer_reset(1); timer_start(1);
    ConvTranspose2d(fm7_3, model8_0_weight, model8_0_bias, fm8_0, 2, 1);
    ReLU(fm8_0);
    Conv2d(fm8_0, model8_2_weight, model8_2_bias, fm8_1, 1, 1, 1);
    ReLU(fm8_1);
    Conv2d(fm8_1, model8_4_weight, model8_4_bias, fm8_2, 1, 1, 1);
    ReLU(fm8_2);
    Conv2d(fm8_2, model8_6_weight, model8_6_bias, fm8_3, 1, 0, 1);
    //PRINTF_WITH_RANK("Block 8 done! (%f s)", timer_read(1));
    //DumpTensor("fm8_3.txt", fm8_3, 3);

    /*
     * Wrap-up block
     */
    //timer_reset(1); timer_start(1);
    Softmax(fm8_3, fm_softmax);
    Conv2d(fm_softmax, model_out_weight, {}, fm_model_out, 1, 0, 1, false);
    Upsample(fm_model_out, fm_upsample4, 4);
    UnnormalizeAB(fm_upsample4, pre_AB);
    alloc_out(pre_AB, image_AB);
    //PRINTF_WITH_RANK("Block output done! (%f s)", timer_read(1));
    //DumpTensor("image_AB.txt", pre_AB, 3);
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
void NormalizeL(Tensor input, Tensor output) {
  int H = input.shape[0], W = input.shape[1];
  CHECK_ERROR(input.shape[2] == 1 && output.shape[2] == 1 && output.shape[0] == H && output.shape[1] == W, "Size mismatch");
  #pragma omp parallel
  {
    #pragma omp for
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        output.buf[h * W + w] = (input.buf[h * W + w] - 50) / 100;
      }
    }
  }
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

void Conv2d_(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, bool has_bias) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  int K = weight.shape[3], R = weight.shape[0], S = weight.shape[1];
  int OH = output.shape[0], OW = output.shape[1];
  CHECK_ERROR(OH == (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1, "Output height mismatch");
  CHECK_ERROR(OW == (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1, "Output width mismatch");
  CHECK_ERROR(weight.shape[2] == C && (!has_bias || bias.shape[0] == K) && output.shape[2] == K, "Channel size mismatch");

  #pragma omp parallel
  { 
    #pragma omp for collapse(2)
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        for (int k =0 ; k<K; ++k){
          output.buf[oh * OW * K + ow * K + k]= has_bias ? bias.buf[k] : 0;
        }
        
        for (int r = 0; r < R; ++r) {
          int h = oh * stride - pad + dilation*r;
          for (int s = 0; s < S; ++s) {
            int w = ow * stride - pad + dilation*s;
            for (int c = 0; c < C; ++c) {
              if (h < 0 || h >= H || w < 0 || w >= W) continue;
              float i = input.buf[h * C * W + w * C + c];
              for (int k =0 ; k<K; ++k){
                float f = weight.buf[r * S * K * C  + s * K * C + c * K +k]; 
                output.buf[oh * OW * K + ow * K + k] += i * f;
              }
            }
          }
        }
      }
    } 
  }    
}

void Conv2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, bool has_bias) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  int K = weight.shape[3], R = weight.shape[0], S = weight.shape[1];
  int OH = output.shape[0], OW = output.shape[1];
  CHECK_ERROR(OH == (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1, "Output height mismatch");
  CHECK_ERROR(OW == (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1, "Output width mismatch");
  CHECK_ERROR(weight.shape[2] == C && (!has_bias || bias.shape[0] == K) && output.shape[2] == K, "Channel size mismatch");
  float* input_col=(float*)malloc(R*S*C*OH*OW*sizeof(float));
  #pragma omp parallel
  { 
    #pragma omp for collapse(2)
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        for (int k =0 ; k<K; ++k){
          output.buf[oh * OW * K + ow * K + k]= has_bias ? bias.buf[k] : 0;
        }
        for (int r = 0; r < R; ++r) {
          int h = oh * stride - pad + dilation*r;
          for (int s = 0; s < S; ++s) {
            int w = ow * stride - pad + dilation*s;
            for (int c = 0; c < C; ++c) {
              if (h < 0 || h >= H || w < 0 || w >= W) input_col[oh*OW*R*S*C + ow*R*S*C+ r*S*C+ s*C + c] = 0;
              else input_col[oh*OW*R*S*C + ow*R*S*C+ r*S*C+ s*C + c] = input.buf[h * C * W + w * C + c];
            }
          }
        }
      }
    } 

    #pragma omp for
    for (int oo=0; oo<OH*OW; oo+=4){
      for (int rsc=0;rsc<R*S*C;rsc++){
        float a0=input_col[oo*R*S*C+rsc];
        float a1=input_col[(oo+1)*R*S*C+rsc];
        float a2=input_col[(oo+2)*R*S*C+rsc];
        float a3=input_col[(oo+3)*R*S*C+rsc];
        for (int k=0;k<K;k++){
          float b=weight.buf[rsc*K+k];
          output.buf[oo*K+k]+=a0*b;
          output.buf[(oo+1)*K+k]+=a1*b;
          output.buf[(oo+2)*K+k]+=a2*b;
          output.buf[(oo+3)*K+k]+=a3*b;
        }
      }
    }

    // #pragma omp for
    // for (int oo=0; oo<OH*OW; oo++){
    //   for (int k=0;k<K;k++){
    //     output.buf[oo*K+k];
    //   }
    //   for (int rsc=0;rsc<R*S*C;rsc++){
    //     for (int k=0;k<K;k++){
    //       output.buf[oo*K+k]+=input_col[oo*R*S*C+rsc]*weight.buf[rsc*K+k];
    //     }
    //   }
    // }

  }
  free(input_col);
}

/*
 * ReLU
 * Formula: y = max(x, 0)
 */
void ReLU(Tensor inout) {
  int C = inout.shape[2], H = inout.shape[0], W = inout.shape[1];
  #pragma omp parallel
  {
    #pragma omp for
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
          int idx = h * W * C + w * C + c;
          inout.buf[idx] = inout.buf[idx] > 0 ? inout.buf[idx] : 0;
        }
      }
    }
  }
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
void BatchNorm2d(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, const float eps) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];

  CHECK_ERROR(weight.shape[0] == C && bias.shape[0] == C && running_mean.shape[0] == C && running_var.shape[0] == C, "Shape mismatch");
  CHECK_ERROR(output.shape[2] == C && output.shape[0] == H && output.shape[1] == W, "Shape mismatch");
  #pragma omp parallel
  {
    #pragma omp for
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
          int idx = h * W * C + w * C + c;
          output.buf[idx] = (input.buf[idx] - running_mean.buf[c]) / sqrtf(running_var.buf[c] + eps) * weight.buf[c] + bias.buf[c];
        }
      }
    }
  }
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
void ConvTranspose2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  int K = weight.shape[3], R = weight.shape[0], S = weight.shape[1];
  int OH = output.shape[0], OW = output.shape[1];

  CHECK_ERROR(OH == (H - 1) * stride - 2 * pad + R, "Output height mismatch");
  CHECK_ERROR(OW == (W - 1) * stride - 2 * pad + S, "Output width mismatch");
  CHECK_ERROR(weight.shape[2] == C && bias.shape[0] == K && output.shape[2] == K, "Channel size mismatch");

  #pragma omp parallel
  {
    #pragma omp for collapse(2)
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        for (int k =0; k < K; ++k){
          output.buf[oh * OW * K + ow * K + k]= bias.buf[k];
        }


        for (int r = 0; r < R; ++r) {
          int h = (oh + pad - r) / stride;
          if ((oh + pad - r) % stride != 0) continue;
          for (int s = 0; s < S; ++s) {
            if ((ow + pad - s) % stride != 0) continue;
            int w = (ow + pad - s) / stride;
            if (h < 0 || h >= H || w < 0 || w >= W) continue;
            for (int c = 0; c < C; ++c) {
              float i = input.buf[h * W * C + w * C + c];
              for (int k = 0; k < K; ++k) {
                float f = weight.buf[r * S * C * K + s * C * K + c*K + k];
                output.buf[oh * OW * K + ow * K + k]+=i*f;
              }
            }
          }
        }
      }
    }
  }
}

/*
 * Softmax
 * Formula: y = e^x / sum(e^x)
 */
void Softmax(Tensor input, Tensor output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  CHECK_ERROR(output.shape[2] == C && output.shape[0] == H && output.shape[1] == W, "shape mismatch");
  #pragma omp parallel
  {
    #pragma omp for
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        float sum = 0;
        for (int c = 0; c < C; ++c) {
          sum += expf(input.buf[h * W * C + w * C + c]);
        }
        for (int c = 0; c < C; ++c) {
          output.buf[h * W * C + w * C + c] = expf(input.buf[h * W * C + w * C + c]) / sum;
        }
      }
    }
  }
}

/*
 * Bilinear interpolation
 * input shape = (C, H, W)
 * output shape = (C, floor(H * scale_factor), floor(W * scale_factor))
 */
void Upsample(Tensor input, Tensor output, float scale_factor) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  int OH = output.shape[0], OW = output.shape[1];
  CHECK_ERROR(output.shape[2] == C && OH == floorf(H * scale_factor) && OW == floorf(W * scale_factor), "shape mismatch");

  #pragma omp parallel
  {
    #pragma omp for
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
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
          output.buf[oh * OW * C + ow * C + c] = w00 * input.buf[h0 * W * C+ w0 * C + c]
                                                + w01 * input.buf[h0 * W * C+ w1 * C + c]
                                                + w10 * input.buf[h1 * W * C+ w0 * C + c]
                                                + w11 * input.buf[h1 * W * C+ w1 * C + c];
        }
      }
    }
  }

}

/*
 * Unnormalize A and B channel
 * Formula: y = x * 110
 */
void UnnormalizeAB(Tensor input, Tensor output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  CHECK_ERROR(output.shape[2] == C && output.shape[0] == H && output.shape[1] == W, "shape mismatch");

    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
          output.buf[h * W * C + w * C + c] = input.buf[h * W * C + w * C + c] * 110;
        }
      }
    }
  
}

void alloc_out(Tensor input, Tensor output) {
  int C = input.shape[2], H = input.shape[0], W = input.shape[1];
  #pragma omp parallel
  {
    #pragma omp for
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          output.buf[c * H * W + h * W + w] = input.buf[h * W * C + w * C + c];
        }
      }
    }
  }
}
