



struct Tensor {
  // Pointer to data
  float* buf;
  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape  = {2, 3}
  int shape[4];
};

void gpu_malloc(float** adr, int size);

void gpu_free(float** adr);

void gpu_memcpy(float** in, float** out, int size, bool to_gpu);

void Conv2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_output, float* gpu_input_col, bool has_bias = true);
void ConvTranspose2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_output, float* gpu_input_col);

void alloc_out(Tensor input, Tensor output, float* gpu_input, float* gpu_output);
void Upsample(Tensor input, Tensor output, float scale_factor, float* gpu_input, float* gpu_output);
void UnnormalizeAB(Tensor input, Tensor output,float* gpu_input,float* gpu_weight);
void Softmax(Tensor input, Tensor output, float* gpu_input, float* gpu_output);
void BatchNorm2d(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, float* gpu_input, float* gpu_weight, float* gpu_bias, float* gpu_running_mean, float* gpu_running_var, float* gpu_output, const float eps = 1e-5);
void ReLU(Tensor inout, float* gpu_inout);
void NormalizeL(Tensor input, Tensor output, float* gpu_input, float* gpu_output);
