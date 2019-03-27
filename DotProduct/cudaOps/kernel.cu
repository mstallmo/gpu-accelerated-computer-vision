#include "kernel.cuh"
#include <iostream>

__global__ void dotGpu(float *d_a, float *d_b, float *d_c) {
  __shared__ float partial_sum[1024];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int index = threadIdx.x;

  float sum = 0;
  while (tid < 1024) {
    sum += d_a[tid] * d_b[tid];
    tid += blockDim.x * gridDim.x;
  }

  partial_sum[index] = sum;

  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (index < i) {
      partial_sum[index] += partial_sum[index + i];
    }
    __syncthreads();
    i /= 2;
  }
  if (index == 0)
    d_c[blockIdx.x] = partial_sum[0];
}

void runDotGpu() {
  float *h_a, *h_b, h_c, *partial_sum;
  float *d_a, *d_b, *d_partial_sum;

  int block_calc = (1024 + 1024 - 1) / 1024;
  int blocksPerGrid = (32 < block_calc ? 32 : block_calc);

  h_a = (float *)malloc(1024 * sizeof(float));
  h_b = (float *)malloc(1024 * sizeof(float));
  partial_sum = (float *)malloc(blocksPerGrid * sizeof(float));

  cudaMalloc((void **)&d_a, 1024 * sizeof(float));
  cudaMalloc((void **)&d_b, 1024 * sizeof(float));
  cudaMalloc((void **)&d_partial_sum, blocksPerGrid * sizeof(float));

  for (int i = 0; i < 1024; i++) {
    h_a[i] = i;
    h_b[i] = 2;
  }

  cudaMemcpy(d_a, h_a, 1024 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, 1024 * sizeof(float), cudaMemcpyHostToDevice);

  dotGpu<<<blocksPerGrid, 1024>>>(d_a, d_b, d_partial_sum);

  cudaMemcpy(partial_sum, d_partial_sum, blocksPerGrid * sizeof(float),
             cudaMemcpyDeviceToHost);

  h_c = 0;
  for (int i = 0; i < blocksPerGrid; i++) {
    h_c += partial_sum[i];
  }

  std::cout << "The computed dot product is " << h_c << std::endl;

#define cpu_sum(x) (x * (x + 1))
  if (h_c == cpu_sum((float)(1024 - 1))) {
    std::cout << "The dot product computed by the gpu is correct" << std::endl;
  } else {
    std::cout << "There was an error in doct product computation" << std::endl;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_partial_sum);
  free(h_a);
  free(h_b);
  free(partial_sum);
}
