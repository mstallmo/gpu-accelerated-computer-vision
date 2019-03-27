#include "kernel.cuh"

__constant__ int constant_f;
__constant__ int constant_g;
#define N 5

__global__ void constantMemoryGpu(float *d_in, float *d_out) {
  int tid = threadIdx.x;
  d_out[tid] = constant_f * d_in[tid] + constant_g;
}

void constantMemory(float *h_in, float *h_out) {
  float *d_in, *d_out;

  int h_f = 2;
  int h_g = 20;

  cudaMalloc((void **)&d_in, sizeof(float) * N);
  cudaMalloc((void **)&d_out, sizeof(float) * N);

  cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(constant_f, &h_f, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(constant_g, &h_g, sizeof(int), 0, cudaMemcpyHostToDevice);

  constantMemoryGpu<<<1, N>>>(d_in, d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(d_in);
  cudaFree(d_out);
}
