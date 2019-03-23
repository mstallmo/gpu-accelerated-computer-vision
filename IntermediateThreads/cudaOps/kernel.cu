#include "kernel.cuh"

#define N 5000

__global__ void gpuAddKernel(int *d_a, int *d_b, int *d_c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < N) {
    d_c[tid] = d_a[tid] + d_b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

void gpuAdd(int *h_a, int *h_b, int *h_c) {

  int *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, N * sizeof(int));
  cudaMalloc((void **)&d_b, N * sizeof(int));
  cudaMalloc((void **)&d_c, N * sizeof(int));

  cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

  gpuAddKernel<<<512, 1024>>>(d_a, d_b, d_c);

  cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
