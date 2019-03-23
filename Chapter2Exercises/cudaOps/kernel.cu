#include "kernel.cuh"
#include <stdio.h>

__global__ void subtractKernel(int d_a, int d_b, int *d_c) { *d_c = d_a - d_b; }

__global__ void multiplyKernel(int *d_a, int *d_b, int *d_c) {
  *d_c = (*d_a) * (*d_b);
}

__global__ void parallelKernel() {
  int bdx = blockIdx.x;
  int tdx = threadIdx.x;

  printf("Block %d; Thread %d\n", bdx, tdx);
}

__global__ void cubeVectorKernel(int *d_in, int *d_out) {
  int tdx = threadIdx.x;
  int temp = d_in[tdx];
  d_out[tdx] = temp * temp * temp;
}

int subtractGpu(int h_a, int h_b) {
  int h_c;
  int *d_c;
  cudaMalloc((void **)&d_c, sizeof(int));

  subtractKernel<<<1, 1>>>(h_a, h_b, d_c);

  cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_c);

  return h_c;
}

int multiplyGpu(int h_a, int h_b) {
  int h_c;
  int *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, sizeof(int));
  cudaMalloc((void **)&d_b, sizeof(int));
  cudaMalloc((void **)&d_c, sizeof(int));

  cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

  multiplyKernel<<<1, 1>>>(d_a, d_b, d_c);

  cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return h_c;
}

void parallel1() {
  printf("10 blocks, 500 threads\n");
  parallelKernel<<<10, 500>>>();
}

void parallel2() {
  printf("5 blocks, 1000 threads\n");
  parallelKernel<<<5, 1000>>>();
}

void parallel3() {
  printf("1000 blocks, 5 threads\n");
  parallelKernel<<<1000, 5>>>();
}

cudaDeviceProp getDeviceVersion() {
  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, 0);
  return device_props;
}

void gpuCube(int *h_in, int *h_out) {
  int *d_in, *d_out;

  cudaMalloc((void **)&d_in, 50 * sizeof(int));
  cudaMalloc((void **)&d_out, 50 * sizeof(int));

  cudaMemcpy(d_in, h_in, 50 * sizeof(int), cudaMemcpyHostToDevice);

  cubeVectorKernel<<<1, 50>>>(d_in, d_out);

  cudaMemcpy(h_out, d_out, 50 * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}
