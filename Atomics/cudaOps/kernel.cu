#include <iostream>
#include "kernel.cuh"

#define NUM_THREADS 10000
#define SIZE 10
#define BLOCK_WIDTH 100

__global__ void atomicAddGpu(int *d_a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  tid = tid % SIZE;
  atomicAdd(&d_a[tid], 1);
}

void cudaAtomicAdd() {
  int h_a[SIZE];
  const int ARRAY_BYTES = SIZE * sizeof(int);

  int *d_a;
  cudaMalloc((void **)&d_a, ARRAY_BYTES);

  cudaMemset((void *)d_a, 0, ARRAY_BYTES);

  atomicAddGpu<<<NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_a);

  cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  cudaFree(d_a);

  std::cout << "Number of times a particular Array index has been incremented is " << std::endl;
  for (int i = 0; i < SIZE; i++) {
      std::cout << "index: " << i << " --> " << h_a[i] << " times" << std::endl;
  }
}
