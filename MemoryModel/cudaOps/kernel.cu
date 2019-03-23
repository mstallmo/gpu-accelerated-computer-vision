#include "kernel.cuh"
#include <iostream>
#include <stdio.h>

#define N 5

__global__ void gpuGlobalMemory(int *d_a) { d_a[threadIdx.x] = threadIdx.x; }

__global__ void gpuLocalMemory(int d_in) {
  int t_local;
  t_local = d_in * threadIdx.x;
  printf("The value of Local variable in current thread is: %d\n", t_local);
}

__global__ void gpuSharedMemory(float *d_a) {
  int i, index = threadIdx.x;
  float average, sum = 0.0f;

  __shared__ float sh_arr[10];

  sh_arr[index] = d_a[index];

  __syncthreads();
  for (i = 0; i <= index; i++) {
    sum += sh_arr[i];
  }

  average = sum / (index + 1.0f);
  d_a[index] = average;
  sh_arr[index] = average;
}

void printGlobalMemory() {
  int h_a[N];
  int *d_a;

  cudaMalloc((void **)&d_a, N * sizeof(int));
  cudaMemcpy((void *)d_a, (void *)h_a, N * sizeof(int), cudaMemcpyHostToDevice);

  gpuGlobalMemory<<<1, N>>>(d_a);

  cudaMemcpy((void *)h_a, (void *)d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "Array in Global Memory is: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "At Index: " << i << " --> " << h_a[i] << std::endl;
  }
}

void printLocalMemory() {
  std::cout << "Use of local memory on GPU:" << std::endl;
  gpuLocalMemory<<<1, N>>>(5);
  cudaDeviceSynchronize();
}

void printSharedMemory() {
  float h_a[10];
  float *d_a;

  for (int i = 0; i < 10; i++) {
    h_a[i] = i;
  }

  cudaMalloc((void **)&d_a, sizeof(float) * 10);
  cudaMemcpy((void *)d_a, (void *)h_a, sizeof(float) * 10,
             cudaMemcpyHostToDevice);

  gpuSharedMemory<<<1, 10>>>(d_a);

  cudaMemcpy((void *)h_a, (void *)d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
  
  std::cout << "Use of Shared memory on GPU: " << std::endl;
  for (int i = 0; i < 10; i++) {
      std::cout << "The running average after " << i << " element is " << h_a[i] << std::endl;
  }
}
