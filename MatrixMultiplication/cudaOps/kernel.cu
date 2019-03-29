#include "kernel.cuh"
#include <iostream>
#include <math.h>

#define TILE_SIZE 2

__global__ void matrixMulGpuNonShared(float *d_a, float *d_b, float *d_c,
                                      const int size) {
  int row, col;
  col = TILE_SIZE * blockIdx.x + threadIdx.x;
  row = TILE_SIZE * blockIdx.y + threadIdx.y;

  for (int k = 0; k < size; k++) {
    d_c[row * size + col] += d_a[row * size + k] * d_b[row * size + col];
  }
}

__global__ void matrixMulGpu(float *d_a, float *d_b, float *d_c,
                             const int size) {
  int row, col;

  __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

  col = TILE_SIZE * blockIdx.x + threadIdx.x;
  row = TILE_SIZE * blockIdx.y + threadIdx.y;

  for (int i = 0; i < size / TILE_SIZE; i++) {
    shared_a[threadIdx.y][threadIdx.x] =
        d_a[row * size + (i * TILE_SIZE + threadIdx.x)];
    shared_b[threadIdx.y][threadIdx.x] =
        d_b[(i * TILE_SIZE + threadIdx.y) * size + col];
    d_c[row * size + col] += d_a[row * size + i] * d_b[i * size + col];
  }

  __syncthreads();

  for (int j = 0; j < TILE_SIZE; j++) {
    d_c[row * size + col] +=
        shared_a[threadIdx.x][j] * shared_b[j][threadIdx.y];
  }

  __syncthreads();
}

void runMatrixMul() {
  const int size = 6;

  float h_a[size][size], h_b[size][size], h_result[size][size];
  float *d_a, *d_b, *d_result;

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      h_a[i][j] = i;
      h_b[i][j] = j;
    }
  }

  cudaMalloc((void **)&d_a, size * size * sizeof(float));
  cudaMalloc((void **)&d_b, size * size * sizeof(float));
  cudaMalloc((void **)&d_result, size * size * sizeof(float));

  cudaMemcpy(d_a, h_a, size * size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size * size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid(size / TILE_SIZE, size / TILE_SIZE, 1);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

  // matrixMulGpu<<<dimGrid, dimBlock>>>(d_a, d_b, d_result, size);
  matrixMulGpuNonShared<<<dimGrid, dimBlock>>>(d_a, d_b, d_result, size);

  cudaMemcpy(h_result, d_result, size * size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);

  std::cout << "The result of the Matrix multiplication is: " << std::endl;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      std::cout << h_result[i][j] << " ";
    }
    std::cout << std::endl;
  }
}
