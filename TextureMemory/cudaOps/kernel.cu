#include "kernel.cuh"
#include <iostream>

#define NUM_THREADS 10
#define N 10

texture<float, 1, cudaReadModeElementType> textureRef;

__global__ void textureMemoryGpu(int n, float *d_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float temp = tex1D(textureRef, float(idx));
    d_out[idx] = temp;
  }
}

void runTextureMemoryGpu() {
  int num_blocks = N / NUM_THREADS + ((N % NUM_THREADS) ? 1 : 0);

  float *d_out;
  cudaMalloc((void **)&d_out, N * sizeof(float));
  float *h_out = (float *)malloc(sizeof(float) * N);
  float h_in[N];

  for (int i = 0; i < N; i++) {
    h_in[i] = float(i);
  }

  cudaArray *cu_Array;
  cudaMallocArray(&cu_Array, &textureRef.channelDesc, N, 1);
  cudaMemcpyToArray(cu_Array, 0, 0, h_in, sizeof(float) * N,
                    cudaMemcpyHostToDevice);

  cudaBindTextureToArray(textureRef, cu_Array);

  textureMemoryGpu<<<num_blocks, NUM_THREADS>>>(N, d_out);

  cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  std::cout << "Use of Texture memory on GPU" << std::endl;
  for (int i = 0; i < N; i++) {
      std::cout << "Average between the two nearest element is " << h_out[i] << std::endl;
  }

  free(h_out);
  cudaFree(d_out);
  cudaFreeArray(cu_Array);
  cudaUnbindTexture(textureRef);
}
