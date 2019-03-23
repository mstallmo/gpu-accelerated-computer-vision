#include "kernel.cuh"

#define N 100

__global__ void gpuAddKernel(int const* const d_a, int const* const d_b, int* d_c)
{
    int tid = blockIdx.x;
    if (tid < N)
        d_c[tid] = d_a[tid] + d_b[tid];
}

void gpuAdd(int const * const h_a, int const * const h_b, int* h_c)
{
    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    clock_t start_d = clock();

    gpuAddKernel << <N, 1 >> > (d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

