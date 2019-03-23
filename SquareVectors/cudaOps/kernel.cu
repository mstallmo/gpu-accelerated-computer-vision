#include "kernel.cuh"

#define N 5

__global__ void gpuSquareKernel(float* d_in, float* d_out)
{
    int tid = threadIdx.x;
    float temp = d_in[tid];
    d_out[tid] = temp * temp;
}

void gpuSquare(float* h_in, float* h_out)
{
    float *d_in, *d_out;

    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    gpuSquareKernel << <1, N >> > (d_in, d_out);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_out);
}
