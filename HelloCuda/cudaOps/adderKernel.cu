#include "adderKernel.cuh"

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c) {
    *d_c = *d_a + *d_b;
}

int gpuAdd(int const a, int const b) {
    int h_c;
    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    gpuAdd << <1, 1 >> > (d_a, d_b, d_c);

    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return h_c;
}
