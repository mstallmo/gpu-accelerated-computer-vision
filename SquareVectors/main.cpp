#include <iostream>
#include "cudaOps/kernel.cuh"

#define N 5

int main() {
    float h_in[N], h_out[N];

    for (int i = 0; i < N; i++)
    {
        h_in[i] = i;
    }

    gpuSquare(h_in, h_out);

    std::cout << "Square of Number on GPU" << std::endl;
    for (int i = 0; i < N; i++)
    {
        std::cout << "The square of " << h_in[i] << " is " << h_out[i] << std::endl;
    }

    return 0;
}
