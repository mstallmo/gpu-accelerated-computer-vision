#include "cudaOps/kernel.cuh"
#include <iostream>

int main() {
    float h_in[5], h_out[5];

    for (int i = 0; i < 5; i++) {
        h_in[i] = static_cast<float>(i);
    }

    constantMemory(h_in, h_out);

    for (int i = 0; i < 5; i++) {
        std::cout << "The expression at " << i << " is " << h_out[i] << std::endl;
    }

    return 0;
}
