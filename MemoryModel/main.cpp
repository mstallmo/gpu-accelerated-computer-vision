#include "cudaOps/kernel.cuh"

int main() {
    printGlobalMemory();

    printLocalMemory();

    printSharedMemory();

    return 0;
}
