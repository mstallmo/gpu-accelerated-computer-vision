#include "Adder.h"
#include "cudaOps/adderKernel.cuh"

Adder::Adder(int const firstValue, int const secondValue):
    firstValue(firstValue),
    secondValue(secondValue)
{
}

Adder::~Adder()
{
}

int const Adder::add() 
{
    return gpuAdd(this->firstValue, this->secondValue);
}
