#include "cudaOps/kernel.cuh"
#include <iostream>

bool isVersionHighEnough() {
  cudaDeviceProp version = getDeviceVersion();
  if (version.major >= 5 && version.minor >= 0) {
    return true;
  } else {
    return false;
  }
}

int main() {
  int subResult = subtractGpu(20, 10);
  std::cout << "20 - 10 = " << subResult << std::endl;

  int multResult = multiplyGpu(100, 100);
  std::cout << "100 * 100 = " << multResult << std::endl;

  parallel1();
  cudaDeviceSynchronize();

  parallel2();
  cudaDeviceSynchronize();

  parallel3();
  cudaDeviceSynchronize();

  if (isVersionHighEnough()) {
    std::cout << "Cuda device version is above 5.0" << std::endl;
  }

  int h_in[50], h_out[50];

  for (int i = 0; i < 50; i++) {
    h_in[i] = i;
  }

  gpuCube(h_in, h_out);

  std::cout << "Cube of Number on GPU" << std::endl;
  for (int i = 0; i < 50; i++) {
    std::cout << "The cube of " << h_in[i] << " is " << h_out[i] << std::endl;
  }

  return 0;
}
