#include "cudaOps/kernel.cuh"
#include <iostream>

#define N 5000

int main() {
  int h_a[N], h_b[N], h_c[N];

  for (int i = 0; i < N; i++) {
    h_a[i] = 2 * i * i;
    h_b[i] = i;
  }

  gpuAdd(h_a, h_b, h_c);

  int Correct = 1;
  for (int i = 0; i < N; i++) {
    if ((h_a[i] + h_b[i] != h_c[i])) {
      Correct = 0;
    }
  }

  if (Correct == 1) {
    std::cout << "GPU has computed sum correctly" << std::endl;
  } else {
      std::cout << "There is an error in the GPU computation" << std::endl;
  }

  return 0;
}
