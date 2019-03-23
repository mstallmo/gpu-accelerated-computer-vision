#include <iostream>
#include "Adder.h"

int main(void) {
    Adder adder(2048, 4096);

    int const result = adder.add();
    std::cout << "2048 + 4096 = " << result << std::endl;

    return 0;
}
