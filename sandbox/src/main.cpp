#include <iostream>

#include <synapse/tensor.h>
#include "synapse/activation.h"
using namespace synapse;

int main()
{
    tensor t(4, 4);
    tensor t1(4, 4);
    tensor t3(4, 4);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
           t(i, j) = (i + 1) + (j + 1);
           t1(i, j) = (i + 1) + (j + 1);
        }
    }

    t3 = t.mul(t1, 0, 1);
    // t3 = t.pointwise_mul(t1);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << t3(i, j) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
