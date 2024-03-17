#include <iostream>
#include <random>

#include <synapse/tensor.h>
#include "synapse/activation.h"
#include "synapse/graph.h"
using namespace synapse;

int main() {

    int batchSize = 17;
    graph g;

    g.add_layer(new dense_layer("initial", 32, activation::sigmoid));
    g.add_layer(new dense_layer("secondary", 64, activation::sigmoid));
    g.add_layer(new dense_layer("secondary", 128, activation::sigmoid));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    tensor input(batchSize, 1, 48, 48);
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < 48; ++j) {
            for (int k = 0; k < 48; ++k) {
                input(i,0,  j, k) = dis(gen);
            }
        }
    }
    auto* out = g.forward(&input);
}
