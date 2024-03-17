//
// Created by Acer on 16.3.2024 г..
//

#include "synapse/layers/dense.h"

#include <random>

#include "synapse/mem.h"
#include "synapse/tensor.h"

namespace synapse {
    dense_layer::dense_layer(const std::string &name, const uint32_t units, const activation activation)
        : base_layer(name), input(nullptr), units(units), batchSize(0), activationFn(activation) {
    }

    dense_layer::~dense_layer() noexcept {
        delete output;
        delete weights;
        delete biases;
    }


    void dense_layer::init_weights_bias(unsigned int seed) {
        if (weights == nullptr || biases == nullptr)
            return;

        std::random_device rd;
        std::mt19937 gen(seed == 0 ? rd() : seed);

        // uniform distribution
        auto dim = input->dim();
        auto width = dim_width(weights->dim());
        float range = sqrt(6.f / (dim[1] * dim[2] * dim[3])); // He's initialization
        std::uniform_real_distribution<> dis(-range, range);

        for (int i = 0; i < width; i++) {
            weights->elements[i] = static_cast<float>(dis(gen));
        }
        for (int i = 0; i < units; i++)
            biases->elements[i] = 0.f;

    }

    tensor *dense_layer::forward(tensor *in) {
        const auto inputDim = in->dim();
        int inputSize = inputDim[1] * inputDim[2] * inputDim[3];
        if (weights == nullptr) {
            weights = new tensor(1, 1, inputSize, units);
            biases = new tensor(1, 1, units);
        }

        if (input == nullptr || batchSize != inputDim[0]) {
            input = in;
            batchSize = inputDim[0];

            output = new tensor(batchSize, units);

            if (!frozen) {
                init_weights_bias(0);
            }
        }

        // output = input * weights + bias
        // [batchSize x units x 1 x 1] = [batchSize x (c x h x w) x 1 x 1] . [1 x 1 x (c x h x w) x units]

        input->reshape(batchSize, inputSize, 1, 1);
        weights->reshape(inputSize, units, 1, 1);

        *output = input->mul(*weights);
        *output = output->pointwise_add(*biases);

        input->reshape(inputDim);
        weights->reshape(1, 1, inputSize, units);

        apply_activation(output, activationFn);

        return output;
    }

    tensor* dense_layer::backward(tensor *din) {
        return nullptr;
    }
}
