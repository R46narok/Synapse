//
// Created by Acer on 16.3.2024 г..
//

#ifndef DENSE_H
#define DENSE_H

#include <string>

#include "synapse/activation.h"
#include "synapse/tensor.h"
#include "synapse/layers/base_layer.h"

namespace synapse {
    class dense_layer : public base_layer {
    public:
        dense_layer(const std::string& name, uint32_t units, activation activation = activation::none);
        ~dense_layer() noexcept override;

        tensor* forward(tensor* in) override;
        tensor* backward(tensor* din) override;

    private:
        void init_weights_bias(unsigned int seed);

        bool frozen = false;

        tensor* weights = nullptr;
        tensor* biases = nullptr;
        tensor* input = nullptr;
        tensor* output = nullptr;

        std::unique_ptr<tensor> dweights;
        std::unique_ptr<tensor> dbiases;

        uint32_t units;
        uint32_t batchSize;
        activation activationFn;
    };
}

#endif //DENSE_H
