//
// Created by Acer on 16.3.2024 г..
//

#ifndef BASE_LAYER_H
#define BASE_LAYER_H
#include <string>

#include "synapse/tensor.h"

namespace synapse {
    class base_layer {
    public:
        virtual ~base_layer() = default;

        virtual tensor* forward(tensor* input) = 0;
        virtual tensor* backward(tensor* dinput) = 0;
    protected:
        base_layer(std::string name);
        std::string name;
    };
}

#endif //BASE_LAYER_H
