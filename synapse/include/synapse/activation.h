//
// Created by Acer on 16.3.2024 г..
//

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "synapse/tensor.h"

namespace synapse {

    enum class activation {
        sigmoid, relu, leaky_relu, none
    };
    void activation_sigmoid(tensor* tensor);
    void apply_activation(tensor* tensor, activation act);
}

#endif //ACTIVATION_H
