//
// Created by Acer on 16.3.2024 г..
//

#ifndef DENSE_H
#define DENSE_H

#include <string>

#include "base_layer.h"
#include "synapse/activation.h"

namespace synapse {
    class dense_layer : public base_layer {
    public:
        dense_layer(const std::string& name, uint32_t units, activation activation = activation::none);
    private:
        uint32_t units;
        activation activationFn;
    };
}

#endif //DENSE_H
