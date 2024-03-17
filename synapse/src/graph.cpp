//
// Created by Acer on 17.3.2024 г..
//

#include "synapse/graph.h"

namespace synapse {
    graph::~graph() noexcept {
        for (auto& layer : layers) {
            delete layer;
        }
    }

    void graph::add_layer(base_layer *layer) {
        layers.push_back(layer);
    }

    tensor* graph::forward(tensor *input) const {
        tensor* output = input;

        for (auto& layer : layers) {
            output = layer->forward(output);
        }

        return output;
    }
}
