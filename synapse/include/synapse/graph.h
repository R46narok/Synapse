//
// Created by Acer on 16.3.2024 г..
//

#ifndef GRAPH_H
#define GRAPH_H

#include "synapse/layers/dense.h"

#include <vector>

namespace synapse {
    class graph { // for now its just a forward impl
    public:
        graph() = default;
        ~graph() noexcept;

        void add_layer(base_layer* layer);
        tensor* forward(tensor* input) const;
    private:
        std::vector<base_layer*> layers;
    };

}

#endif //GRAPH_H
