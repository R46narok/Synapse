//
// Created by Acer on 16.3.2024 г..
//

#ifndef MEM_H
#define MEM_H

#include <array>

namespace synapse {
    inline uint32_t dim_width(std::array<int, 4> dim) {
        return dim[0] * dim[1] * dim[2] * dim[3];
    }
}

#endif //MEM_H
