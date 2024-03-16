//
// Created by Acer on 16.3.2024 г..
//

#include "synapse/activation.h"
#include "synapse/mem.h"

#include <immintrin.h> // Include for AVX2 intrinsics
#include <cmath>
#include <iostream>

namespace synapse {
    void activation_sigmoid(tensor *tensor) {
    auto dim = tensor->dim();
    uint32_t size = dim_width(dim);

    for (int i = 0; i < size; i += 8) {
        __m256 tensorElements = _mm256_loadu_ps(&tensor->elements[i]);
        __m256 expNegX = _mm256_exp_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), tensorElements));
        __m256 denominator = _mm256_add_ps(_mm256_set1_ps(1.0f), expNegX);
        __m256 result = _mm256_div_ps(_mm256_set1_ps(1.0f), denominator);

        _mm256_storeu_ps(&tensor->elements[i], result);
    }

    for (int i = size - (size % 8); i < size; ++i) {
        float denominator = 1.0f + exp(-1.0f * tensor->elements[i]);
        tensor->elements[i] = 1.0f / denominator;
    }
}

    void apply_activation(tensor *tensor, activation act) {
        switch (act) {
            case activation::sigmoid:
                activation_sigmoid(tensor);
                break;
            case activation::relu:
                break;
            case activation::leaky_relu:
                break;
        }

    }
}
