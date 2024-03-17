//
// Created by Acer on 16.3.2024 г..
//

#include "synapse/tensor.h"
#include <immintrin.h> // Include for AVX2 intrinsics
#include <iostream>

namespace synapse {
    tensor::tensor(int n, int c, int h, int w)
        : n(n), c(c), h(h), w(w) {
        elements = static_cast<float *>(_aligned_malloc(n * c * h * w * sizeof(float), 32));
    }

    tensor::~tensor() {
        if (elements != nullptr) {
            _aligned_free(elements);
        }
    }

    void tensor::reshape(int new_n, int new_c, int new_h, int new_w) {
        n = new_n;
        c = new_c;
        h = new_h;
        w = new_w;
    }

    void tensor::reshape(const std::array<int, 4> &dim) {
        n = dim[0];
        c = dim[1];
        h = dim[2];
        w = dim[3];
    }

    tensor_op tensor::pointwise_mul(tensor &other) {
        return {this, &other, tensor_op::pointwise_mul};
    }

    tensor_op tensor::pointwise_add(tensor &other) {
        return {this, &other, tensor_op::pointwise_add};
    }

    tensor_op tensor::pointwise_subtr(tensor &other) {
        return {this, &other, tensor_op::pointwise_subtr};
    }

    tensor_op tensor::pointwise_divide(tensor &other) {
        return {this, &other, tensor_op::pointwise_divide};
    }

    tensor_op tensor::mul(tensor &other) {
        return {this, &other, tensor_op::mul};
    }

    float &tensor::operator()(int x, int y, int z, int v) {
        return elements[x * c * h * w + y * h * w + z * w + v];
    }

    void tensor::pointwise_mul_impl(tensor *left, tensor *right, tensor *result) {
        auto resultDim = result->dim();

        __m256 vecLeft, vecRight, vecResult;
        uint32_t size = resultDim[0] * resultDim[1] * resultDim[2] * resultDim[3];

        for (uint32_t i = 0; i < size; i += 8) {
            vecLeft = _mm256_loadu_ps(&left->elements[i]);
            vecRight = _mm256_loadu_ps(&right->elements[i]);
            vecResult = _mm256_mul_ps(vecLeft, vecRight);
            _mm256_storeu_ps(&result->elements[i], vecResult);
        }
    }

    void tensor::pointwise_add_impl(tensor *left, tensor *right, tensor *result) {
        auto resultDim = result->dim();

        __m256 vecLeft, vecRight, vecResult;
        uint32_t size = resultDim[0] * resultDim[1] * resultDim[2] * resultDim[3];

        for (uint32_t i = 0; i < size; i += 8) {
            vecLeft = _mm256_loadu_ps(&left->elements[i]);
            vecRight = _mm256_loadu_ps(&right->elements[i]);
            vecResult = _mm256_add_ps(vecLeft, vecRight);
            _mm256_storeu_ps(&result->elements[i], vecResult);
        }
    }

    void tensor::pointwise_subtr_impl(tensor *left, tensor *right, tensor *result) {
        auto resultDim = result->dim();

        __m256 vecLeft, vecRight, vecResult;
        uint32_t size = resultDim[0] * resultDim[1] * resultDim[2] * resultDim[3];

        for (uint32_t i = 0; i < size; i += 8) {
            vecLeft = _mm256_loadu_ps(&left->elements[i]);
            vecRight = _mm256_loadu_ps(&right->elements[i]);
            vecResult = _mm256_sub_ps(vecLeft, vecRight);
            _mm256_storeu_ps(&result->elements[i], vecResult);
        }
    }

    void tensor::pointwise_divide_impl(tensor *left, tensor *right, tensor *result) {
        auto resultDim = result->dim();

        __m256 vecLeft, vecRight, vecResult;
        uint32_t size = resultDim[0] * resultDim[1] * resultDim[2] * resultDim[3];

        for (uint32_t i = 0; i < size; i += 8) {
            vecLeft = _mm256_loadu_ps(&left->elements[i]);
            vecRight = _mm256_loadu_ps(&right->elements[i]);
            vecResult = _mm256_div_ps(vecLeft, vecRight);
            _mm256_storeu_ps(&result->elements[i], vecResult);
        }
    }

    int get_index(uint32_t axis, int i, int j, int k, int n, int p, int q) {
        return (axis == 0)
                   ? i * n * p * q + j * p * q + k * q
                   : (axis == 1)
                         ? i * p * q + j * q + k
                         : (axis == 2)
                               ? i * q + j
                               : i;
    }


    void tensor::mul_impl(tensor *left, tensor *right, tensor *result) { // temorary linear approach
        int n = result->dim()[0];
        int c = result->dim()[1];

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < left->dim()[1]; ++k) {
                    sum += left->operator()(i, k, 0, 0) * right->operator()(k, j, 0, 0);
                }
                result->operator()(i, j, 0, 0) = sum;
            }
        }
    }

    tensor &tensor::operator=(const tensor_op &op) {
        switch (op.code) {
            case tensor_op::pointwise_mul:
                pointwise_mul_impl(op.left, op.right, this);
                break;
            case tensor_op::pointwise_add:
                pointwise_add_impl(op.left, op.right, this);
                break;
            case tensor_op::pointwise_subtr:
                pointwise_subtr_impl(op.left, op.right, this);
                break;
            case tensor_op::pointwise_divide:
                pointwise_divide_impl(op.left, op.right, this);
                break;
            case tensor_op::mul:
                mul_impl(op.left, op.right, this);
                break;
            default:
                break;
        }

        return *this;
    }
}
