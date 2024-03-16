//
// Created by Acer on 16.3.2024 г..
//

#include "synapse/tensor.h"
#include <immintrin.h> // Include for AVX2 intrinsics
#include <iostream>

namespace synapse {
    tensor::tensor(int m, int n, int p, int q)
        : m(m), n(n), p(p), q(q) {
        elements = static_cast<float*>(_aligned_malloc(m * n * p * q * sizeof(float), 32));
    }

    tensor::~tensor() {
        _aligned_free(elements);
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

    tensor_op tensor::mul(tensor &other, uint32_t firstAxis, uint32_t secondAxis) {
        return {this, &other, tensor_op::mul, firstAxis, secondAxis};
    }

    float& tensor::operator()(int x, int y, int z, int w) {
        return elements[ x * n * p * q + y * p * q + z * q + w];
    }

    void tensor::pointwise_mul_impl(tensor* left, tensor* right, tensor* result) {
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

    void tensor::pointwise_add_impl(tensor* left, tensor* right, tensor* result) {
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

    void tensor::pointwise_subtr_impl(tensor* left, tensor* right, tensor* result) {
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

    void tensor::pointwise_divide_impl(tensor* left, tensor* right, tensor* result) {
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
        return (axis == 0) ? i * n * p * q + j * p * q + k * q : (axis == 1) ? i * p * q + j * q + k : (axis == 2) ? i * q + j : i;
    }


    void tensor::mul_impl(tensor *left, tensor *right, tensor *result, uint32_t firstAxis, uint32_t secondAxis) {
        std::array<int, 4> leftDim = left->dim();
        std::array<int, 4> rightDim = right->dim();
        std::array<int, 4> resultDim = result->dim();

        if (leftDim[firstAxis] != rightDim[secondAxis]) {
            std::cerr << "Dimensions mismatch for multiplication!" << std::endl;
            return;
        }

        int m = resultDim[0];
        int n = resultDim[1];
        int p = resultDim[2];
        int q = resultDim[3];

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < p; ++k) {
                    for (int l = 0; l < q; l += 8) {
                        __m256 sum = _mm256_setzero_ps();
                        for (int r = 0; r < leftDim[firstAxis]; ++r) {
                            __m256 leftVec = _mm256_loadu_ps(&(*left)(firstAxis == 0 ? r : i,
                                                                       firstAxis == 1 ? r : j,
                                                                       firstAxis == 2 ? r : k,
                                                                       firstAxis == 3 ? r : l));
                            __m256 rightVec = _mm256_loadu_ps(&(*right)(secondAxis == 0 ? r : i,
                                                                         secondAxis == 1 ? r : j,
                                                                         secondAxis == 2 ? r : k,
                                                                         secondAxis == 3 ? r : l));
                            sum = _mm256_fmadd_ps(leftVec, rightVec, sum); // Fused multiply-add operation
                        }
                        _mm256_storeu_ps(&(*result)(i, j, k, l) , sum);
                    }
                }
            }
        }

    }

    tensor& tensor::operator=(const tensor_op &op) {
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
                mul_impl(op.left, op.right, this, op.firstAxis, op.secondAxis);
                break;
            default:
                break;
        }

        return *this;
    }
}
