//
// Created by Acer on 16.3.2024 г..
//

#ifndef TENSOR_H
#define TENSOR_H
#include <memory>
#include <array>

namespace synapse {
    struct tensor_op;

    class tensor {
    public:
        tensor(int m = 1, int n = 1, int p = 1, int q = 1);
        ~tensor();

        tensor_op pointwise_mul(tensor& other);
        tensor_op pointwise_add(tensor& other);
        tensor_op pointwise_subtr(tensor& other);
        tensor_op pointwise_divide(tensor& other);
        tensor_op mul(tensor& other, uint32_t firstAxis, uint32_t secondAxis);

        tensor& operator=(const tensor_op& op);
        float& operator()(int x = 0, int y = 0, int z = 0, int w = 0);

        std::array<int, 4> dim() { return {m, n, p, q}; }
    private:
        int m, n, p, q;
        float* elements;

    private:
        static void pointwise_mul_impl(tensor* left, tensor* right, tensor* result);
        static void pointwise_add_impl(tensor* left, tensor* right, tensor* result);
        static void pointwise_subtr_impl(tensor* left, tensor* right, tensor* result);
        static void pointwise_divide_impl(tensor* left, tensor* right, tensor* result);
        static void mul_impl(tensor* left, tensor* right, tensor* result, uint32_t firstAxis, uint32_t secondAxis);
    };

    struct tensor_op {
        tensor* left;
        tensor* right;
        uint32_t code;
        uint32_t firstAxis;
        uint32_t secondAxis;

        enum {
            pointwise_mul, pointwise_divide, pointwise_add, pointwise_subtr,
            mul
        };
    };


}

#endif //TENSOR_H
