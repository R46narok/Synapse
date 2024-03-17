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
        tensor(int n = 1, int c = 1, int h = 1, int w = 1);
        ~tensor();

        void reshape(int new_n, int new_c, int new_h, int new_w);
        void reshape(const std::array<int, 4>& dim);

        tensor_op pointwise_mul(tensor& other);
        tensor_op pointwise_add(tensor& other);
        tensor_op pointwise_subtr(tensor& other);
        tensor_op pointwise_divide(tensor& other);
        tensor_op mul(tensor& other);

        tensor& operator=(const tensor_op& op);
        float& operator()(int x = 0, int y = 0, int z = 0, int v = 0);

        std::array<int, 4> dim() const { return {n, c, h, w}; }

        friend void activation_sigmoid(tensor* tensor);
        friend class dense_layer;

    private:
        int n, c, h, w;
        float* elements = nullptr;

    private:
        static void pointwise_mul_impl(tensor* left, tensor* right, tensor* result);
        static void pointwise_add_impl(tensor* left, tensor* right, tensor* result);
        static void pointwise_subtr_impl(tensor* left, tensor* right, tensor* result);
        static void pointwise_divide_impl(tensor* left, tensor* right, tensor* result);
        static void mul_impl(tensor* left, tensor* right, tensor* result);
    };

    struct tensor_op {
        tensor* left;
        tensor* right;
        uint32_t code;

        enum {
            pointwise_mul, pointwise_divide, pointwise_add, pointwise_subtr,
            mul
        };
    };


}

#endif //TENSOR_H
