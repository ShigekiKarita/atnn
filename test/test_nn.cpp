#define _GLIBCXX_DEBUG
#define DEBUG

#include <assert.h>
#include <iostream>
#include <ATen/Functions.h>

#include <atnn/atnn.hpp>


int main() {
    for (auto device: {at::CPU, at::CUDA}){
        at::Tensor t = device(at::kFloat).ones({3, 4, 5, 6});
        atnn::Variable x(t);
        auto conv2d = std::make_shared<atnn::modules::Conv2d>(4, 2);
        if (device == at::CUDA) conv2d->toBackend(at::kCUDA);

        assert(atnn::shape_is(conv2d->weight, {2, 4, 3, 3}));
        auto y= conv2d->forward(x);
        assert(atnn::shape_is(y, {3, 2, 5-2, 6-2}));
        auto gy = device(at::kFloat).rand(y.sizes());
        y.backward(gy);
        assert(atnn::allclose(y.grad(), gy));
        assert(atnn::shape_is(x.grad(), x.sizes()));
    }
}
