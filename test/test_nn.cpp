#define _GLIBCXX_DEBUG
#define DEBUG

#include <ATen/Functions.h>
#include <atnn/atnn.hpp>

/* TODO: support higher order modules
struct Net : atnn::Module<Net> {
    atnn::ModulePtr conv2d = std::make_shared<atnn::modules::Conv2d>(4, 2, {3, 3});
    atnn::ModulePtr relu = std::make_shared<atnn::modules::ReLU>();

    Net() {
        this->submodules = {conv2d, relu};
    }

    auto forward(Variable x) {
        auto h = this->conv2d->forward(x);
        return this->relu->forward(h);
    }
}
*/


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
        at::Tensor t = device(at::kFloat).randn({3, 4, 5, 6});
        atnn::Variable x(t);
        auto conv2d = std::make_shared<atnn::modules::Conv2d>(4, 2);
        auto relu = std::make_shared<atnn::modules::ReLU>();
        if (device == at::CUDA) {
            conv2d->toBackend(at::kCUDA);
        }
        assert(atnn::shape_is(conv2d->weight, {2, 4, 3, 3}));
        auto y = conv2d->forward(x);
        auto z = relu->forward(y);
        assert((z.data() >= 0.0).all());
        assert((z.data() * (z.data() != y.data()).toType(at::kFloat) == 0.0).all());

        assert(atnn::shape_is(z, {3, 2, 5-2, 6-2}));
        auto gz = device(at::kFloat).randn(z.sizes());
        z.backward(gz);
        assert(atnn::allclose(z.grad(), gz));
        assert(atnn::shape_is(x.grad(), x.sizes()));
    });
}
