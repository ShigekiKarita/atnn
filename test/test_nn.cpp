#include <ATen/Functions.h>
#include <atnn/atnn.hpp>
#include <tuple>

namespace M = atnn::modules;

struct Net : atnn::ModuleSet {
    std::shared_ptr<M::Conv2d> conv2d = std::make_shared<M::Conv2d>(4, 2);
    std::shared_ptr<M::ReLU> relu = std::make_shared<M::ReLU>();

    Net() {
        this->modules = {conv2d, relu};
    }

    auto operator()(atnn::Variable x) {
        auto h = conv2d->forward(x);
        return std::make_tuple(h, relu->forward(h));
    }
};


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
        at::Tensor t = device(at::kFloat).randn({3, 4, 5, 6});
        atnn::Variable x(t);

        auto net = Net();
        if (device == at::CUDA) {
            net.toBackend(at::kCUDA);
        }

        atnn::Variable y, z;
        std::tie(y, z) = net(x);
        assert((z.data() >= 0.0).all());
        assert((z.data() * (z.data() != y.data()).toType(at::kFloat) == 0.0).all());

        assert(atnn::shape_is(z, {3, 2, 5-2, 6-2}));
        auto gz = device(at::kFloat).randn(z.sizes());
        z.backward(gz);
        assert(atnn::allclose(z.grad(), gz));
        assert(atnn::shape_is(x.grad(), x.sizes()));
    });
}
