#include <ATen/Functions.h>
#include <atnn/atnn.hpp>
#include <tuple>


namespace M = atnn::modules;

struct Net : atnn::ModuleSet {
    std::shared_ptr<M::Conv2d> conv2d = std::make_shared<M::Conv2d>(4, 2);
    std::shared_ptr<M::Sigmoid> sigmoid = std::make_shared<M::Sigmoid>();

    Net() {
        this->modules = {conv2d, sigmoid};
    }

    auto operator()(atnn::Variable x) const {
        auto h = conv2d->forward(x);
        return std::make_tuple(h, sigmoid->forward(h));
    }
};


template <typename F, typename D>
void test_nonlinearity(D device) {
    auto module = std::make_shared<M::Sigmoid>();
    if (device == at::CUDA) { module->toBackend(at::kCUDA); }
    auto f = [=](auto xs) { return atnn::VList {module->forward(xs[0])};};

    atnn::Variable x(device(at::kFloat).randn({3, 4, 5, 6}));
    auto gy = device(at::kFloat).ones_like(x.data());
    atnn::grad_check(f, {x}, {gy}, 1e-2);
}


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {

        test_nonlinearity<M::Sigmoid>(device);
        test_nonlinearity<M::Tanh>(device);
        test_nonlinearity<M::ReLU>(device);

        atnn::Variable x(device(at::kFloat).randn({3, 4, 5, 6}));
        auto gz = device(at::kFloat).ones({3, 2, 3, 4});

        auto conv2d = std::make_shared<M::Conv2d>(4, 2);
        if (device == at::CUDA) {
            conv2d->toBackend(at::kCUDA);
        }
        auto f0 = [=](auto xs) { return atnn::VList {conv2d->forward(xs[0])};};
        atnn::grad_check(f0, {x}, {gz}, 1e-1);

        auto net = Net();
        if (device == at::CUDA) {
            net.toBackend(at::kCUDA);
        }
        auto f2 = [=](auto xs) {
            atnn::Variable y, z;
            std::tie(y, z) = net(xs[0]);
            return atnn::VList { z }; // TODO: grad_check y and z
        };
        atnn::grad_check(f2, {x}, {gz}, 1e-2, 1e-3, 1e-4);

        /*
        atnn::Variable y, z;
        std::tie(y, z) = net(x);
        // auto out = net(x);
        // auto y = std::get<0>(out);
        // auto z = std::get<1>(out);
        assert((z.data() >= 0.0).all());
        assert((z.data() * (z.data() != y.data()).toType(at::kFloat) == 0.0).all());

        assert(atnn::shape_is(z, {3, 2, 5-2, 6-2}));

        z.backward(gz);
        assert(atnn::allclose(z.grad(), gz));
        assert(atnn::shape_is(x.grad(), x.sizes()));
        */
    });
}
