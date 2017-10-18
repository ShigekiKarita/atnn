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

    auto operator()(atnn::Variable x) const {
        auto h = conv2d->forward(x);
        return std::make_tuple(h, relu->forward(h));
    }
};


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
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
        auto f1 = [=](auto xs) {
            atnn::Variable y, z;
            std::tie(y, z) = net(xs[0]);
            return atnn::VList { y }; // TODO: grad_check with multiple outputs
        };
        atnn::grad_check(f1, {x}, {gz}, 1e-1); // , 1e-2, 1e-3, 5e-3);

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
