#define _GLIBCXX_DEBUG
#define DEBUG

#include <atnn/atnn.hpp>


// you can define your own module's impl outside of it.
struct PowImpl {
    // you cannot create non-static members. use ctx for storing dynamic members
    template <typename Context>
    static auto forward(Context ctx, atnn::TList x) {
        ctx->save_for_backward(x);
        return x[0].pow(ctx->n);
    }

    template <typename Context>
    static atnn::TList backward(Context ctx, atnn::TList gy) {
        auto&& _x = ctx->saved_tensors[0];
        return {gy[0] * _x.pow(ctx->n - 1) * ctx->n};
    }
};


struct Pow : atnn::Module<Pow> {
    using Impl = struct PowImpl;
    double n = 2;
    Pow(double n) : n(n) {}
};


// or inline impl style
struct Add : atnn::Module<Add> {
    using Impl = struct {
        template <typename Context>
        static auto forward(Context ctx, atnn::TList x) {
            ctx->save_for_backward(x);
            return x[0] + x[1];
        }

        template <typename Context>
        static atnn::TList backward(Context ctx, atnn::TList gy) {
            auto&& _x = ctx->saved_tensors[0];
            return {gy[0], gy[0]};
        }
    };
};


auto allclose(at::Tensor actual, at::Tensor desired, float rtol=1e-7, float atol=0) {
    return ((actual - desired).abs() <= desired.abs() * rtol + atol).all();
}


int main() {
    {
        at::Tensor d = CUDA(at::kFloat).rand({3, 4});
        auto d_clone = d.clone();

        auto v0 = std::make_shared<atnn::Variable>(d * 3);
        auto func = std::make_shared<Pow>(2);
        auto v1 = func->forward(v0);
        assert(allclose(v1->data, v0->data.pow(2)));
        v1->backward(v1->data/2);

        assert(allclose(d, d_clone)); // check unchanged
        assert(allclose(v1->grad, v1->data/2));
        assert(allclose(v0->grad, v1->grad * v0->data * 2));
        auto prev_g1 = v1->grad.clone();
        auto prev_g0 = v0->grad.clone();

        auto add = std::make_shared<Add>();
        auto v2 = add->forward(v0, v1); // v2 = v0 + (v0 * v0) -> dv2/dv0 = 1 + 2 * v0

        v2->clear_grads();
        assert(atnn::is_empty(v0->grad));
        assert(atnn::is_empty(v1->grad));

        v2->backward(d);
        assert(allclose(d, d_clone)); // check unchanged
        assert(allclose(v2->grad, d));
        assert(allclose(v1->grad, d));
        assert(allclose(v0->grad, d * (2 * v0->data + 1), 1e-6));

        // TODO: test w/o clear grads
    }

    {
        at::Tensor d = CUDA(at::kFloat).ones({3, 4});
        auto v0 = std::make_shared<atnn::Variable>(d * 3);
        auto v1 = std::make_shared<atnn::Variable>(d * 2);
        auto add = std::make_shared<Add>();
        auto v2 = add->forward(v0, v1);
        assert(allclose(v2->data, d * 5));
        v2->backward(d);
        assert(allclose(v0->grad, d));
        assert(allclose(v1->grad, d));
    }
}
