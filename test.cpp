#define _GLIBCXX_DEBUG
#define DEBUG

#include <atenn/atenn.hpp>


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
        return {gy[0] * _x / ctx->n};
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

// thnn codes


int main() {
    at::Tensor d = CUDA(at::kFloat).ones({3, 4});
    at::Tensor r = CUDA(at::kFloat).zeros({3,4});

    auto v0 = std::make_shared<atnn::Variable>(d * 3);
    auto func = std::make_shared<Pow>(2);
    auto v1 = func->forward(v0);
    std::cout << *v1 << std::endl;

    v1->backward(v1->data/2);
    std::cout << v1 << std::endl;
    std::cout << v0 << std::endl;

    auto add = std::make_shared<Add>();
    auto v2 = add->forward(v0, v1);
    v2->backward(d);
    std::cout << v1 << std::endl;
    std::cout << v0 << std::endl;
}
