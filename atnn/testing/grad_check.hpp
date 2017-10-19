#pragma once

#include <ATen/ATen.hpp>
#include "../tuple.hpp"
#include "../testing.hpp"


namespace atnn {
    namespace testing {

        VList to_vlist(Variable v) {
            return {v};
        }

        VList to_vlist(VList v) {
            return v;
        }

        template <typename ... Ts>
        VList to_vlist(std::tuple<Ts...> vtuple) {
            auto varray = to_array(vtuple);
            return VList(varray.begin(), varray.end());
        }

        template <typename F>
        auto numeric_grad(F func, VList inputs, TList grad_outputs, float eps=1e-3) {
            TList grad_inputs;
            grad_inputs.reserve(inputs.size());
            for (size_t n = 0; n < inputs.size(); ++n) {
                auto x = inputs[n];
                auto x_flatten = x.data().view(-1);
                auto gx_data = x.data().type().zeros_like(x.data());
                auto gx_flatten = gx_data.view(-1);

                for (long i = 0; i < x_flatten.size(0); ++i) {
                    float xi_origin = at::Scalar(x_flatten[i]).toFloat();
                    x_flatten[i] = xi_origin + eps;
                    auto a = to_vlist(func(inputs));
                    x_flatten[i] = xi_origin - eps;
                    auto b = to_vlist(func(inputs));
                    x_flatten[i] = xi_origin;

                    for (size_t m = 0; m < grad_outputs.size(); ++ m) {
                        float g = (grad_outputs[m] * (a[m].data() - b[m].data())).sum().toDouble() / (2.0 * eps);
                        gx_flatten[i] += at::Scalar(g);
                    }
                }
                grad_inputs.push_back(gx_data);
            }
            return grad_inputs;
        }

        at::Tensor limit_view(at::Tensor t, long max_view=100) {
            return max_view >= t.view(-1).size(0) ? t : t.view(-1).narrow(0, 0, max_view);
        }

        template <typename F>
        auto grad_check(F func, VList inputs, TList grad_outputs, float eps=1e-3, float rtol=1e-3, float atol=1e-5, long max_view=20) {
            const auto ngs = numeric_grad(func, inputs, grad_outputs, eps);
            auto outs = to_vlist(func(inputs));
            ATNN_ASSERT_EQ(outs.size(), grad_outputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                inputs[i].clear_grads();
            }
            for (size_t i = 0; i < outs.size(); ++i) {
                outs[i].backward(grad_outputs[i]);
            }
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::ostringstream ss;
                ss << "backprop grad:\n" << limit_view(inputs[i].grad(), max_view) << "\n"
                   << "numerical grad:\n" << limit_view(ngs[i], max_view);
                ATNN_ASSERT_MSG(allclose(inputs[i].grad(), ngs[i], rtol, atol), ss.str().c_str());
            }
        }
    } // namespace testing
} // namespace atnn
