#pragma once

#include "fwd.hpp"

namespace atnn {
    namespace autograd {

        struct Threshold : Function {
            virtual TList forward(ModulePtr ctx, TList xs) const {
                ATNN_ASSERT_EQ(xs.size(), 1);
                ctx->save_for_backward(xs);
                return {at::threshold_forward(xs[0], this->threshold, this->value, this->inplace)};
            }

            virtual TList backward(ModulePtr ctx, TList gy) const {
                ATNN_ASSERT_EQ(gy.size(), 1);
                auto x = ctx->saved_tensors()[0];
                ATNN_ASSERT_SHAPE_EQ(gy[0].sizes(), x.sizes());
                return {at::threshold_backward(gy[0], x, this->threshold, this->value, this->inplace)};
            }

            at::Scalar threshold, value;
            bool inplace;
            Threshold(double threshold, double value, bool inplace=false)
                : threshold(threshold), value(value), inplace(inplace)
                {}
        };

        struct ReLU : Threshold {
            explicit ReLU(bool inplace=false) : Threshold(0.0, 0.0, inplace) {}
        };

    }
}
