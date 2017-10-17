#pragma once

#include <ATen/ATen.h>

#include "autograd.hpp"
#include "testing.hpp"


namespace atnn {

    namespace modules {

        struct Threshold : atnn::Module<Threshold> {
            using Function = struct {
                template <typename Context>
                static auto forward(Context ctx, atnn::TList xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    ctx->save_for_backward(xs);
                    return at::threshold_forward(xs[0], ctx->threshold, ctx->value, ctx->inplace);
                }

                template <typename Context>
                static atnn::TList backward(Context ctx, atnn::TList gy) {
                    ATNN_ASSERT_EQ(gy.size(), 1);
                    auto x = ctx->saved_tensors[0];
                    ATNN_ASSERT_SHAPE_EQ(gy[0].sizes(), x.sizes());
                    return {at::threshold_backward(gy[0], x, ctx->threshold, ctx->value, ctx->inplace)};
                }
            };

            at::Scalar threshold, value;
            bool inplace;
            Threshold(double threshold, double value, bool inplace=false)
                : threshold(threshold), value(value), inplace(inplace)
                {}
        };

        struct ReLU : Threshold {
            explicit ReLU(bool inplace=false) : Threshold(0.0, 0.0, inplace) {}
        };

        struct Conv2d : atnn::Module<Conv2d> {
            using Function = struct {
                // static inline Tensor & conv2d_forward_out(
                //   Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias,
                //   IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input);
                template <typename Context>
                static auto forward(Context ctx, atnn::TList xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    ATNN_ASSERT_EQ(xs[0].dim(), 4);
                    ctx->save_for_backward(xs);
                    auto&& x = xs[0];
                    at::Tensor output = x.type().zeros_like(x);
                    atnn::to_backend_of(ctx->finput, x);
                    atnn::to_backend_of(ctx->fgrad_input, x);
                    return at::conv2d_forward_out(output, x, ctx->weight.data(), ctx->kernel_size, ctx->bias.data(),
                                                  ctx->stride, ctx->padding, ctx->finput, ctx->fgrad_input);
                }

                // static inline std::tuple<Tensor &,Tensor &,Tensor &> conv2d_backward_out(
                //   Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output,
                //   const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding,
                //   const Tensor & finput, const Tensor & fgrad_input);
                template <typename Context>
                static atnn::TList backward(Context ctx, atnn::TList gy) {
                    ATNN_ASSERT_EQ(gy.size(), 1);
                    ATNN_ASSERT_EQ(gy[0].dim(), 4);
                    auto&& grad_output = gy[0];
                    auto&& x = ctx->saved_tensors[0];
                    auto grad_input = x.type().zeros_like(x);
                    atnn::to_backend_of(ctx->finput, x);
                    atnn::to_backend_of(ctx->fgrad_input, x);
                    at::conv2d_backward_out(grad_input, ctx->weight.ptr->grad, ctx->weight.ptr->grad, grad_output,
                                            x, ctx->weight.data(), ctx->kernel_size, ctx->stride, ctx->padding,
                                            ctx->finput, ctx->fgrad_input);
                    return {grad_input};
                }
            };

            atnn::Variable weight, bias;
            at::Tensor finput, fgrad_input; // buffers
            at::IntList kernel_size, stride, padding;

            Conv2d(long in_channels, long out_channels, at::IntList kernel_size={3, 3}, at::IntList stride={1, 1}, at::IntList padding={0, 0})
                // TODO: init nicely
                : weight(CPU(at::kFloat).randn({out_channels, in_channels, kernel_size[0], kernel_size[1]}))
                , bias(CPU(at::kFloat).zeros(out_channels))
                , finput(CPU(at::kFloat).zeros(out_channels))
                , fgrad_input(CPU(at::kFloat).zeros(out_channels))
                , kernel_size(kernel_size)    
                , stride(stride)
                , padding(padding) {
                this->parameters = {this->weight, this->bias};
            }
        };

    }
}
