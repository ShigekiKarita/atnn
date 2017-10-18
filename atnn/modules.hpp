#pragma once

#include <ATen/ATen.h>
#include <ATen/Functions.h>

#include "autograd.hpp"
#include "testing.hpp"

#define ATNN_UNARY_STATIC_FUNCTION(module, prefix)                      \
    struct module : atnn::Module<module> {                              \
        using Function = struct {                                       \
            template <typename Context>                                 \
            static auto forward(Context ctx, atnn::TList xs) {          \
                ATNN_ASSERT_EQ(xs.size(), 1);                           \
                auto y = at::prefix##_forward(xs[0]);                   \
                ctx->save_for_backward({y});                            \
                return y;                                               \
            }                                                           \
            template <typename Context>                                 \
            static atnn::TList backward(Context ctx, atnn::TList gy) {  \
                ATNN_ASSERT_EQ(gy.size(), 1);                           \
                auto y = ctx->saved_tensors[0];                         \
                ATNN_ASSERT_SHAPE_EQ(gy[0].sizes(), y.sizes());         \
                return {at::prefix##_backward(gy[0], y)};               \
            }                                                           \
        };                                                              \
    }

#define ATNN_NORMALIZED_STATIC_FUNCTION(module, prefix)                 \
    struct module : atnn::Module<module> {                              \
        using Function = struct {                                       \
            template <typename Context>                                 \
            static auto forward(Context ctx, atnn::TList xs) {          \
                ATNN_ASSERT_EQ(xs.size(), 1);                           \
                auto y = at::prefix##_forward(xs[0]);                   \
                ctx->save_for_backward({xs[0], y});                     \
                return y;                                               \
            }                                                           \
            template <typename Context>                                 \
            static atnn::TList backward(Context ctx, atnn::TList gy) {  \
                ATNN_ASSERT_EQ(gy.size(), 1);                           \
                auto x = ctx->saved_tensors[0];                         \
                auto y = ctx->saved_tensors[1];                         \
                ATNN_ASSERT_SHAPE_EQ(gy[0].sizes(), x.sizes());         \
                return {at::prefix##_backward(gy[0], x, y)};            \
            }                                                           \
        };                                                              \
    }


namespace atnn {

    namespace modules {

        ATNN_UNARY_STATIC_FUNCTION(Sigmoid, _sigmoid);
        ATNN_UNARY_STATIC_FUNCTION(Tanh, _tanh);
        ATNN_NORMALIZED_STATIC_FUNCTION(LogSoftmax, log_softmax);
        ATNN_NORMALIZED_STATIC_FUNCTION(Softmax, softmax);
        // TODO: cross_entropy and nll_loss

        struct NLLLoss : atnn::Module<NLLLoss> {
            using Function = struct {
                template <typename Context>
                static auto forward(Context ctx, atnn::TList xs) {
                    ATNN_ASSERT_EQ(xs.size(), 2);
                    ctx->save_for_backward(xs);
                    return at::nll_loss_forward(xs[0], xs[1], ctx->weight, ctx->size_average, ctx->ignore_index, ctx->total_weight);
                }

                template <typename Context>
                static atnn::TList backward(Context ctx, atnn::TList) {
                    auto xs = ctx->saved_tensors;
                    return {at::nll_loss_backward(xs[0], xs[1], ctx->weight, ctx->size_average, ctx->ignore_index, ctx->total_weight)};
                }
            };

            bool size_average;
            int64_t ignore_index;
            at::Tensor weight, total_weight;
            NLLLoss(bool size_average=true, int64_t ignore_index=-1, at::Tensor weight={}, at::Tensor total_weight={})
                : size_average(size_average), ignore_index(ignore_index), weight(weight), total_weight(total_weight)
                {}
        };

        struct MSELoss : atnn::Module<MSELoss> {
            using Function = struct {
                template <typename Context>
                static auto forward(Context ctx, atnn::TList xs) {
                    ATNN_ASSERT_EQ(xs.size(), 2);
                    ctx->save_for_backward(xs);
                    return at::mse_loss_forward(xs[0], xs[1], ctx->size_average, ctx->reduce);
                }

                template <typename Context>
                static atnn::TList backward(Context ctx, atnn::TList gy) {
                    auto xs = ctx->saved_tensors;

                    // TODO: support user-defined gy?
                    at::Tensor grad_out;
                    if (gy.size() == 0) { grad_out = xs[0].type().ones_like(xs[0]); }
                    else { ATNN_ASSERT_EQ(gy.size(), 1); grad_out == 0; }
                    return {at::mse_loss_backward(grad_out, xs[0], xs[1], ctx->size_average, ctx->reduce)};
                }
            };

            const bool size_average = true;
            const bool reduce = true;
            MSELoss(bool size_average=true, bool reduce=true)
                : size_average(size_average), reduce(reduce)
                {}
        };

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

        struct Linear : atnn::Module<Linear> {
            // https://github.com/chainer/chainer/blob/master/chainer/functions/connection/linear.py
            using Function = struct {
                template <typename Context>
                static auto forward(Context ctx, atnn::TList xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    ctx->save_for_backward(xs);

                    auto y = xs[0].mm(ctx->weight.data().t());
                    if (ctx->bias.data().defined()) {
                        y += ctx->bias.data().expand(y.sizes());
                    }
                    return y;
                }
                template <typename Context>
                static atnn::TList backward(Context ctx, atnn::TList gy) {
                    ATNN_ASSERT_EQ(gy.size(), 1);
                    auto x = ctx->saved_tensors[0];
                    auto gx = gy[0].mm(ctx->weight.data());
                    auto grad_weight = gy[0].t().mm(x.t());

                    // FIXME: assign grad uniformliy instead of separately
                    // now: parameters.grad (set inside function), arguments.grad (set outside function)
                    // refactor: set them outside uniformly and call Funtion from Module
                    // Module<Derived>.forward(VList xs) { return this->function(this, this->parameters ++ xs) }
                    if (ctx->weight.grad().defined())
                        ctx->weight.ptr->grad += grad_weight;
                    else ctx->weight.ptr->grad = grad_weight;

                    at::Tensor grad_bias;
                    if (ctx->bias.data().defined()) {
                        grad_bias = gy[0].sum(0);
                        if (ctx->bias.grad().defined())
                            ctx->bias.ptr->grad += grad_bias;
                        else ctx->bias.ptr->grad = grad_bias; 
                    }
                    return {gx}; // FIXME: return {gx, gw, gb}
                }
            };

            Variable weight, bias;
            Linear(long in_features, long out_features, bool use_bias=true)
                : weight(CPU(at::kFloat).randn({out_features, in_features}))
                , bias(use_bias ? CPU(at::kFloat).randn({out_features}) : at::Tensor{})
                {}
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
                    // FIXME: increment grad instead of assign?
                    at::Tensor grad_weight, grad_bias;
                    grad_weight = gy[0].type().zeros(ctx->weight.sizes());
                    grad_bias = gy[0].type().zeros(ctx->bias.sizes());
                    at::conv2d_backward_out(grad_input, grad_weight, grad_bias, grad_output,
                                            x, ctx->weight.data(), ctx->kernel_size, ctx->stride, ctx->padding,
                                            ctx->finput, ctx->fgrad_input);

                    if (ctx->weight.grad().defined())
                        ctx->weight.ptr->grad += grad_weight;
                    else ctx->weight.ptr->grad = grad_weight;

                    if (ctx->bias.grad().defined())
                        ctx->bias.ptr->grad += grad_bias;
                    else ctx->bias.ptr->grad = grad_bias;
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
