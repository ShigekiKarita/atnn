#pragma once

#include <assert.h>
#include <vector>
#include <memory>
#include <iostream>

#include <ATen/ATen.h>


namespace atnn {

    struct Variable;
    using VariablePtr = std::shared_ptr<Variable>;

    // struct VariablePtr {
    //     std::shared_ptr<Variable> ptr;

    //     VariablePtr(at::Tensor data) {
    //         this->ptr = std::make_shared<Variable>(data);
    //     }

    //     auto operator->() {
    //         return ptr;
    //     }
    // };

    using VList = std::vector<VariablePtr>;
    using TList = std::vector<at::Tensor>;


    struct ModuleBase : std::enable_shared_from_this<ModuleBase> {
        VList vargs, vrets;
        virtual TList backward(TList grads) = 0;
    };
    using ModulePtr = std::shared_ptr<ModuleBase>;


    bool is_empty(at::Tensor t) {
        return !t.defined() || t.dim() == 0;
    }


    struct Variable {
        at::Tensor data, grad;
        ModulePtr module;
        bool train = true;

        Variable(at::Tensor t, bool train=true) : data(t), train(train) {}

        bool is_leaf() const { return this->module == nullptr; }

        auto& children() { return this->module->vargs; }

        void clear_grads() {
            this->grad = at::Tensor();
            if (!this->is_leaf()) {
                for (auto& v: this->children()) {
                    v->clear_grads();
                }
            }
        }

        auto backward(at::Tensor grad) {
            if (is_empty(this->grad)) {
                this->grad = grad.clone();
            } else {
                this->grad += grad;
            }
            if (this->is_leaf()) return;

            VList& brothers = this->module->vrets;
            for (auto&& b: brothers) {
                if (is_empty(b->grad)) return;
            }

            TList accumulated_grads;
            accumulated_grads.reserve(brothers.size());
            for (auto&& b: brothers) {
                accumulated_grads.push_back(b->grad);
            }

            TList next_grads = this->module->backward(accumulated_grads);
            assert(next_grads.size() == this->children().size());
            for (size_t i = 0; i < next_grads.size(); ++i) {
                this->children()[i]->backward(next_grads[i]);
            }
        }
    };

    std::ostream& operator<<(std::ostream &strm, const Variable &v) {
        return strm << "Variable(\n"
                    << "data=\n" << v.data
                    << "\ngrad=\n" << v.grad
                    << "\n)";
    }

    std::ostream& operator<<(std::ostream &strm, const VariablePtr &v) {
        return strm << "VariablePtr(\n"
                    << "data=\n" << v->data
                    << "\ngrad=\n" << v->grad
                    << "\n)";
    }


/**
   Module stores Parameters and Variables for Derived::Impl (static class with forward/backward functions)
*/
    template <class Derived>
    struct Module : ModuleBase {
        TList saved_tensors;

        virtual ~Module() {}

        template <class T>
        auto set_vargs(T&& t) { return t; }

        auto set_vargs(VariablePtr v) {
            this->vargs.push_back(v);
            return v->data;
        }

        auto set_vrets(at::Tensor t) {
            auto v = std::make_shared<Variable>(t);
            v->module = shared_from_this();
            this->vrets.clear();
            this->vrets.push_back(v);
            return v;
        }

        auto set_vrets(TList ts) {
            this->vrets.clear();
            this->vrets.reserve(ts.size());
            for (auto&& t: ts) {
                auto v = Variable(t);
                v.module = shared_from_this();
                this->vrets.push_back(v);
            }
            return this->vrets;
        }

        template <class ... Args>
        auto forward(Args ... args) {
            return set_vrets(Derived::Impl::forward(static_cast<Derived*>(this),
                                                    {this->set_vargs(args)...}));
        }

        virtual TList backward(TList grads) {
            return Derived::Impl::backward(static_cast<Derived*>(this), grads);
        }

        template <class ... Args>
        auto operator()(Args ... args) {
            return this->forward(args...);
        }

        void save_for_backward(TList tensors){
            bool train = true;
            for (auto&& v: this->vargs) {
                train &= v->train;
                if (!train) return;
            }
            this->saved_tensors = tensors;
        }
    };

} // namespace atnn
