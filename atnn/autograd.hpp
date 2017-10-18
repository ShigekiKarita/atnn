#pragma once

#include <assert.h>
#include <vector>
#include <memory>
#include <iostream>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <queue>

#include <ATen/ATen.h>

#include "testing.hpp"

namespace atnn {
    struct Variable;
    using VList = std::vector<Variable>;
    using TList = std::vector<at::Tensor>;

    struct ModuleBase : std::enable_shared_from_this<ModuleBase> {
        TList saved_tensors;
        VList vargs, vrets;
        virtual TList backward(TList grads) = 0;
        virtual void toBackend(at::Backend b) = 0;
    };

    struct VariableImpl {
        at::Tensor data, grad;
        VariableImpl(at::Tensor data) : data(data) {}
    };

    using ModulePtr = std::shared_ptr<ModuleBase>;

    struct Variable {
        bool train = true;
        std::shared_ptr<VariableImpl> ptr;
        ModulePtr module;

        struct Hash {
            size_t operator()(const Variable& v) const {
                return reinterpret_cast<size_t>(v.ptr.get()) / sizeof(VariableImpl);
            }
        };

        struct Equal {
            bool operator()(const Variable& a, const Variable& b) const {
                return b.ptr.get() == a.ptr.get();
            }
        };

        bool operator==(const Variable& that) const {
            return Equal()(*this, that);
        }

        bool operator!=(const Variable& that) const {
            return !Equal()(*this, that);
        }

        using Set = std::unordered_set<Variable, Hash, Equal>;

        template <typename Value>
        using Map = std::unordered_map<Variable, Value, Hash, Equal>;

        Variable() {}

        Variable(at::Tensor data, bool train=true)
            : train(train), ptr(std::make_shared<VariableImpl>(data)) {}

        Variable& operator=(const Variable&) = default;

        auto data() const {
            return this->ptr->data;
        }

        auto grad() const {
            return this->ptr->grad;
        }

        auto sizes() const {
            return this->ptr->data.sizes();
        }

        void clear_grads() {
            this->ptr->grad = at::Tensor();
            if (!this->is_leaf()) {
                for (auto& v: this->children()) {
                    v.clear_grads();
                }
            }
        }

        auto& set_module(ModulePtr m) {
            this->module = m;
            return *this;
        }

        bool is_leaf() const { return this->module == nullptr; }

        VList& children() { return this->module->vargs; }

        VList& brothers() { return this->module->vrets; }

        auto backward(at::Tensor grad) {
            ATNN_ASSERT_SHAPE_EQ(this->sizes(), grad.sizes());
            if (is_empty(this->ptr->grad)) {
                this->ptr->grad = grad.clone();
            } else {
                this->ptr->grad += grad;
            }

            if (this->is_leaf()) return; // stop the recursion

            for (auto&& b: this->brothers()) {
                if (is_empty(b.grad())) return;
            }

            TList accumulated_grads;
            accumulated_grads.reserve(this->brothers().size());
            for (auto&& b: this->brothers()) {
                accumulated_grads.push_back(b.grad());
            }

            TList next_grads = this->module->backward(accumulated_grads);
            ATNN_ASSERT_EQ(next_grads.size(), this->children().size());
            for (size_t i = 0; i < next_grads.size(); ++i) {
                this->children()[i].backward(next_grads[i]);
            }
        }

        auto backward() {
            
        }
    };

    

    std::ostream& operator<<(std::ostream &strm, const Variable &v) {
        return strm << "Variable(\n"
                    << "data=\n" << v.data()
                    << "\ngrad=\n" << v.grad()
                    << "\n)";
    }

    template <typename T1, typename T2>
    static void to_backend_of(T1& src, const T2& dst) {
        const auto src_backend = src.type().backend();
        const auto dst_backend = dst.type().backend();
        if (src_backend != dst_backend) {
            src = src.toBackend(dst_backend);
        }
    }

    template <typename T>
    static void to_backend_of(T& src, at::Backend b) {
        const auto src_backend = src.type().backend();
        if (src_backend != b) {
            src = src.toBackend(b);
        }
    }

    struct ModuleSet : atnn::ModuleBase {
        std::vector<atnn::ModulePtr> modules;

        void toBackend(at::Backend b) override {
            for (auto& m: this->modules) {
                m->toBackend(b);
            }
        }

        virtual TList backward(TList grads [[gnu::unused]]) {
            // FIXME: find better way.
            ATNN_ASSERT_MSG(false, "never call this");
            return {};
        }
    };

/**
   Module stores Parameters and Variables
   for Derived::Function (static class with forward/backward functions)
*/
    template <class Derived>
    struct Module : ModuleBase {
        VList parameters;
        std::vector<ModulePtr> submodules;

        Derived* dthis = static_cast<Derived*>(this);

        virtual ~Module() {}

        template <class T>
        auto set_vargs(T&& t) { return t; }

        auto set_vargs(Variable v) {
            this->vargs.push_back(v);
            return v.data();
        }

        auto set_vrets(at::Tensor t) {
            auto v = Variable(t).set_module(shared_from_this());
            this->vrets.push_back(v);
            return v;
        }

        auto set_vrets(TList ts) {
            this->vrets.reserve(ts.size());
            for (auto&& t: ts) {
                this->vrets.push_back(
                    Variable(t).set_module(shared_from_this()));
            }
            return this->vrets;
        }

        template <class ... Args>
        auto forward(Args ... args) {
            // FIXME: cannot backward through multiple applied module like RNN because of clear.
            // TODO: make this struct Edge { VList vargs, vrets; }
            this->vargs.clear();
            this->vrets.clear();
            return set_vrets(Derived::Function::forward(dthis, {this->set_vargs(args)...}));
        }

        TList backward(TList grads) override {
            return Derived::Function::backward(dthis, grads);
        }

        void save_for_backward(TList tensors){
            bool train = true;
            for (auto&& v: this->vargs) {
                train &= v.train;
                if (!train) return;
            }
            this->saved_tensors = tensors;
        }

        void toBackend(at::Backend b) override {
            for (auto& p: this->parameters) {
                p.ptr->data = p.data().toBackend(b);
                if (!is_empty(p.grad())) {
                    p.ptr->grad = p.grad().toBackend(b);
                }
            }
            for (auto& t: this->saved_tensors) {
                t = t.toBackend(b);
            }
            for (auto& m: this->submodules) {
                m->toBackend(b);
            }
        }
    };

} // namespace atnn


