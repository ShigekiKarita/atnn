#pragma once

#include "fwd.hpp"

namespace atnn {
    namespace autograd {

        template <class T>
        auto set_vargs(T&& t) { return t; }
        auto set_vargs(Variable v) {
            return v.data();
        }

        auto set_vrets(at::Tensor t) {
            return Variable(t);
        }
        auto set_vrets(TList ts) {
            VList vrets;
            vrets.reserve(ts.size());
            for (auto&& t: ts) {
                vrets.push_back(std::make_shared<Variable>(t));
            }
            return vrets;
        }

        struct Module : ModuleBase, std::enable_shared_from_this<Module>, HashableMixin<Module> {
            TList saved_tensors;
            bool train;
            VList parameters;
            std::vector<ModulePtr> submodules;

            virtual VList forward(VList xs) = 0;

            void save_for_backward(TList tensors) {
                if (train) this->saved_tensors = tensors;
            }

            template <class ... Args>
            auto operator()(Args ... args) {
                return set_vrets(this->forward({set_vargs(args)...}));
            }

            virtual void toBackend(at::Backend b) {
                for (auto& p: this->parameters) {
                    p->toBackend(b);
                }
                for (auto& t: this->saved_tensors) {
                    t = t.toBackend(b);
                }
                for (auto& m: this->submodules) {
                    m->toBackend(b);
                }
            }
        };

    } // namespace autograd
} // namespace atnn
