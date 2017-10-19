#pragma once

#include "fwd.hpp"

#include <unordered_map>
#include <unordered_set>

namespace atnn {
    namespace autograd {
        struct VariableImpl {
            at::Tensor data, grad;
            VariableImpl(at::Tensor data) : data(data) {}
        };

        struct Variable : VariableBase, HashableMixin<Variable> {
            bool train = true;
            std::shared_ptr<VariableImpl> ptr_;
            std::vector<std::shared_ptr<Variable>> ancestors;
            // FunctionPtr func;
            long arg;

            auto& ptr() {
                return this->ptr_;
            }

            const auto& ptr() const {
                return this->ptr_;
            }
            Variable() {}

            Variable(at::Tensor data, bool train=true)
                : train(train), ptr_(std::make_shared<VariableImpl>(data)) {}

            Variable& operator=(const Variable&) = default;

            bool is_leaf() const {
                return this->ancestors.size() == 0;
            }

            auto data() const {
                return this->ptr_->data;
            }

            auto grad() const {
                return this->ptr_->grad;
            }

            auto sizes() const {
                return this->ptr_->data.sizes();
            }

            void clear_grads() {
                this->ptr_->grad = at::Tensor();
                if (!this->is_leaf()) {
                    for (auto& v: this->ancestors) {
                        v->clear_grads();
                    }
                }
            }

            virtual void toBackend(at::Backend b) {
                this->ptr_->data = this->data().toBackend(b);
                if (!is_empty(this->grad())) {
                    this->ptr_->grad = this->grad().toBackend(b);
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
    } // namespace autograd
} // namespace atnn
