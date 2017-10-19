#pragma once

#include <unordered_map>
#include <unordered_set>
#include <memory>

#include <ATen/ATen.h>
#include "../testing.hpp"

namespace atnn {
    namespace autograd {
        template <typename T>
        struct HashableMixin  {
            struct Hash {
                size_t operator()(const T& t) const {
                    return reinterpret_cast<size_t>(t.ptr().get()) / sizeof(decltype(*t.ptr()));
                }
            };

            struct Equal {
                bool operator()(const T& a, const T& b) const {
                    return b.ptr().get() == a.ptr().get();
                }
            };

            using Set = std::unordered_set<T, Hash, Equal>;

            template <typename Value>
            using Map = std::unordered_map<T, Value, Hash, Equal>;

        private:
            friend bool operator==(const T& t, const T& that) {
                return Equal()(t, that);
            }

            friend bool operator!=(const T& t, const T& that) {
                return !Equal()(t, that);
            }
        };

        struct VariableBase;
        using VariablePtr = std::shared_ptr<VariableBase>;
        struct VariableBase {
            virtual void toBackend(at::Backend) = 0;
        };
        using VList = std::vector<VariablePtr>;
        using TList = std::vector<at::Tensor>;

        struct ModuleBase {
            virtual TList saved_tensors() = 0;
            virtual void save_for_backward(TList tensors) = 0;
            virtual void toBackend(at::Backend) = 0;
        };
        using ModulePtr = std::shared_ptr<ModuleBase>;

        struct Function {
            virtual TList forward(ModulePtr, TList) const { return {}; }
            virtual TList backward(ModulePtr, TList) const { return {}; }
        };
        using FunctionPtr = std::unique_ptr<Function>;
    }
}
