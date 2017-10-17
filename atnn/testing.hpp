#pragma once
#include <ATen/ATen.h>
#include <iostream>
#include <chrono>


#ifdef HAVE_BOOST // a BOOST dependent part
// BOOST_ENABLE_ASSERT_DEBUG_HANDLER is defined for the whole project
#include <boost/stacktrace.hpp>
#include <boost/format.hpp>


namespace boost {
    inline void assertion_failed_msg(char const* expr, char const* msg, char const* function, char const* file, long line) {
        std::cerr << "===== ATNN-Assetion failed =====\n"
                  << "Expression: " << expr << "\n"
                  << "Function:   " << function << " in " << file << "(" << line << ")\n"
                  << "Message:    " << (msg ? msg : "<none>") << "\n"
                  << "\n"
                  << "=========== Backtrace ==========\n"
                  << boost::stacktrace::stacktrace() << '\n';
        std::abort();
    }

    inline void assertion_failed(char const* expr, char const* function, char const* file, long line) {
        ::boost::assertion_failed_msg(expr, 0 /*nullptr*/, function, file, line);
    }
} // namespace boost

namespace atnn {
    typedef boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace> traced;
    template <class E>
    void throw_with_trace(const E& e) {
        throw boost::enable_error_info(e)
            << traced(boost::stacktrace::stacktrace());
    }
}

#define ATNN_ASSERT BOOST_ASSERT
#define ATNN_ASSERT_MSG BOOST_ASSERT_MSG
#define ATNN_ASSERT_EQ(a, b) do { BOOST_ASSERT_MSG((a) == (b), (boost::format("%d != %d") % (a) % (b)).str().c_str()); } while (0)
#define ATNN_ASSERT_SHAPE_EQ(a, b) do { \
        ATNN_ASSERT_MSG(atnn::shape_eq((a), (b)),                      \
                         (boost::format("%1 != %2") % atnn::to_tensor(a) % atnn::to_tensor(b)).str().c_str()); } while (0)

#else // not HAVE_BOOST

#define ATNN_ASSERT assert
#define ATNN_ASSERT_MSG(expr, msg) do { assert((expr) && (msg)); } while(0)
#define ATNN_ASSERT_EQ(a, b) do { ATNN_ASSERT((a) == (b)); } while (0)
#define ATNN_ASSERT_SHAPE_EQ(a, b) do { ATNN_ASSERT(atnn::shape_eq((a), (b))); } while (0)

namespace atnn {
    template <class E>
    void throw_with_trace(const E& e) {
        throw e;
    }
}

#endif


namespace atnn {

    bool is_empty(at::Tensor t) {
        return !t.defined() || t.dim() == 0;
    }

    template <typename dtype=float>
    bool allclose(at::Tensor actual, at::Tensor desired, float rtol=1e-7, dtype atol=0) {
        ATNN_ASSERT(!atnn::is_empty(actual));
        ATNN_ASSERT(!atnn::is_empty(desired));
        return ((actual - desired).abs() <= desired.abs() * rtol + atol).all();
    }

    inline static auto to_tensor(at::IntList a) {
        return CPU(at::kLong).tensorFromBlob(const_cast<long*>(a.begin()), {static_cast<long>(a.size())});
    }

    template <typename T>
    bool shape_is(T t, at::IntList shape) {
        auto a = to_tensor(t.sizes());
        auto b = to_tensor(shape);
        bool ok = a.size(0) == b.size(0) && (a == b).all();
        if (!ok) std::cerr << "shape does not match:\n  lhs=" << a << "\n  rh=" << b << std::endl;
        return ok;
    }

    template <typename T1, typename T2>
    bool shape_eq(T1 t1, T2 t2) {
        auto a = to_tensor(t1);
        auto b = to_tensor(t2);
        bool ok = a.size(0) == b.size(0) && (a == b).all();
        return ok;
    }

    template <typename F>
    void test_common(int argc [[gnu::unused]], char** argv, F proc, bool cpu_only=false) {
        std::cout << argv[0] << std::flush;
        for (auto device: {at::CPU, at::CUDA}){
            if (cpu_only && device == at::CPU) continue;
            auto start_time = std::chrono::high_resolution_clock::now();
            proc(device);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto elapsed = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
            std::cout << ", "<< at::toString(device(at::kFloat).backend())
                      << ": " << elapsed << " sec" << std::flush;
        }
        std::cout << std::endl;
    }

} // namespace atnn
