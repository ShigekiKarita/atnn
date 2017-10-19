#include <atnn/autograd/variable.hpp>
#include <atnn/autograd/function.hpp>
#include <atnn/autograd/module.hpp>

#include <atnn/testing.hpp>

using namespace atnn::autograd;
// using atnn::autograd::Variable;

int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {

    Variable v0(device(at::kFloat).rand({3, 4}));
    Variable v1(device(at::kFloat).rand({3, 4}));
    Variable v2 = v1; // assign
    Variable v2c(v1); // copy ctor
    Variable v3(device(at::kFloat).rand({3, 4}));

    // reference semantics
    v2.ptr()->data[0][0] = -100;  // mutate
    ATNN_ASSERT_EQ(at::Scalar(v1.data()[0][0]).toDouble(), -100);
    ATNN_ASSERT_EQ(at::Scalar(v2c.data()[0][0]).toDouble(), -100);

    // test equality
    ATNN_ASSERT(v1 == v2);
    ATNN_ASSERT(v1 == v2c);
    ATNN_ASSERT(v0 != v1);
    ATNN_ASSERT(v0 != v2);

    // test hash
    auto hash = Variable::Hash();
    ATNN_ASSERT(hash(v1) == hash(v2));
    ATNN_ASSERT(hash(v1) == hash(v2c));
    ATNN_ASSERT(hash(v0) != hash(v1));
    ATNN_ASSERT(hash(v0) != hash(v2));

    // test unordered_map<Variable>
    Variable::Set set {v0, v1};
    ATNN_ASSERT_EQ(set.count(v2), 1); // v1 == v2
    ATNN_ASSERT_EQ(set.count(v2c), 1); // v1 == v2c
    ATNN_ASSERT_EQ(set.count(v3), 0); // not found

    Variable::Map<double> map {{v0, -1}, {v1, 1}};
    ATNN_ASSERT_EQ(map[v0], -1);
    ATNN_ASSERT_EQ(map[v1], 1);
    ATNN_ASSERT_EQ(map[v2], 1); // v1 == v2
    ATNN_ASSERT_EQ(map[v2c], 1); // v1 == v2c
    ATNN_ASSERT_EQ(map.count(v3), 0); // not found


    // FunctionPtr f0 = std::make_unique<Threshold>(0.0, 0.0);
    // FunctionPtr f1 = std::make_unique<Threshold>(0.0, 0.0);
    // auto f2 = f1;
    // FunctionPtr f3(f1);
    // FunctionPtr f4 = std::make_shared<ReLU>();
    // ATNN_ASSERT_EQ(f1, f2);
    // ATNN_ASSERT_EQ(f1, f3);
    // ATNN_ASSERT(f0 != f1);
    // ATNN_ASSERT(f0 != f2);
    // ATNN_ASSERT(f0 != f3);

    // std::unordered_set<FunctionPtr> fset = {f0, f1};
    // ATNN_ASSERT_EQ(fset.count(f2), 1);
    // ATNN_ASSERT_EQ(fset.count(f3), 1);
    // ATNN_ASSERT_EQ(fset.count(f4), 0);

    // std::unordered_map<FunctionPtr, std::unordered_map<int, Variable> > fmap
    // {
    //     {f0, {{0, v0}, {1, v1}}},
    //     {f1, {{2, v2}, {3, v3}}}
    // };
    // ATNN_ASSERT_EQ(fmap[f0][0], v0);
    // ATNN_ASSERT_EQ(fmap[f1][2], v2);
    // ATNN_ASSERT_EQ(fmap[f2][3], v3);
    // ATNN_ASSERT_EQ(fmap.count(f4), 0);

    });
}
