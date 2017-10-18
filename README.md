# ATNN

computational graph library for [ATEN](https://github.com/zdevito/ATen)

## TODO

+ computational graph
  + support RNN and multiple backwards
+ gradient check utility
  + check Conv2d (in `test/test_nn.cpp`) -> DONE
  + check RNN
+ optimizers
  + add `std::vector<Variable> Module<Impl>.parameters` to track trainable variables
  + support double backward?
+ serialization
  + HDF5? https://support.hdfgroup.org/HDF5/doc/cpplus_RM/examples.html
  + header-only HDF5 wrapper https://github.com/BlueBrain/HighFive
+ try mnist with TensorDataset and DatasetIterator in `ATen/src/data`
+ CUDNN support
  + use pytorch functions https://github.com/pytorch/pytorch/blob/master/torch/csrc/cudnn/Conv.h


## Concepts

+ Tensor: is provided by ATen and equals to Torch's Tensor.
+ Variable: owns a tape of computational graphs and Tensors as data/grad like PyTorch.
+ Function: owns `static std::array<Tensor, N> forward/backward(Context, Tensor...) const` functions without any states.
+ Module: owns some Variable as trainable parameters and modules, `std::array<Variable, M> forward(Variable...)`


## brief algorithm of backprop

forward part

1. pick some input Variables `v1_1, v1_2, ...`
2. apply a function `{t2_1, t2_2, ...} = f1.forward(v1_1.data, v1_2.data, ...)` via a module
3. set `v2_1 = {ancestors: {1: v1_1, 2: v1_2, ...}, func: f1, arg: 1, data: t2_1}`, `v2_2 = {ancestors: {1: v1_1, 2: v1_2, ...}, func: f1, arg: 2, data: t2_2}`, ...

backward part

(init)
1. pick one `Variable v1` with its known `v1.grad` (e.g., a loss is a good start because it always has `v1.grad=1`)
2. compute all the grads `{a.grad | a in v1.ancestors} += v1.func.backward(v1.arg: v1.grad)`
3. build `fdict = { a.func: {a.arg: a} | a in v1.ancestors }`

(loop)
1. pick functions from `fs = {f | f in fdict if fdict[f].size() == func.n_args }`
2. exit if `fs = {}`
3. compute all the grads again `new_vars = sum { {a.grad | a in v.ancestors } += f.backward(v.arg: v.grad) | v in fdict[f], f in fs }`
4. update new ones `{ fdict[v.func][v.arg] = v | v in new_vars }`
5. remove completed ones `fdict.remove(f) in fs`
6. loop to first
 

## test

``` console
# if you do not have Boost >= 1.65.0
make test

# with Boost.StackTrace for better error messages
make USE_BOOST=true test
```

## ATen installation guide

see the pytorch's instruction


``` console
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install numpy pyyaml mkl setuptools cmake cffi

# Add LAPACK support for the GPU
conda install -c soumith magma-cuda80 # or magma-cuda75 if CUDA 7.5


# install ATen
rehash
git clone https://github.com/zdevito/ATen.git
cd ATen
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/stage
make install -j4
```

update your bashrc and `exec -l $SHELL`

``` bash
export ATEN_ROOT=<where you download>/build/stage
export C_INCLUDE_PATH=$ATEN_ROOT/include:$C_INCLUDE_PATH
export LD_LIBARRY_PATH=$ATEN_ROOT/lib64/$LD_LIBARARY_PATH
```

TODO: do not build by yourself. use anaconda installed libTH*.so
(e.g., anaconda3/envs/dnn/lib/python3.6/site-packages/torch/lib )

### known error

build again without `-j4` or patch this if you failed to build `lib/THC/generic/THCTensorMathPairwise.cu`


``` diff
diff --git a/lib/THC/generic/THCTensorMathPairwise.cu b/lib/THC/generic/THCTensorMathPairwise.cu
index e14df07..68364e2 100644
--- a/lib/THC/generic/THCTensorMathPairwise.cu
+++ b/lib/THC/generic/THCTensorMathPairwise.cu
@@ -44,7 +44,7 @@ THC_API void
 THCTensor_(add_scaled)(THCState *state, THCTensor *self_, THCTensor *src_, real value, real alpha)
 {
 #ifdef THC_REAL_IS_HALF
-  auto v = THC_half2float(value) * THC_half2float(alpha);
+  float v = THC_half2float(value) * THC_half2float(alpha);
   THCTensor_(add)(state, self_, src_, THC_float2half(v));
 #else
   THCTensor_(add)(state, self_, src_, value * alpha);
```
