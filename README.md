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
