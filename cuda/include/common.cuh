#ifndef MNV2_CUDA_COMMON_CUH
#define MNV2_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(                                                \
          std::string("CUDA error at ") + __FILE__ + ":" +                     \
          std::to_string(__LINE__) + " - " + cudaGetErrorString(err));         \
    }                                                                          \
  } while (0)

#endif  // MNV2_CUDA_COMMON_CUH
