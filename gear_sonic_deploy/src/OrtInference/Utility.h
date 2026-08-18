#pragma once

#include <vector>

// CPU-backend counterpart of TRTInference/Utility.h.
//
// The TensorRT path allocates I/O buffers with cudaMallocHost so the GPU can
// DMA straight out of page-locked host memory. ONNX Runtime's CPU execution
// provider reads the tensor in place on the host, so there is nothing to pin --
// a plain std::vector is both sufficient and one allocation cheaper.
//
// Keeping the alias name identical lets encoder.hpp / control_policy.hpp share
// one set of buffer declarations across both backends.
template<typename T>
using TPinnedVector = std::vector<T>;
