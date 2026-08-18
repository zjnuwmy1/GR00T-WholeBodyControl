#pragma once

// ONNX Runtime (CPU execution provider) backend, API-compatible with
// TRTInference/InferenceEngine.h.
//
// Why this exists: PolicyEngine (control_policy.hpp) and EncoderEngine
// (encoder.hpp) hardcode TRTInferenceEngine, and TensorRT is CUDA-only. Mirroring
// the same surface here lets those two classes switch backend with a typedef
// instead of a rewrite, which keeps the fork mergeable with upstream.
//
// Semantic differences from the TensorRT engine, all intentional:
//   - There is no device. SetInputData copies into a host staging buffer,
//     Enqueue() runs the session synchronously, GetOutputData copies back out.
//   - The *Async overloads are exact aliases of the sync ones; the stream
//     argument is accepted and ignored so call sites need no #ifdef.
//   - Enqueue() BLOCKS until inference finishes. The TensorRT path relies on a
//     following cudaStreamSynchronize() to do that; here it has already happened,
//     and the stub cudaStreamSynchronize() below is a no-op.
//   - Precision::FP16 is accepted but ignored: ORT's CPU provider has no fast
//     fp16 kernels, so the fp32 graph is always used.

#include "Utility.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

enum class Precision
{
    FP32,
    FP16
};

enum class DataType
{
    FLOAT,
    HALF,
    INT8,
    INT32,
    BOOL,
    UINT8,
    INT64,
    UNKNOWN
};

struct Options
{
    using AxisNames = std::map< std::string, std::map<int, std::string> >;
    using AxisSizes = std::map<std::string, std::tuple<int, int, int> >;
    using ShapeTensorSizes = std::map<std::string, std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> >;

    Precision        precision = Precision::FP32;
    AxisNames        dynamic_axes_names;
    AxisSizes        dynamic_axes_sizes;
    ShapeTensorSizes shape_tensor_sizes;
    std::tuple<int, int, int> defaultSizes = { 1,8,16 };
    int              deviceID = 0;
};

// Signature-compatible stand-in for the TensorRT engine-build step. ONNX Runtime
// consumes the .onnx directly, so this only passes the path through -- there is
// no build cost and nothing is cached on disk.
bool ConvertONNXToTRT(
    const Options& options,
    const std::string& onnxModelPath,
    std::string& generatedTRTFile,
    const std::string prefix = "",
    bool forceConvert = false
);

// Stub CUDA surface so call sites keep their `cudaStream_t` signatures. Streams
// are always nullptr and every operation is a no-op returning success.
typedef struct CUstream_st *cudaStream_t;
typedef int cudaError_t;
constexpr cudaError_t cudaSuccess = 0;
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { if (s) *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t)   { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t)   { return "no CUDA (ONNX Runtime CPU backend)"; }

// Inert CUDA-graph surface. Capture is an optional latency optimisation with an
// Enqueue() fallback in both PolicyEngine::Infer() and EncoderEngine::Encode():
//   if (graph_captured_ && cuda_graph_exec_ != nullptr) { launch graph }
//   else                                                { Enqueue(stream) }
// These stubs deliberately leave the exec handle null, so CaptureGraph() runs to
// completion but every inference still takes the Enqueue() path. Nothing here
// silently changes numerics -- it only declines the fast path.
typedef struct CUgraph_st     *cudaGraph_t;
typedef struct CUgraphExec_st *cudaGraphExec_t;
enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1, cudaStreamCaptureModeRelaxed = 2 };
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { return cudaSuccess; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* g) { if (g) *g = nullptr; return cudaSuccess; }
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t, void*, void*, unsigned long long) { if (e) *e = nullptr; return cudaSuccess; }
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t)         { return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }

class OrtInferenceEngine
{
public:
    OrtInferenceEngine();
    ~OrtInferenceEngine();

    using AxisSizes = std::map<std::string, int >;
    bool Initialize(const std::string& modelPath, int deviceID, const Options::AxisNames& axisNames = {});
    bool InitInputs(const AxisSizes& axisSizes = {});
    void Destroy();

    void SetInputData(const std::string& name, const void* data, size_t byteCount);
    template<typename T> void SetInputData(const std::string& name, const T* data, size_t elementCount);
    template<typename T> void SetInputData(const std::string& name, const TPinnedVector<T>& data);

    void GetOutputData(const std::string& name, void* data, size_t byteCount);
    template<typename T> void GetOutputData(const std::string& name, T* data, size_t elementCount);
    template<typename T> void GetOutputData(const std::string& name, TPinnedVector<T>& data);

    void SetInputDataAsync(const std::string& name, const void* data, size_t byteCount, cudaStream_t stream);
    template<typename T> void SetInputDataAsync(const std::string& name, const T* data, size_t elementCount, cudaStream_t stream);
    template<typename T> void SetInputDataAsync(const std::string& name, const TPinnedVector<T>& data, cudaStream_t stream);

    void GetOutputDataAsync(const std::string& name, void* data, size_t byteCount, cudaStream_t stream);
    template<typename T> void GetOutputDataAsync(const std::string& name, T* data, size_t elementCount, cudaStream_t stream);
    template<typename T> void GetOutputDataAsync(const std::string& name, TPinnedVector<T>& data, cudaStream_t stream);

    std::vector<std::string> GetInputTensorNames() const;
    std::vector<std::string> GetOutputTensorNames() const;

    bool GetTensorShape(std::string name, std::vector<int64_t>& shape) const;
    DataType GetTensorDataType(std::string name) const;
    bool Enqueue(cudaStream_t stream);

private:
    class Impl;
    std::shared_ptr<Impl> m_impl = nullptr;
};

#include "InferenceEngine.inl"
