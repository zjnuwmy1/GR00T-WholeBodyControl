#include "InferenceEngine.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace {

DataType FromOnnxType(ONNXTensorElementDataType t)
{
    switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return DataType::FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DataType::HALF;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return DataType::INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return DataType::INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return DataType::BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return DataType::UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return DataType::INT64;
    default:                                    return DataType::UNKNOWN;
    }
}

size_t ElementSize(DataType t)
{
    switch (t) {
    case DataType::FLOAT: case DataType::INT32:            return 4;
    case DataType::HALF:                                   return 2;
    case DataType::INT8: case DataType::BOOL: case DataType::UINT8: return 1;
    case DataType::INT64:                                  return 8;
    default:                                               return 0;
    }
}

}  // namespace

bool ConvertONNXToTRT(
    const Options& /*options*/,
    const std::string& onnxModelPath,
    std::string& generatedTRTFile,
    const std::string /*prefix*/,
    bool /*forceConvert*/)
{
    // ORT loads the .onnx as-is; there is no engine-build stage to run or cache.
    generatedTRTFile = onnxModelPath;
    return !onnxModelPath.empty();
}

// One tensor's name, resolved shape, dtype and host-side staging buffer.
struct Binding
{
    std::string          name;
    std::vector<int64_t> shape;
    DataType             dtype = DataType::UNKNOWN;
    size_t               element_count = 0;
    std::vector<uint8_t> buffer;
};

class OrtInferenceEngine::Impl
{
public:
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "g1_deploy_ort"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo     memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Binding> inputs;
    std::vector<Binding> outputs;
    // Name -> index, so the string-keyed API stays O(1) on the 50 Hz path.
    std::unordered_map<std::string, size_t> input_index;
    std::unordered_map<std::string, size_t> output_index;

    bool initialized = false;

    Binding* FindInput(const std::string& name)
    {
        auto it = input_index.find(name);
        return it == input_index.end() ? nullptr : &inputs[it->second];
    }
    Binding* FindOutput(const std::string& name)
    {
        auto it = output_index.find(name);
        return it == output_index.end() ? nullptr : &outputs[it->second];
    }
    const Binding* FindAny(const std::string& name) const
    {
        auto i = input_index.find(name);
        if (i != input_index.end()) return &inputs[i->second];
        auto o = output_index.find(name);
        if (o != output_index.end()) return &outputs[o->second];
        return nullptr;
    }
};

OrtInferenceEngine::OrtInferenceEngine() = default;
OrtInferenceEngine::~OrtInferenceEngine() { Destroy(); }

bool OrtInferenceEngine::Initialize(const std::string& modelPath, int /*deviceID*/, const Options::AxisNames& /*axisNames*/)
{
    try {
        m_impl = std::make_shared<Impl>();

        m_impl->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Cap the intra-op pool and disable spin-waiting. The first cut used
        // SetIntraOpNumThreads(0) (= one full-size pool per session, spinning
        // between inferences); with encoder+decoder+planner sessions live the
        // deploy pinned 21.5 of 24 logical cores, starving the MuJoCo sim and
        // the 50 Hz control thread, and planner inference degraded ~9x (28 ms
        // -> 255 ms) from self-inflicted contention. Measured on a 24-thread
        // Ryzen AI 9 HX 370 at 4 threads, spinning off: encoder 4.9 ms,
        // decoder 2.2 ms, planner 28 ms -- all comfortably inside their 20 /
        // 100 ms budgets. Override via WBC_ORT_INTRA_THREADS if a different
        // host wants a different trade.
        int intra_threads = 4;
        if (const char* env = std::getenv("WBC_ORT_INTRA_THREADS")) {
            const int v = std::atoi(env);
            if (v > 0) intra_threads = v;
        }
        m_impl->session_options.SetIntraOpNumThreads(intra_threads);
        m_impl->session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        m_impl->session_options.SetExecutionMode(ORT_SEQUENTIAL);

        m_impl->session = std::make_unique<Ort::Session>(
            m_impl->env, modelPath.c_str(), m_impl->session_options);

        const size_t n_in  = m_impl->session->GetInputCount();
        const size_t n_out = m_impl->session->GetOutputCount();

        // GetTensorTypeAndShapeInfo() returns a non-owning view into the TypeInfo,
        // so the TypeInfo must be a named local. Binding it to a temporary leaves
        // the view dangling and GetShape() then reads freed memory.
        for (size_t i = 0; i < n_in; ++i) {
            Binding b;
            auto name = m_impl->session->GetInputNameAllocated(i, m_impl->allocator);
            b.name = name.get();
            Ort::TypeInfo type_info = m_impl->session->GetInputTypeInfo(i);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            b.shape = info.GetShape();
            b.dtype = FromOnnxType(info.GetElementType());
            m_impl->input_index[b.name] = m_impl->inputs.size();
            m_impl->inputs.push_back(std::move(b));
        }
        for (size_t i = 0; i < n_out; ++i) {
            Binding b;
            auto name = m_impl->session->GetOutputNameAllocated(i, m_impl->allocator);
            b.name = name.get();
            Ort::TypeInfo type_info = m_impl->session->GetOutputTypeInfo(i);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            b.shape = info.GetShape();
            b.dtype = FromOnnxType(info.GetElementType());
            m_impl->output_index[b.name] = m_impl->outputs.size();
            m_impl->outputs.push_back(std::move(b));
        }

        m_impl->initialized = true;
        std::cout << "✓ OrtInferenceEngine loaded (CPU EP): " << modelPath
                  << "  inputs=" << n_in << " outputs=" << n_out << std::endl;
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "✗ OrtInferenceEngine::Initialize - " << e.what() << std::endl;
        m_impl.reset();
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "✗ OrtInferenceEngine::Initialize - " << e.what() << std::endl;
        m_impl.reset();
        return false;
    }
}

bool OrtInferenceEngine::InitInputs(const AxisSizes& axisSizes)
{
    if (!m_impl || !m_impl->initialized) {
        std::cerr << "✗ OrtInferenceEngine::InitInputs - not initialized" << std::endl;
        return false;
    }

    auto size_binding = [&](Binding& b) -> bool {
        // Resolve symbolic/dynamic extents (ORT reports them as -1). Prefer an
        // explicit override from axisSizes, else assume 1 -- the deploy models
        // are all batch-1 and every other axis is static.
        auto it = axisSizes.find(b.name);
        const int dyn = (it != axisSizes.end()) ? it->second : 1;
        for (auto& d : b.shape) {
            if (d < 0) d = dyn;
        }
        b.element_count = std::accumulate(b.shape.begin(), b.shape.end(),
                                          static_cast<size_t>(1), std::multiplies<size_t>());
        const size_t esz = ElementSize(b.dtype);
        if (esz == 0) {
            std::cerr << "✗ OrtInferenceEngine::InitInputs - unsupported dtype for '" << b.name << "'" << std::endl;
            return false;
        }
        b.buffer.assign(b.element_count * esz, 0);
        return true;
    };

    for (auto& b : m_impl->inputs)  { if (!size_binding(b)) return false; }
    for (auto& b : m_impl->outputs) { if (!size_binding(b)) return false; }
    return true;
}

void OrtInferenceEngine::Destroy()
{
    if (m_impl) {
        m_impl->session.reset();
        m_impl.reset();
    }
}

void OrtInferenceEngine::SetInputData(const std::string& name, const void* data, size_t byteCount)
{
    if (!m_impl) return;
    Binding* b = m_impl->FindInput(name);
    if (!b) {
        std::cerr << "✗ OrtInferenceEngine::SetInputData - unknown input '" << name << "'" << std::endl;
        return;
    }
    if (byteCount > b->buffer.size()) {
        std::cerr << "✗ OrtInferenceEngine::SetInputData - '" << name << "' expects "
                  << b->buffer.size() << " bytes, got " << byteCount << std::endl;
        return;
    }
    std::memcpy(b->buffer.data(), data, byteCount);
}

void OrtInferenceEngine::GetOutputData(const std::string& name, void* data, size_t byteCount)
{
    if (!m_impl) return;
    Binding* b = m_impl->FindOutput(name);
    if (!b) {
        std::cerr << "✗ OrtInferenceEngine::GetOutputData - unknown output '" << name << "'" << std::endl;
        return;
    }
    if (byteCount > b->buffer.size()) {
        std::cerr << "✗ OrtInferenceEngine::GetOutputData - '" << name << "' holds "
                  << b->buffer.size() << " bytes, asked for " << byteCount << std::endl;
        return;
    }
    std::memcpy(data, b->buffer.data(), byteCount);
}

void OrtInferenceEngine::SetInputDataAsync(const std::string& name, const void* data, size_t byteCount, cudaStream_t /*stream*/)
{
    SetInputData(name, data, byteCount);
}

void OrtInferenceEngine::GetOutputDataAsync(const std::string& name, void* data, size_t byteCount, cudaStream_t /*stream*/)
{
    GetOutputData(name, data, byteCount);
}

std::vector<std::string> OrtInferenceEngine::GetInputTensorNames() const
{
    std::vector<std::string> names;
    if (!m_impl) return names;
    names.reserve(m_impl->inputs.size());
    for (const auto& b : m_impl->inputs) names.push_back(b.name);
    return names;
}

std::vector<std::string> OrtInferenceEngine::GetOutputTensorNames() const
{
    std::vector<std::string> names;
    if (!m_impl) return names;
    names.reserve(m_impl->outputs.size());
    for (const auto& b : m_impl->outputs) names.push_back(b.name);
    return names;
}

bool OrtInferenceEngine::GetTensorShape(std::string name, std::vector<int64_t>& shape) const
{
    if (!m_impl) return false;
    const Binding* b = m_impl->FindAny(name);
    if (!b) return false;
    shape = b->shape;
    return true;
}

DataType OrtInferenceEngine::GetTensorDataType(std::string name) const
{
    if (!m_impl) return DataType::UNKNOWN;
    const Binding* b = m_impl->FindAny(name);
    return b ? b->dtype : DataType::UNKNOWN;
}

bool OrtInferenceEngine::Enqueue(cudaStream_t /*stream*/)
{
    if (!m_impl || !m_impl->initialized) {
        std::cerr << "✗ OrtInferenceEngine::Enqueue - not initialized" << std::endl;
        return false;
    }
    try {
        // Wrap the staging buffers as tensors. CreateTensor does not copy, so
        // this allocates no per-call storage on the 50 Hz path.
        std::vector<const char*> in_names, out_names;
        std::vector<Ort::Value>  in_values;
        in_names.reserve(m_impl->inputs.size());
        in_values.reserve(m_impl->inputs.size());

        for (auto& b : m_impl->inputs) {
            in_names.push_back(b.name.c_str());
            const size_t esz = ElementSize(b.dtype);
            ONNXTensorElementDataType ot;
            switch (b.dtype) {
            case DataType::FLOAT: ot = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;  break;
            case DataType::HALF:  ot = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; break;
            case DataType::INT8:  ot = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;   break;
            case DataType::INT32: ot = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;  break;
            case DataType::BOOL:  ot = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;   break;
            case DataType::UINT8: ot = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;  break;
            case DataType::INT64: ot = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;  break;
            default:
                std::cerr << "✗ OrtInferenceEngine::Enqueue - unsupported dtype for '" << b.name << "'" << std::endl;
                return false;
            }
            in_values.push_back(Ort::Value::CreateTensor(
                m_impl->memory_info, b.buffer.data(), b.element_count * esz,
                b.shape.data(), b.shape.size(), ot));
        }

        out_names.reserve(m_impl->outputs.size());
        for (auto& b : m_impl->outputs) out_names.push_back(b.name.c_str());

        auto results = m_impl->session->Run(
            Ort::RunOptions{nullptr},
            in_names.data(), in_values.data(), in_values.size(),
            out_names.data(), out_names.size());

        // Copy results into the staging buffers so GetOutputData can serve them
        // after Run() returns and `results` goes out of scope.
        for (size_t i = 0; i < m_impl->outputs.size() && i < results.size(); ++i) {
            Binding& b = m_impl->outputs[i];
            auto info = results[i].GetTensorTypeAndShapeInfo();
            const size_t n   = info.GetElementCount();
            const size_t esz = ElementSize(FromOnnxType(info.GetElementType()));
            const size_t bytes = n * esz;
            if (bytes > b.buffer.size()) b.buffer.resize(bytes);
            std::memcpy(b.buffer.data(), results[i].GetTensorRawData(), bytes);
            b.element_count = n;
        }
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "✗ OrtInferenceEngine::Enqueue - " << e.what() << std::endl;
        return false;
    }
}
