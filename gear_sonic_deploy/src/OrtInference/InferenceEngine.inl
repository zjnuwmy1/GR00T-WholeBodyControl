#pragma once

// Typed forwarders onto the void*/byte-count overloads -- mirrors
// TRTInference/InferenceEngine.inl one-for-one.

template<typename T>
void OrtInferenceEngine::SetInputData(const std::string& name, const T* data, size_t elementCount)
{
    SetInputData(name, static_cast<const void*>(data), sizeof(T) * elementCount);
}

template<typename T>
void OrtInferenceEngine::SetInputData(const std::string& name, const TPinnedVector<T>& data)
{
    SetInputData(name, static_cast<const void*>(data.data()), sizeof(T) * data.size());
}

template<typename T>
void OrtInferenceEngine::GetOutputData(const std::string& name, T* data, size_t elementCount)
{
    GetOutputData(name, static_cast<void*>(data), sizeof(T) * elementCount);
}

template<typename T>
void OrtInferenceEngine::GetOutputData(const std::string& name, TPinnedVector<T>& data)
{
    GetOutputData(name, static_cast<void*>(data.data()), sizeof(T) * data.size());
}

// The stream parameter is accepted and dropped: ORT's CPU provider has no
// asynchronous queue, so these are the synchronous paths under another name.
template<typename T>
void OrtInferenceEngine::SetInputDataAsync(const std::string& name, const T* data, size_t elementCount, cudaStream_t stream)
{
    SetInputDataAsync(name, static_cast<const void*>(data), sizeof(T) * elementCount, stream);
}

template<typename T>
void OrtInferenceEngine::SetInputDataAsync(const std::string& name, const TPinnedVector<T>& data, cudaStream_t stream)
{
    SetInputDataAsync(name, static_cast<const void*>(data.data()), sizeof(T) * data.size(), stream);
}

template<typename T>
void OrtInferenceEngine::GetOutputDataAsync(const std::string& name, T* data, size_t elementCount, cudaStream_t stream)
{
    GetOutputDataAsync(name, static_cast<void*>(data), sizeof(T) * elementCount, stream);
}

template<typename T>
void OrtInferenceEngine::GetOutputDataAsync(const std::string& name, TPinnedVector<T>& data, cudaStream_t stream)
{
    GetOutputDataAsync(name, static_cast<void*>(data.data()), sizeof(T) * data.size(), stream);
}
