set(SRC_FILES
    "InferenceEngine.cpp"
    "InferenceEngine.h"
    "InferenceEngine.inl"
    "Utility.h"
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

add_library(OrtInference ${SRC_FILES})

set_target_properties(OrtInference PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(OrtInference
    PRIVATE ${PROJECT_SOURCE_DIR}/src
    PUBLIC  ${onnxruntime_INCLUDE_DIRS})

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(OrtInference PRIVATE -Wno-deprecated-declarations -std=c++20)
endif()

target_link_libraries(OrtInference onnxruntime::onnxruntime)
