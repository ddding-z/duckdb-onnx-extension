//===----------------------------------------------------------------------===//
//                         DuckONNX
//
// duckonnx/core/utils/model_cache.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckonnx/common.hpp"
// #include "duckdb/common/vector.hpp"
#include <unordered_map>
#include <queue>
#include <shared_mutex>
#include <memory>
#include <string>
#include <iostream>
#include <mutex>
#include "onnxruntime_cxx_api.h"

namespace duckonnx {
namespace core {

class ModelCache {
public:
    static const size_t capacity = 10;
    static std::unordered_map<std::string, std::shared_ptr<Ort::Session>> modelcache;
    static std::queue<std::string> model_queue;
    static std::shared_mutex map_mutex;

    static std::shared_ptr<Ort::Session> getOrCreateSession(const std::string &key, const Ort::Env &env,
                                                            const Ort::SessionOptions &options);
};

} // namespace core
} // namespace duckonnx




