#include "duckonnx/core/utils/model_cache.hpp"

namespace duckonnx {
namespace core {

std::unordered_map<std::string, std::shared_ptr<Ort::Session>> ModelCache::modelcache;
std::queue<std::string> ModelCache::model_queue;
std::shared_mutex ModelCache::map_mutex;

std::shared_ptr<Ort::Session> ModelCache::getOrCreateSession(const std::string &key, const Ort::Env &env,
                                                             const Ort::SessionOptions &options) {
    {
        std::shared_lock<std::shared_mutex> sharedLock(map_mutex);
        auto it = modelcache.find(key);
        if (it != modelcache.end()) {
            return it->second;
        }
    }

    std::unique_lock<std::shared_mutex> lock(map_mutex);
    auto it = modelcache.find(key);
    if (it != modelcache.end()) {
        return it->second;
    }

    std::shared_ptr<Ort::Session> session;
    try {
        session = std::make_shared<Ort::Session>(env, key.c_str(), options);
    } catch (const Ort::Exception &e) {
        std::cerr << "Failed to create session: " << e.what() << std::endl;
        return nullptr;
    }

    if (modelcache.size() >= capacity) {
        const std::string &old_key = model_queue.front();
        modelcache.erase(old_key);
        model_queue.pop();
    }
    model_queue.push(key);
    modelcache[key] = session;

    return session;
}

} // namespace core
} // namespace duckonnx
