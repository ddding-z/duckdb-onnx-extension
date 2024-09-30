#pragma once

#include "duckonnx/common.hpp"

namespace duckonnx {
namespace core {

struct CoreOnnxOptimizer {
  static void Register(DatabaseInstance &db) {
    RegisterOnnxSkl2onnxOptimizer(db);
    RegisterOnnxPruningOptimizer(db);
    RegisterOnnxPushdownOptimizer(db);
  }

private:
  static void RegisterOnnxSkl2onnxOptimizer(DatabaseInstance &db);
  static void RegisterOnnxPruningOptimizer(DatabaseInstance &db);
  static void RegisterOnnxPushdownOptimizer(DatabaseInstance &db);

};

} // namespace core

} // namespace duckonnx