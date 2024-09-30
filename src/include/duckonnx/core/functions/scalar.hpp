#pragma once
#include "duckonnx/common.hpp"

namespace duckonnx {

namespace core {

struct CoreScalarFunctions {
public:
  static void Register(DatabaseInstance &db) {
    RegisterOnnxInferenceScalarFunction(db);
  }

private:
  static void RegisterOnnxInferenceScalarFunction(DatabaseInstance &db);

};

} // namespace core

} // namespace duckonnx
