
#include "duckonnx/core/module.hpp"
#include "duckonnx/common.hpp"
#include "duckonnx/core/functions/scalar.hpp"
#include "duckonnx/core/optimizer/duckonnx_optimizer.hpp"

namespace duckonnx {

namespace core {

void CoreModule::Register(DatabaseInstance &db) {
    CoreScalarFunctions::Register(db);
    CoreOnnxOptimizer::Register(db);
}


} // namespace core

} // namespace duckonnx