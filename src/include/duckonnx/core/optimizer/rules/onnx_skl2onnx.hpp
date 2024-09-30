#pragma once

#include "duckonnx/common.hpp"

#include "duckdb/optimizer/optimizer_extension.hpp"

#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/logical_operator.hpp"

#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"


#include <string.h>

namespace duckonnx {

namespace core {


class OnnxSkl2onnxExtension : public OptimizerExtension {
public:
	OnnxSkl2onnxExtension();

	static std::string convertModel(std::string &model_path);

	static bool HasONNXExpressionScan(Expression &expr);

	static bool HasONNXScan(LogicalOperator &op);

	static void OnnxSkl2onnxFunction(OptimizerExtensionInput &input,
	                             duckdb::unique_ptr<LogicalOperator> &plan);
};


} // namespace core

} // namespace duckonnx