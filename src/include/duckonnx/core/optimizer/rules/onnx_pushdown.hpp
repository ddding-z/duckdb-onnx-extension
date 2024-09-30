#pragma once

#include "duckonnx/common.hpp"

#include "onnx/checker.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"

#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"
#include "onnxoptimizer/pass_manager.h"
#include "onnxoptimizer/pass_registry.h"

#include "duckdb/optimizer/optimizer_extension.hpp"

#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <string.h>
#include <vector>

namespace duckonnx {

namespace core {


class OnnxPushdownExtension : public OptimizerExtension {
public:
	OnnxPushdownExtension();

	static std::string onnx_model_path;
	static std::string new_model_path;

	static void reg2reg(std::string &model_path, onnx::graph_node_list &node_list,
	                    std::vector<int64_t> &output_nodes_featureids, std::size_t dim_value);

	static std::set<idx_t> removeUnusedColumns(std::string &model_path, const std::vector<idx_t> &column_indexs,
	                                           onnx::graph_node_list &node_list);

	static std::set<idx_t> OnnxConstructFunction(const std::vector<idx_t> &column_indexs); 
	static bool HasONNXExpressionScan(Expression &expr);
	static bool HasONNXScan(LogicalOperator &op);
	// Optimizer Function
	static void OnnxPushdownFunction(OptimizerExtensionInput &input,
	                                  duckdb::unique_ptr<LogicalOperator> &plan);
};

} // namespace core

} // namespace duckonnx