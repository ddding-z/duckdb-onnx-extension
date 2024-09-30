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

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <string.h>
#include <vector>
#include <unordered_map>
#include <functional>

namespace duckonnx {

namespace core {

class OnnxPruningExtension : public OptimizerExtension {
public:
	OnnxPruningExtension();

	static std::string onnx_model_path;
	static std::string new_model_path;
	static float_t predicate;
	static std::vector<std::string> removed_nodes;

	static std::vector<int64_t> left_nodes;
	static std::vector<int64_t> right_nodes;
	static std::vector<std::string> node_types;
	static std::vector<double> node_thresholds;
	static std::vector<int64_t> target_nodeids;
	static std::vector<double> target_weights;

	static ExpressionType ComparisonOperator;
	static std::unordered_map<ExpressionType, std::function<bool(float_t, float_t)>> comparison_funcs;

	struct NodeID {
		int id;
		std::string node;
	};

	// TODO: need to support nested case
	static bool HasONNXFilter(LogicalOperator &op);

	static int pruning(size_t node_id, size_t depth, std::vector<std::string> &result_nodes,
	                   onnx::graph_node_list &node_list, float_t predicate);

	static void OnnxPruneFunction();

	static void reg2reg(std::string &model_path, onnx::graph_node_list &node_list);
	// constrcut pruned model
	static void OnnxConstructFunction(); 
	static void OnnxPruningFunction(OptimizerExtensionInput &input, duckdb::unique_ptr<LogicalOperator> &plan);
};


} // namespace core

} // namespace duckonnx