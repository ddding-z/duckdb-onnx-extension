#include "duckonnx/core/optimizer/rules/onnx_pushdown.hpp"
#include "duckonnx/core/optimizer/duckonnx_optimizer.hpp"

#include "duckonnx/common.hpp"

#include <iostream>

#include <arpa/inet.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <netdb.h>
#include <netinet/in.h>

#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using namespace onnx::optimization;
using namespace onnx;

namespace duckonnx
{

    namespace core
    {

        std::string OnnxPushdownExtension::onnx_model_path;
        std::string OnnxPushdownExtension::new_model_path;

        OnnxPushdownExtension::OnnxPushdownExtension()
        {
            optimize_function = OnnxPushdownFunction;
        }

        void OnnxPushdownExtension::reg2reg(std::string &model_path, onnx::graph_node_list &node_list,
                                                   std::vector<int64_t> &output_nodes_featureids, std::size_t dim_value)
        {
            int64_t input_n_targets;
            std::vector<int64_t> input_nodes_falsenodeids;
            // std::vector<int64_t> input_nodes_featureids;
            std::vector<double> input_nodes_hitrates;
            std::vector<int64_t> input_nodes_missing_value_tracks_true;
            std::vector<std::string> input_nodes_modes;
            std::vector<int64_t> input_nodes_nodeids;
            std::vector<int64_t> input_nodes_treeids;
            std::vector<int64_t> input_nodes_truenodeids;
            std::vector<double> input_nodes_values;
            std::string input_post_transform;
            std::vector<int64_t> input_target_ids;
            std::vector<int64_t> input_target_nodeids;
            std::vector<int64_t> input_target_treeids;
            std::vector<double> input_target_weights;

            std::unordered_map<std::string, int> attr_map = {{"n_targets", 1},
                                                             {"nodes_falsenodeids", 2},
                                                             {"nodes_featureids", 3},
                                                             {"nodes_hitrates", 4},
                                                             {"nodes_missing_value_tracks_true", 5},
                                                             {"nodes_modes", 6},
                                                             {"nodes_nodeids", 7},
                                                             {"nodes_treeids", 8},
                                                             {"nodes_truenodeids", 9},
                                                             {"nodes_values", 10},
                                                             {"post_transform", 11},
                                                             {"target_ids", 12},
                                                             {"target_nodeids", 13},
                                                             {"target_treeids", 14},
                                                             {"target_weights", 15}};
            for (auto node : node_list)
            {
                for (auto name : node->attributeNames())
                {
                    std::string attr_name = name.toString();
                    auto it = attr_map.find(attr_name);
                    if (it != attr_map.end())
                    {
                        switch (it->second)
                        {
                        case 1:
                            input_n_targets = node->i(name);
                            break;
                        case 2:
                            input_nodes_falsenodeids = node->is(name);
                            break;
                        // case 3:
                        // 	input_nodes_featureids = node->is(name);
                        // 	break;
                        case 4:
                            input_nodes_hitrates = node->fs(name);
                            break;
                        case 5:
                            input_nodes_missing_value_tracks_true = node->is(name);
                            break;
                        case 6:
                            input_nodes_modes = node->ss(name);
                            break;
                        case 7:
                            input_nodes_nodeids = node->is(name);
                            break;
                        case 8:
                            input_nodes_treeids = node->is(name);
                            break;
                        case 9:
                            input_nodes_truenodeids = node->is(name);
                            break;
                        case 10:
                            input_nodes_values = node->fs(name);
                            break;
                        case 11:
                            input_post_transform = node->s(name);
                            break;
                        case 12:
                            input_target_ids = node->is(name);
                            break;
                        case 13:
                            input_target_nodeids = node->is(name);
                            break;
                        case 14:
                            input_target_treeids = node->is(name);
                            break;
                        case 15:
                            input_target_weights = node->fs(name);
                            break;
                        default:
                            break;
                        }
                    }
                }
            }
            // load initial model
            ModelProto initial_model;
            onnx::optimization::loadModel(&initial_model, model_path, true);
            GraphProto *initial_graph = initial_model.mutable_graph();

            ModelProto model;
            GraphProto *graph = model.mutable_graph();
            model.set_ir_version(initial_model.ir_version());

            for (const auto &input : initial_graph->input())
            {
                onnx::ValueInfoProto *new_input = graph->add_input();
                new_input->set_name(input.name());

                // 设置类型 (TypeProto)
                onnx::TypeProto *new_type = new_input->mutable_type();
                new_type->CopyFrom(input.type());

                // 修改Tensor维度
                if (new_type->has_tensor_type())
                {
                    auto *tensor_type = new_type->mutable_tensor_type();
                    onnx::TensorShapeProto temp_shape = tensor_type->shape();
                    tensor_type->clear_shape();
                    for (int i = 0; i < temp_shape.dim_size(); ++i)
                    {
                        const auto &dim = temp_shape.dim(i);
                        auto *new_dim = tensor_type->mutable_shape()->add_dim();
                        if (dim.has_dim_value())
                        {
                            new_dim->set_dim_value(dim_value);
                        }
                        else if (dim.has_dim_param())
                        {
                            new_dim->set_dim_param(dim.dim_param());
                        }
                    }
                }
            }

            for (const auto &output : initial_graph->output())
            {
                onnx::ValueInfoProto *new_output = graph->add_output();
                new_output->CopyFrom(output);
            }

            for (const auto &initializer : initial_graph->initializer())
            {
                onnx::TensorProto *new_initializer = graph->add_initializer();
                new_initializer->CopyFrom(initializer);
            }

            // 设置新模型的opset_import
            *model.mutable_opset_import() = initial_model.opset_import();

            // 3. 添加 TreeEnsembleRegressor 节点
            NodeProto new_node;
            auto initial_node = initial_graph->node()[0];
            new_node.set_op_type(initial_node.op_type());
            new_node.set_domain(initial_node.domain());    // 设置 domain 为 ai.onnx.ml
            new_node.set_name(initial_node.name());        // 设置节点名称
            new_node.add_input(initial_node.input()[0]);   // 输入
            new_node.add_output(initial_node.output()[0]); // 输出

            // 设置节点属性
            // 1. n_targets
            AttributeProto attr_n_targets;
            attr_n_targets.set_name("n_targets");
            attr_n_targets.set_type(AttributeProto::INT);
            attr_n_targets.set_i(input_n_targets);
            *new_node.add_attribute() = attr_n_targets;

            // 2. nodes_falsenodeids
            AttributeProto attr_nodes_falsenodeids;
            attr_nodes_falsenodeids.set_name("nodes_falsenodeids");
            attr_nodes_falsenodeids.set_type(AttributeProto::INTS);
            for (const auto &id : input_nodes_falsenodeids)
            {
                attr_nodes_falsenodeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_falsenodeids;

            // 3. nodes_featureids
            AttributeProto attr_nodes_featureids;
            attr_nodes_featureids.set_name("nodes_featureids");
            attr_nodes_featureids.set_type(AttributeProto::INTS);
            for (const auto &id : output_nodes_featureids)
            {
                attr_nodes_featureids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_featureids;

            // 4. nodes_hitrates
            AttributeProto attr_nodes_hitrates;
            attr_nodes_hitrates.set_name("nodes_hitrates");
            attr_nodes_hitrates.set_type(AttributeProto::FLOATS);
            for (const auto &rate : input_nodes_hitrates)
            {
                attr_nodes_hitrates.add_floats(rate);
            }
            *new_node.add_attribute() = attr_nodes_hitrates;

            // 5. nodes_missing_value_tracks_true
            AttributeProto attr_nodes_missing_value_tracks_true;
            attr_nodes_missing_value_tracks_true.set_name("nodes_missing_value_tracks_true");
            attr_nodes_missing_value_tracks_true.set_type(AttributeProto::INTS);
            for (const auto &id : input_nodes_missing_value_tracks_true)
            {
                attr_nodes_missing_value_tracks_true.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_missing_value_tracks_true;

            // 6. nodes_modes
            AttributeProto attr_nodes_modes;
            attr_nodes_modes.set_name("nodes_modes");
            attr_nodes_modes.set_type(AttributeProto::STRINGS);
            for (const auto &mode : input_nodes_modes)
            {
                attr_nodes_modes.add_strings(mode);
            }
            *new_node.add_attribute() = attr_nodes_modes;

            // 7. nodes_nodeids
            AttributeProto attr_nodes_nodeids;
            attr_nodes_nodeids.set_name("nodes_nodeids");
            attr_nodes_nodeids.set_type(AttributeProto::INTS);
            for (const auto &id : input_nodes_nodeids)
            {
                attr_nodes_nodeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_nodeids;

            // 8. nodes_treeids
            AttributeProto attr_nodes_treeids;
            attr_nodes_treeids.set_name("nodes_treeids");
            attr_nodes_treeids.set_type(AttributeProto::INTS);
            for (const auto &id : input_nodes_treeids)
            {
                attr_nodes_treeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_treeids;

            // 9. nodes_truenodeids
            AttributeProto attr_nodes_truenodeids;
            attr_nodes_truenodeids.set_name("nodes_truenodeids");
            attr_nodes_truenodeids.set_type(AttributeProto::INTS);
            for (const auto &id : input_nodes_truenodeids)
            {
                attr_nodes_truenodeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_truenodeids;

            // 10. nodes_values
            AttributeProto attr_nodes_values;
            attr_nodes_values.set_name("nodes_values");
            attr_nodes_values.set_type(AttributeProto::FLOATS);
            for (const auto &val : input_nodes_values)
            {
                attr_nodes_values.add_floats(val);
            }
            *new_node.add_attribute() = attr_nodes_values;

            // 11. post_transform
            AttributeProto attr_post_transform;
            attr_post_transform.set_name("post_transform");
            attr_post_transform.set_type(AttributeProto::STRING);
            attr_post_transform.set_s(input_post_transform);
            *new_node.add_attribute() = attr_post_transform;

            // 12. target_ids
            AttributeProto attr_target_ids;
            attr_target_ids.set_name("target_ids");
            attr_target_ids.set_type(AttributeProto::INTS);
            for (const auto &id : input_target_ids)
            {
                attr_target_ids.add_ints(id);
            }
            *new_node.add_attribute() = attr_target_ids;

            // 13. target_nodeids
            AttributeProto attr_target_nodeids;
            attr_target_nodeids.set_name("target_nodeids");
            attr_target_nodeids.set_type(AttributeProto::INTS);
            for (const auto &id : input_target_nodeids)
            {
                attr_target_nodeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_target_nodeids;

            // 14. target_treeids
            AttributeProto attr_target_treeids;
            attr_target_treeids.set_name("target_treeids");
            attr_target_treeids.set_type(AttributeProto::INTS);
            for (const auto &id : input_target_treeids)
            {
                attr_target_treeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_target_treeids;

            // 15. target_weights
            AttributeProto attr_target_weights;
            attr_target_weights.set_name("target_weights");
            attr_target_weights.set_type(AttributeProto::FLOATS);
            for (const auto &weight : input_target_weights)
            {
                attr_target_weights.add_floats(weight);
            }
            *new_node.add_attribute() = attr_target_weights;
            // 将新节点添加到图中
            graph->add_node()->CopyFrom(new_node);

            // 获取时间戳
            boost::uuids::uuid uuid = boost::uuids::random_generator()();
            size_t pos = onnx_model_path.find(".onnx");
            std::string model_name = onnx_model_path.substr(0, pos);
            new_model_path = model_name + "_" + boost::uuids::to_string(uuid) + ".onnx";

            saveModel(&model, new_model_path);
        }

        std::set<idx_t> OnnxPushdownExtension::removeUnusedColumns(std::string &model_path, const std::vector<idx_t> &column_indexs,
                                                                          onnx::graph_node_list &node_list)
        {
            std::vector<int64_t> input_nodes_featureids;
            std::vector<int64_t> output_nodes_featureids;

            for (auto node : node_list)
            {
                for (auto name : node->attributeNames())
                {
                    if (strcmp(name.toString(), "nodes_featureids") == 0)
                    {
                        input_nodes_featureids = node->is(name);
                    }
                }
            }
            std::set<idx_t> unique_values(input_nodes_featureids.begin(), input_nodes_featureids.end());
            std::vector<idx_t> used_nodes_featureids(unique_values.begin(), unique_values.end());

            for (size_t i = 0; i < input_nodes_featureids.size(); i++)
            {
                for (size_t j = 0; j < used_nodes_featureids.size(); j++)
                {
                    if (input_nodes_featureids[i] == used_nodes_featureids[j])
                    {
                        output_nodes_featureids.push_back(j);
                    }
                }
            }
            reg2reg(model_path, node_list, output_nodes_featureids, unique_values.size());
            return unique_values;
        }

        std::set<idx_t> OnnxPushdownExtension::OnnxConstructFunction(const std::vector<idx_t> &column_indexs)
        {
            ModelProto model;
            onnx::optimization::loadModel(&model, onnx_model_path, true);
            std::shared_ptr<Graph> graph(ImportModelProto(model));
            auto node_list = graph->nodes();
            return removeUnusedColumns(onnx_model_path, column_indexs, node_list);
        }

        bool OnnxPushdownExtension::HasONNXExpressionScan(Expression &expr)
        {
            if (expr.expression_class == ExpressionClass::BOUND_FUNCTION)
            {
                auto &func_expr = (BoundFunctionExpression &)expr;
                if (func_expr.function.name == "onnx")
                {
                    auto &first_param = (BoundConstantExpression &)*func_expr.children[0];
                    onnx_model_path = first_param.value.ToString();
                    std::vector<idx_t> column_indexs;
                    for (size_t i = 1; i < func_expr.children.size(); i++)
                    {
                        auto &param_expr = func_expr.children[i];
                        auto &col_expr = (BoundColumnRefExpression &)*param_expr;
                        column_indexs.push_back(col_expr.binding.column_index);
                    }
                    std::set<idx_t> filtered_column_indexs = OnnxConstructFunction(column_indexs);
                    duckdb::Value model_path_value(new_model_path);
                    first_param.value = model_path_value;
                    for (int i = func_expr.children.size() - 1; i > 0; i--)
                    {
                        auto &param_expr = func_expr.children[i];
                        auto &col_expr = (BoundColumnRefExpression &)*param_expr;
                        if (filtered_column_indexs.find(col_expr.binding.column_index) == filtered_column_indexs.end())
                        {
                            func_expr.children.erase(func_expr.children.begin() + i);
                        }
                    }
                    return true;
                }
            }
            bool found_onnx = false;
            ExpressionIterator::EnumerateChildren(expr, [&](Expression &child)
                                                  {
			        if (HasONNXExpressionScan(child)) {
				       found_onnx = true;
			    } });
            return found_onnx;
        }

        bool OnnxPushdownExtension::HasONNXScan(LogicalOperator &op)
        {
            for (auto &expr : op.expressions)
            {
                if (HasONNXExpressionScan(*expr))
                {
                    return true;
                }
            }
            for (auto &child : op.children)
            {
                if (HasONNXScan(*child))
                {
                    return true;
                }
            }
            return false;
        }
        // Optimizer Function
        void OnnxPushdownExtension::OnnxPushdownFunction(OptimizerExtensionInput &input,
                                                                duckdb::unique_ptr<LogicalOperator> &plan)
        {
            if (!HasONNXScan(*plan))
            {
                return;
            }
        }

        //------------------------------------------------------------------------------
        // Register functions
        //------------------------------------------------------------------------------
        void CoreOnnxOptimizer::RegisterOnnxPushdownOptimizer(
            DatabaseInstance &db)
        {
            auto &config = DBConfig::GetConfig(db);
            config.optimizer_extensions.push_back(OnnxPushdownExtension());
            config.AddExtensionOption("pushdown", "remove unused onnx model columns", LogicalType::INVALID);
        }

    } // namespace core

} // namespace duckonnx
