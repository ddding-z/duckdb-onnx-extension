#include "duckonnx/core/optimizer/rules/onnx_pruning.hpp"
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
#include <functional>
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

        std::string OnnxPruningExtension::onnx_model_path;
        std::string OnnxPruningExtension::new_model_path;
        float_t OnnxPruningExtension::predicate;
        std::vector<std::string> OnnxPruningExtension::removed_nodes;

        std::vector<int64_t> OnnxPruningExtension::left_nodes;
        std::vector<int64_t> OnnxPruningExtension::right_nodes;
        std::vector<std::string> OnnxPruningExtension::node_types;
        std::vector<double> OnnxPruningExtension::node_thresholds;
        std::vector<int64_t> OnnxPruningExtension::target_nodeids;
        std::vector<double> OnnxPruningExtension::target_weights;

        ExpressionType OnnxPruningExtension::ComparisonOperator;
        std::unordered_map<ExpressionType, std::function<bool(float_t, float_t)>> OnnxPruningExtension::comparison_funcs;

        OnnxPruningExtension::OnnxPruningExtension()
        {
            optimize_function = OnnxPruningFunction;

            comparison_funcs[ExpressionType::COMPARE_LESSTHAN] = [](float_t x, float_t y) -> bool
            {
                return x < y;
            };
            comparison_funcs[ExpressionType::COMPARE_GREATERTHAN] = [](float_t x, float_t y) -> bool
            {
                return x > y;
            };
            comparison_funcs[ExpressionType::COMPARE_LESSTHANOREQUALTO] = [](float_t x, float_t y) -> bool
            {
                return x <= y;
            };
            comparison_funcs[ExpressionType::COMPARE_GREATERTHANOREQUALTO] = [](float_t x, float_t y) -> bool
            {
                return x >= y;
            };
        }

        bool OnnxPruningExtension::HasONNXFilter(LogicalOperator &op)
        {
            for (auto &expr : op.expressions)
            {
                if (expr->expression_class == ExpressionClass::BOUND_COMPARISON)
                {
                    auto &comparison_expr = dynamic_cast<BoundComparisonExpression &>(*expr);
                    if (comparison_expr.left->expression_class == ExpressionClass::BOUND_FUNCTION)
                    {
                        auto &func_expr = (BoundFunctionExpression &)*comparison_expr.left;
                        if (func_expr.function.name == "onnx" && func_expr.children.size() > 1)
                        {
                            auto &first_param = (BoundConstantExpression &)*func_expr.children[0];
                            if (first_param.value.type().id() == LogicalTypeId::VARCHAR)
                            {
                                std::string model_path = first_param.value.ToString();
                                if (comparison_expr.right->type == ExpressionType::VALUE_CONSTANT)
                                {
                                    auto &constant_expr = (BoundConstantExpression &)*comparison_expr.right;
                                    predicate = constant_expr.value.GetValue<float_t> ();
                                    onnx_model_path = model_path;
                                    ComparisonOperator = comparison_expr.type;
                                    // set comparison_expr => =1
                                    comparison_expr.type = ExpressionType::COMPARE_EQUAL;
                                    duckdb::Value value(1.0f);
                                    auto new_constant_expr = std::make_unique<duckdb::BoundConstantExpression>(value);
                                    comparison_expr.right = std::move(new_constant_expr);

                                    boost::uuids::uuid uuid = boost::uuids::random_generator()();
                                    size_t pos = onnx_model_path.find(".onnx");
                                    std::string model_name = onnx_model_path.substr(0, pos);
                                    new_model_path = model_name + "_" + boost::uuids::to_string(uuid) + ".onnx";

                                    duckdb::Value model_path_value(new_model_path);
                                    first_param.value = model_path_value;
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            // 递归检查子节点
            for (auto &child : op.children)
            {
                if (HasONNXFilter(*child))
                {
                    return true;
                }
            }
            return false;
        }

        int OnnxPruningExtension::pruning(size_t node_id, size_t depth, std::vector<std::string> &result_nodes, onnx::graph_node_list &node_list, float_t predicate)
        {
            for (auto node : node_list)
            {
                for (auto name : node->attributeNames())
                {
                    if (strcmp(name.toString(), "nodes_truenodeids") == 0)
                    {
                        left_nodes = node->is(name);
                    }
                    if (strcmp(name.toString(), "nodes_falsenodeids") == 0)
                    {
                        right_nodes = node->is(name);
                    }
                    if (strcmp(name.toString(), "nodes_modes") == 0)
                    {
                        node_types = node->ss(name);
                    }
                    if (strcmp(name.toString(), "nodes_values") == 0)
                    {
                        node_thresholds = node->fs(name);
                    }
                    if (strcmp(name.toString(), "target_nodeids") == 0)
                    {
                        target_nodeids = node->is(name);
                    }
                    if (strcmp(name.toString(), "target_weights") == 0)
                    {
                        target_weights = node->fs(name);
                    }
                }
            }
            result_nodes[node_id] = node_types[node_id];
            auto is_leaf = node_types[node_id] == "LEAF";
            if (is_leaf)
            {
                auto target_id = -1;
                for (size_t ti = 0; ti < target_nodeids.size(); ++ti)
                {
                    int ni = target_nodeids[ti];
                    if (ni == node_id)
                    {
                        target_id = static_cast<int>(ti);
                        break;
                    }
                }
                // modified
                auto result = static_cast<int>(comparison_funcs[ComparisonOperator](target_weights[target_id], predicate));
                result == 1 ? result_nodes[node_id] = "LEAF_TRUE" : result_nodes[node_id] = "LEAF_FALSE";
                // std::cout << "node_id: " << node_id << ", depth: " << depth << ", is_leaf: " << (is_leaf ? "true" :
                // "false")
                //           << ", result: " << result << std::endl;
                return result;
            }
            else
            {
                auto left_node_id = left_nodes[node_id];
                auto left_result = pruning(left_node_id, depth + 1, result_nodes, node_list, predicate);
                auto right_node_id = right_nodes[node_id];
                auto right_result = pruning(right_node_id, depth + 1, result_nodes, node_list, predicate);

                if (left_result == 0 && right_result == 0)
                {
                    // std::cout << "node_id: " << node_id << ", depth: " << depth
                    //           << ", is_leaf: " << (is_leaf ? "true" : "false") << ", result: " << 0 << std::endl;
                    result_nodes[node_id] = "LEAF_FALSE";
                    result_nodes[left_node_id] = "REMOVED";
                    result_nodes[right_node_id] = "REMOVED";
                    return 0;
                }

                if (left_result == 1 && right_result == 1)
                {
                    // std::cout << "node_id: " << node_id << ", depth: " << depth
                    //           << ", is_leaf: " << (is_leaf ? "true" : "false") << ", result: " << 1 << std::endl;
                    result_nodes[node_id] = "LEAF_TRUE";
                    result_nodes[left_node_id] = "REMOVED";
                    result_nodes[right_node_id] = "REMOVED";
                    return 1;
                }
                // std::cout << "node_id: " << node_id << ", depth: " << depth + 1 << ", leaf_depth: " << depth + 1
                //           << std::endl;
                return 2;
            }
        }

        void OnnxPruningExtension::OnnxPruneFunction()
        {
            ModelProto model;
            onnx::optimization::loadModel(&model, onnx_model_path, true);
            std::shared_ptr<Graph> graph(ImportModelProto(model));
            auto node_list = graph->nodes();
            size_t length;
            for (auto node : node_list)
            {
                for (auto name : node->attributeNames())
                {
                    if (strcmp(name.toString(), "nodes_modes") == 0)
                    {
                        length = node->ss(name).size();
                        break;
                    }
                }
            }
            vector<std::string> result_nodes{length, ""};
            pruning(0, 0, result_nodes, node_list, predicate);
            removed_nodes = result_nodes;
        }

        void OnnxPruningExtension::reg2reg(std::string &model_path, onnx::graph_node_list &node_list)
        {
            int64_t input_n_targets;
            std::vector<int64_t> input_nodes_falsenodeids;
            std::vector<int64_t> input_nodes_featureids;
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
                        case 3:
                            input_nodes_featureids = node->is(name);
                            break;
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

            // 1. 计算 leaf_count
            int leaf_count = std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_FALSE") +
                             std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_TRUE");

            // 2. 构建 new_ids
            vector<NodeID> new_ids;
            int id_ = 0;
            for (const auto &node : removed_nodes)
            {
                if (node == "LEAF_FALSE" || node == "LEAF_TRUE" || node == "BRANCH_LEQ")
                {
                    new_ids.push_back({id_, node});
                    id_++;
                }
                else
                {
                    new_ids.push_back({-1, node});
                }
            }

            // 3. 赋值 n_targets
            int n_targets = input_n_targets;

            // 4. 构建 nodes_falsenodeids
            vector<int> nodes_falsenodeids;
            for (size_t i = 0; i < input_nodes_falsenodeids.size(); ++i)
            {
                int ii = input_nodes_falsenodeids[i];
                if (new_ids[i].node != "REMOVED")
                {
                    int value = 0;
                    if (ii >= 0 && static_cast<size_t>(ii) < new_ids.size())
                    {
                        int new_id_value = new_ids[ii].id;
                        value = (new_id_value != -1) ? new_id_value : 0;
                    }
                    nodes_falsenodeids.push_back(value);
                }
            }

            // 5. 构建 nodes_featureids
            vector<int> nodes_featureids;
            for (size_t i = 0; i < input_nodes_featureids.size(); ++i)
            {
                int ii = input_nodes_featureids[i];
                if (new_ids[i].id != -1)
                {
                    int value = (new_ids[i].node == "BRANCH_LEQ") ? ii : 0;
                    nodes_featureids.push_back(value);
                }
            }

            // 6. 构建 nodes_hitrates
            vector<float> nodes_hitrates;
            for (size_t i = 0; i < input_nodes_hitrates.size(); ++i)
            {
                if (new_ids[i].id != -1)
                {
                    nodes_hitrates.push_back(input_nodes_hitrates[i]);
                }
            }

            // 7. 构建 nodes_missing_value_tracks_true
            vector<int> nodes_missing_value_tracks_true;
            for (size_t i = 0; i < input_nodes_missing_value_tracks_true.size(); ++i)
            {
                if (new_ids[i].id != -1)
                {
                    nodes_missing_value_tracks_true.push_back(input_nodes_missing_value_tracks_true[i]);
                }
            }

            // 8. 构建 nodes_modes
            vector<std::string> nodes_modes;
            for (const auto &new_id : new_ids)
            {
                if (new_id.id != -1)
                {
                    std::string mode = (new_id.node == "BRANCH_LEQ") ? "BRANCH_LEQ" : "LEAF";
                    nodes_modes.push_back(mode);
                }
            }

            // 9. 构建 nodes_nodeids
            vector<int> nodes_nodeids;
            for (size_t i = 0; i < input_nodes_nodeids.size(); ++i)
            {
                if (new_ids[i].id != -1)
                {
                    nodes_nodeids.push_back(new_ids[i].id);
                }
            }

            // 10. 构建 nodes_treeids
            vector<int> nodes_treeids;
            for (size_t i = 0; i < input_nodes_treeids.size(); ++i)
            {
                if (new_ids[i].id != -1)
                {
                    nodes_treeids.push_back(input_nodes_treeids[i]);
                }
            }

            // 11. 构建 nodes_truenodeids
            vector<int> nodes_truenodeids;
            for (size_t i = 0; i < input_nodes_truenodeids.size(); ++i)
            {
                int ii = input_nodes_truenodeids[i];
                if (new_ids[i].node != "REMOVED")
                {
                    int value = 0;
                    if (ii >= 0 && static_cast<size_t>(ii) < new_ids.size())
                    {
                        int new_id_value = new_ids[ii].id;
                        value = (new_id_value != -1) ? new_id_value : 0;
                    }
                    nodes_truenodeids.push_back(value);
                }
            }

            // 12. 构建 nodes_values
            vector<float> nodes_values;
            for (size_t i = 0; i < input_nodes_values.size(); ++i)
            {
                if (new_ids[i].id != -1)
                {
                    float value = (new_ids[i].node == "BRANCH_LEQ") ? input_nodes_values[i] : 0.0f;
                    nodes_values.push_back(value);
                }
            }

            // 13. 赋值 post_transform
            string post_transform = input_post_transform;

            // 14. 构建 target_ids
            vector<int> target_ids(leaf_count, 0);

            // 15. 构建 target_nodeids
            vector<int> target_nodeids;
            for (const auto &new_id : new_ids)
            {
                if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE")
                {
                    target_nodeids.push_back(new_id.id);
                }
            }

            // 16. 构建 target_treeids
            vector<int> target_treeids(leaf_count, 0);

            // 17. 构建 target_weights
            vector<float> target_weights;
            for (const auto &new_id : new_ids)
            {
                if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE")
                {
                    float weight = (new_id.node == "LEAF_TRUE") ? 1.0f : 0.0f;
                    target_weights.push_back(weight);
                }
            }

            // ------------------

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
                new_input->CopyFrom(input);
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
            attr_n_targets.set_i(n_targets);
            *new_node.add_attribute() = attr_n_targets;

            // 2. nodes_falsenodeids
            AttributeProto attr_nodes_falsenodeids;
            attr_nodes_falsenodeids.set_name("nodes_falsenodeids");
            attr_nodes_falsenodeids.set_type(AttributeProto::INTS);
            for (const auto &id : nodes_falsenodeids)
            {
                attr_nodes_falsenodeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_falsenodeids;

            // 3. nodes_featureids
            AttributeProto attr_nodes_featureids;
            attr_nodes_featureids.set_name("nodes_featureids");
            attr_nodes_featureids.set_type(AttributeProto::INTS);
            for (const auto &id : nodes_featureids)
            {
                attr_nodes_featureids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_featureids;

            // 4. nodes_hitrates
            AttributeProto attr_nodes_hitrates;
            attr_nodes_hitrates.set_name("nodes_hitrates");
            attr_nodes_hitrates.set_type(AttributeProto::FLOATS);
            for (const auto &rate : nodes_hitrates)
            {
                attr_nodes_hitrates.add_floats(rate);
            }
            *new_node.add_attribute() = attr_nodes_hitrates;

            // 5. nodes_missing_value_tracks_true
            AttributeProto attr_nodes_missing_value_tracks_true;
            attr_nodes_missing_value_tracks_true.set_name("nodes_missing_value_tracks_true");
            attr_nodes_missing_value_tracks_true.set_type(AttributeProto::INTS);
            for (const auto &id : nodes_missing_value_tracks_true)
            {
                attr_nodes_missing_value_tracks_true.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_missing_value_tracks_true;

            // 6. nodes_modes
            AttributeProto attr_nodes_modes;
            attr_nodes_modes.set_name("nodes_modes");
            attr_nodes_modes.set_type(AttributeProto::STRINGS);
            for (const auto &mode : nodes_modes)
            {
                attr_nodes_modes.add_strings(mode);
            }
            *new_node.add_attribute() = attr_nodes_modes;

            // 7. nodes_nodeids
            AttributeProto attr_nodes_nodeids;
            attr_nodes_nodeids.set_name("nodes_nodeids");
            attr_nodes_nodeids.set_type(AttributeProto::INTS);
            for (const auto &id : nodes_nodeids)
            {
                attr_nodes_nodeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_nodeids;

            // 8. nodes_treeids
            AttributeProto attr_nodes_treeids;
            attr_nodes_treeids.set_name("nodes_treeids");
            attr_nodes_treeids.set_type(AttributeProto::INTS);
            for (const auto &id : nodes_treeids)
            {
                attr_nodes_treeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_treeids;

            // 9. nodes_truenodeids
            AttributeProto attr_nodes_truenodeids;
            attr_nodes_truenodeids.set_name("nodes_truenodeids");
            attr_nodes_truenodeids.set_type(AttributeProto::INTS);
            for (const auto &id : nodes_truenodeids)
            {
                attr_nodes_truenodeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_nodes_truenodeids;

            // 10. nodes_values
            AttributeProto attr_nodes_values;
            attr_nodes_values.set_name("nodes_values");
            attr_nodes_values.set_type(AttributeProto::FLOATS);
            for (const auto &val : nodes_values)
            {
                attr_nodes_values.add_floats(val);
            }
            *new_node.add_attribute() = attr_nodes_values;

            // 11. post_transform
            AttributeProto attr_post_transform;
            attr_post_transform.set_name("post_transform");
            attr_post_transform.set_type(AttributeProto::STRING);
            attr_post_transform.set_s(post_transform);
            *new_node.add_attribute() = attr_post_transform;

            // 12. target_ids
            AttributeProto attr_target_ids;
            attr_target_ids.set_name("target_ids");
            attr_target_ids.set_type(AttributeProto::INTS);
            for (const auto &id : target_ids)
            {
                attr_target_ids.add_ints(id);
            }
            *new_node.add_attribute() = attr_target_ids;

            // 13. target_nodeids
            AttributeProto attr_target_nodeids;
            attr_target_nodeids.set_name("target_nodeids");
            attr_target_nodeids.set_type(AttributeProto::INTS);
            for (const auto &id : target_nodeids)
            {
                attr_target_nodeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_target_nodeids;

            // 14. target_treeids
            AttributeProto attr_target_treeids;
            attr_target_treeids.set_name("target_treeids");
            attr_target_treeids.set_type(AttributeProto::INTS);
            for (const auto &id : target_treeids)
            {
                attr_target_treeids.add_ints(id);
            }
            *new_node.add_attribute() = attr_target_treeids;

            // 15. target_weights
            AttributeProto attr_target_weights;
            attr_target_weights.set_name("target_weights");
            attr_target_weights.set_type(AttributeProto::FLOATS);
            for (const auto &weight : target_weights)
            {
                attr_target_weights.add_floats(weight);
            }
            *new_node.add_attribute() = attr_target_weights;

            // 将新节点添加到图中
            graph->add_node()->CopyFrom(new_node);

            saveModel(&model, new_model_path);
        }

        void OnnxPruningExtension::OnnxConstructFunction()
        {
            ModelProto model;
            onnx::optimization::loadModel(&model, onnx_model_path, true);
            std::shared_ptr<Graph> graph(ImportModelProto(model));
            auto node_list = graph->nodes();
            reg2reg(onnx_model_path, node_list);
        }

        void OnnxPruningExtension::OnnxPruningFunction(OptimizerExtensionInput &input,
                                                       duckdb::unique_ptr<LogicalOperator> &plan)
        {
            if (!HasONNXFilter(*plan))
            {
                return;
            }
            OnnxPruneFunction();
            OnnxConstructFunction();
        }

        //------------------------------------------------------------------------------
        // Register functions
        //------------------------------------------------------------------------------
        void CoreOnnxOptimizer::RegisterOnnxPruningOptimizer(
            DatabaseInstance &db)
        {
            auto &config = DBConfig::GetConfig(db);
            config.optimizer_extensions.push_back(OnnxPruningExtension());
            config.AddExtensionOption("pruning", "pruning onnx model", LogicalType::INVALID);
        }

    } // namespace core

} // namespace duckonnx
