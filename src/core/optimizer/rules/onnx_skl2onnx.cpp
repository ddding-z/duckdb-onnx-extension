#include "duckonnx/core/optimizer/rules/onnx_skl2onnx.hpp"
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


namespace duckonnx
{

    namespace core
    {

        OnnxSkl2onnxExtension::OnnxSkl2onnxExtension()
        {
            optimize_function = OnnxSkl2onnxFunction;
        }

        std::string OnnxSkl2onnxExtension::convertModel(std::string &model_path)
        {
            boost::uuids::uuid uuid = boost::uuids::random_generator()();

            size_t pos = model_path.find_last_of("/");
            std::string model_name = model_path.substr(pos + 1);
            std::string prefix = model_path.substr(0, pos);

            pos = model_name.find(".joblib");
            std::string new_model_name = model_name.substr(0, pos) + "_" + boost::uuids::to_string(uuid) + ".onnx";
            std::string command = std::string("./../../exe/exe.linux-x86_64-3.9/convert ") +
                                  prefix + "/" + model_name + " " + prefix + "/" +
                                  new_model_name;
            int ret = system(command.c_str());
            if (ret != 0)
            {
                std::cerr << "convert failed!" << std::endl;
                return model_path;
            }
            
            return prefix + "/" + new_model_name;
        }

        bool OnnxSkl2onnxExtension::HasONNXExpressionScan(Expression &expr)
        {
            if (expr.expression_class == ExpressionClass::BOUND_FUNCTION)
            {
                auto &func_expr = (BoundFunctionExpression &)expr;
                if (func_expr.function.name == "onnx")
                {
                    auto &first_param = (BoundConstantExpression &)*func_expr.children[0];
                    std::string onnx_model_path = first_param.value.ToString();
                    size_t pos = onnx_model_path.find(".joblib");
                    if (pos != std::string::npos)
                    {
                        std::string new_model_path = convertModel(onnx_model_path);
                        if (new_model_path == onnx_model_path)
                        {
                            std::cerr << "convert failed!" << std::endl;
                            return false;
                        }
                        // std::cout<<new_model_path<<std::endl;
                        duckdb::Value model_path_value(new_model_path);
                        first_param.value = model_path_value;
                        return true;
                    }
                    return false;
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

        bool OnnxSkl2onnxExtension::HasONNXScan(LogicalOperator &op)
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

        void OnnxSkl2onnxExtension::OnnxSkl2onnxFunction(OptimizerExtensionInput &input,
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
        void CoreOnnxOptimizer::RegisterOnnxSkl2onnxOptimizer(
            DatabaseInstance &db)
        {
            auto &config = DBConfig::GetConfig(db);
            config.optimizer_extensions.push_back(OnnxSkl2onnxExtension());
            config.AddExtensionOption("skl2onnx", "convert sklearn model to onnx model", LogicalType::INVALID);
        }
    }
}