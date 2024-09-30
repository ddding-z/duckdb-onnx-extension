#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "duckonnx/common.hpp"
#include <duckonnx/core/functions/scalar.hpp>
#include <duckonnx_extension.hpp>

#include "duckonnx/core/utils/model_cache.hpp"

#include "onnxruntime_cxx_api.h"

namespace duckonnx {

namespace core {


// Function to perform inference const vector<const void *> &input_buffers
vector<float> InferenceModel(const std::string &model_path, const vector<const void *> &input_buffers,
                             const vector<int64_t> &input_shape) {

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceModel");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	std::shared_ptr<Ort::Session> session = ModelCache::getOrCreateSession(model_path, env, session_options);
	if (!session) {
		std::cerr << "Failed to create session: " << std::endl;
		return {};
	}

	Ort::AllocatorWithDefaultOptions allocator;
	vector<std::string> input_node_names;
	vector<std::string> output_node_names;
	size_t numInputNodes = session->GetInputCount();
	size_t numOutputNodes = session->GetOutputCount();
	input_node_names.reserve(numInputNodes);
	output_node_names.reserve(numOutputNodes);

	vector<int64_t> adjusted_input_shape = input_shape;
	for (size_t i = 0; i < numInputNodes; i++) {
		auto input_name = session->GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());

		Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		vector<int64_t> input_dims = input_tensor_info.GetShape();

		for (auto &dim : input_dims) {
			if (dim == -1) {
				dim = input_shape[0];
			}
		}
		adjusted_input_shape = input_dims;
	}

	for (size_t i = 0; i < numOutputNodes; i++) {
		auto output_name = session->GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());

		Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();

		for (auto &dim : output_dims) {
			if (dim == -1) {
				dim = input_shape[0];
			}
		}
	}

	// Create input tensor
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor(nullptr);
	if (input_buffers.empty() || input_buffers[0] == nullptr) {
		std::cerr << "Error: Input buffers are empty or null." << std::endl;
		return {};
	}

	// 调整列存数据为行优先数据
	vector<float> row_major_data(adjusted_input_shape[0] * adjusted_input_shape[1]);

	int num_rows = adjusted_input_shape[0]; // 样本数 (行数)
	int num_cols = adjusted_input_shape[1]; // 特征数 (列数)

	for (int col = 0; col < num_cols; ++col) {
		// 提前将列的 void* 转换为 float*
		const float *column_data = static_cast<const float *>(input_buffers[col]);
		// 遍历行数据，转换为行优先存储
		for (int row = 0; row < num_rows; ++row) {
			row_major_data[row * num_cols + col] = column_data[row];
		}
	}

	// Get the data type of the input tensor from the ONNX model
	size_t input_index = 0;
	Ort::TypeInfo type_info = session->GetInputTypeInfo(input_index);
	Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
	auto data_type = tensor_info.GetElementType();

	input_tensor = Ort::Value::CreateTensor<float>(memory_info,                 // 内存信息
	                                               row_major_data.data(),       // 指向行优先的数据
	                                               row_major_data.size(),       // 数据元素总数
	                                               adjusted_input_shape.data(), // 张量形状
	                                               adjusted_input_shape.size()  // 维度数
	);

	// 获取输出张量形状, 目前假设只有一个输出
	vector<int64_t> output_shape;
	try {
		Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		output_shape = output_tensor_info.GetShape();
		// 检查输出张量中是否有动态维度
		for (auto &dim : output_shape) {
			if (dim == -1) {
				dim = input_shape[0];
			}
		}
	} catch (const Ort::Exception &e) {
		std::cerr << "Failed to get output shape: " << e.what() << std::endl;
		return {};
	}

	// 创建输出张量
	vector<float_t> output_data(output_shape[0]);
	Ort::Value output_tensor(nullptr);
	try {
		output_tensor = Ort::Value::CreateTensor<float_t>(memory_info, output_data.data(), output_data.size(),
		                                                  output_shape.data(), output_shape.size());
	} catch (const Ort::Exception &e) {
		std::cerr << "Failed to create output tensor: " << e.what() << std::endl;
		return {};
	}

	vector<const char *> input_node_names_c;
	vector<const char *> output_node_names_c;
	for (const auto &name : input_node_names) {
		input_node_names_c.push_back(name.c_str());
	}
	for (const auto &name : output_node_names) {
		output_node_names_c.push_back(name.c_str());
	}

	try {
		session->Run(Ort::RunOptions {nullptr}, input_node_names_c.data(), &input_tensor, 1, output_node_names_c.data(),
		             &output_tensor, 1);
	} catch (const Ort::Exception &e) {
		std::cerr << "Failed to run inference: " << e.what() << std::endl;
		return {};
	}
	// auto adjusted_output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
	// Extract output data
	float *output_data_ptr = output_tensor.GetTensorMutableData<float>();
	vector<float> results(output_data_ptr, output_data_ptr + output_data.size());
	return results;
}


static void OnnxInferenceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &model_path_vector = args.data[0];
	// check data type
	if (model_path_vector.GetType().id() != LogicalTypeId::VARCHAR) {
		std::cerr << "Error: Model path column must be VARCHAR type." << std::endl;
		return;
	}
	// Assume VARCHAR
	auto model_path_data = FlatVector::GetData<string_t>(model_path_vector);
	idx_t model_path_index = 0;
	// make sure index valid
	if (model_path_index >= args.size()) {
		std::cerr << "Error: Invalid model path index." << std::endl;
		return;
	}
	std::string model_path = model_path_data[model_path_index].GetString();
	vector<UnifiedVectorFormat> feature_data(args.ColumnCount() - 1);
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		// std::cout << "Processing column: " << col_idx << std::endl;
		// std::cout << "Data type: " << args.data[col_idx].GetType().ToString() << std::endl; // 打印数据类型
		args.data[col_idx].ToUnifiedFormat(args.size(), feature_data[col_idx - 1]);
	}

	vector<const void *> input_buffers;
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		const void *ptr = feature_data[col_idx - 1].data;
		input_buffers.push_back(ptr);
		// 假设数据是 float 类型
	}
	// input_shape: (batch_size, num_features)
	vector<int64_t> input_shape = {(int64_t)args.size(), (int64_t)args.ColumnCount() - 1};

	// Run inference
	auto inference_results = InferenceModel(model_path, input_buffers, input_shape);

	// Write output vector
	result.SetVectorType(VectorType::FLAT_VECTOR);

	if (inference_results.size() != args.size()) {
		std::cerr << "Error: Inference results size mismatch." << std::endl;
		return;
	}
	auto result_data = FlatVector::GetData<float_t>(result);
	for (idx_t row_idx = 0; row_idx < args.size(); row_idx++) {
		result_data[row_idx] = inference_results[row_idx];
	}
	result.Verify(args.size());
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterOnnxInferenceScalarFunction(
    DatabaseInstance &db) {
  ExtensionUtil::RegisterFunction(
      db,
      ScalarFunction("onnx",
                     {LogicalType::VARCHAR, LogicalType::ANY},
                     LogicalType::FLOAT,
                     OnnxInferenceFunction,
					 nullptr,
					 nullptr,
					 nullptr,
					 nullptr,
					 LogicalType::ANY));
					 
}

} // namespace core

} // namespace duckonnx
