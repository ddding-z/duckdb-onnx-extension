#pragma once

#include "duckonnx/common.hpp"
// #include "duckdb/parser/parsed_expression.hpp"
// #include "onnx/core/utils/compressed_sparse_row.hpp"
// #include "duckdb/parser/parsed_data/create_property_graph_info.hpp"

namespace duckdb {

class DuckonnxExtension : public Extension {
public:
	void Load(DuckDB &db) override;
	std::string Name() override;
    // std::string Version() const override;
};

} // namespace duckdb
