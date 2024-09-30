#define DUCKDB_EXTENSION_MAIN

#include "duckonnx_extension.hpp"
#include "duckonnx/common.hpp"
#include "duckonnx/core/module.hpp"

namespace duckdb {

static void LoadInternal(DatabaseInstance &instance) {
	duckonnx::core::CoreModule::Register(instance);
	
}

void DuckonnxExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string DuckonnxExtension::Name() {
	return "duckonnx";
}

// std::string DuckonnxExtension::Version() const {
// #ifdef EXT_VERSION_ONNX
// 	return EXT_VERSION_ONNX;
// #else
// 	return "";
// #endif
// }

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void duckonnx_init(DatabaseInstance &db) {
	LoadInternal(db);
    // duckdb::DuckDB db_wrapper(db);
    // db_wrapper.LoadExtension<duckdb::OnnxExtension>();
}

DUCKDB_EXTENSION_API const char *duckonnx_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
