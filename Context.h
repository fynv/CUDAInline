#pragma once

#include <vector>
#include <string>
#include "DeviceViewable.h"

namespace CUInline
{
	struct dim_type
	{
		unsigned int x, y, z;
	};

	void set_libnvrtc_path(const char* path);

	void SetVerbose(bool verbose = true);

	// reflection 
	size_t SizeOf(const char* cls);
	bool QueryStruct(const char* name_struct, const std::vector<const char*>& name_members, size_t* offsets);

	// Adding definitions to device code
	void AddIncludeDir(const char* path);
	void AddBuiltInHeader(const char* name, const char* content);
	void AddCodeBlock(const char* code);
	void AddInlcudeFilename(const char* fn);
	void AddConstantObject(const char* name, const DeviceViewable& obj);
	std::string AddStruct(const char* struct_body);

	void Wait();

	class Kernel
	{
	public:
		size_t num_params() const { return m_param_names.size(); }

		Kernel(const std::vector<const char*>& param_names, const char* code_body);
		bool calc_optimal_block_size(const DeviceViewable** args, int& sizeBlock, unsigned sharedMemBytes = 0);
		bool calc_number_blocks(const DeviceViewable** args, int sizeBlock, int& numBlocks, unsigned sharedMemBytes = 0);
		bool launch(dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes = 0);

	private:
		std::vector<std::string> m_param_names;
		std::string m_code_body;

	};
}