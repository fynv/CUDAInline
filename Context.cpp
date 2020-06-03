#include "Context.h"
#include <unordered_map>
#include <unordered_set>
#include <string.h>
#include <stdio.h>
#include <unqlite.h>
#include <shared_mutex>
#include "cuda_wrapper.h"
#include "nvtrc_wrapper.h"
#include "launch_calc.h"
#include "crc64.h"
#include "cuda_inline_headers_global.hpp"

namespace CUInline
{
	typedef unsigned int KernelId_t;

	class Context
	{
	public:
		static void set_libnvrtc_path(const char* path);
		static bool try_init();
		static Context& get_context();

		void set_verbose(bool verbose = true);

		// reflection 
		size_t size_of(const char* cls);
		bool query_struct(const char* name_struct, const std::vector<const char*>& name_members, size_t* offsets);
		bool calc_optimal_block_size(const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, int& sizeBlock, unsigned sharedMemBytes = 0);
		bool calc_number_blocks(const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, int sizeBlock, int& numBlocks, unsigned sharedMemBytes = 0);
		bool launch_kernel(dim_type gridDim, dim_type blockDim, const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, unsigned sharedMemBytes = 0);

		void add_include_dir(const char* path);
		void add_built_in_header(const char* name, const char* content);
		void add_code_block(const char* code);
		void add_inlcude_filename(const char* fn);
		void add_constant_object(const char* name, const DeviceViewable& obj);
		std::string add_struct(const char* struct_body);

	private:
		Context();
		~Context();

		bool _src_to_ptx(const char* src, std::vector<char>& ptx, size_t& ptx_size);
		KernelId_t _build_kernel(const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body);
		int _launch_calc(KernelId_t kid, unsigned sharedMemBytes);
		int _persist_calc(KernelId_t kid, int numBlocks, unsigned sharedMemBytes);
		bool _launch_kernel(KernelId_t kid, dim_type gridDim, dim_type blockDim, const std::vector<CapturedDeviceViewable>& arg_map, unsigned sharedMemBytes);

		static const char* s_libnvrtc_path;

		bool m_verbose;
		std::vector<std::string> m_include_dirs;
		std::vector<const char*> m_name_built_in_headers;
		std::vector<const char*> m_content_built_in_headers;
		std::vector<std::string> m_code_blocks;
		std::vector<std::pair<std::string, ViewBuf>> m_constants;

		std::string m_header_of_structs;
		std::string m_name_header_of_structs;
		std::unordered_set<int64_t> m_known_structs;
		std::shared_mutex m_mutex_structs;

		std::unordered_map<std::string, size_t> m_size_of_types;
		std::shared_mutex m_mutex_sizes;

		std::unordered_map<std::string, std::vector<size_t>> m_offsets_of_structs;
		std::shared_mutex m_mutex_offsets;

		struct Kernel;
		std::vector<Kernel*> m_kernel_cache;
		std::unordered_map<int64_t, KernelId_t> m_kernel_id_map;
		std::shared_mutex m_mutex_kernels;
	};

}

#include "impl_context.inl"

namespace CUInline
{
	void set_libnvrtc_path(const char* path)
	{
		Context::set_libnvrtc_path(path);
	}

	bool TryInit()
	{
		if (Context::try_init())
		{
			Context::get_context();
			return true;
		}
		return false;
	}

	void SetVerbose(bool verbose)
	{
		Context& ctx = Context::get_context();
		ctx.set_verbose(verbose);
	}

	size_t SizeOf(const char* cls)
	{
		Context& ctx = Context::get_context();
		return ctx.size_of(cls);
	}

	bool QueryStruct(const char* name_struct, const std::vector<const char*>& name_members, size_t* offsets)
	{
		Context& ctx = Context::get_context();
		return ctx.query_struct(name_struct, name_members, offsets);
	}

	void AddIncludeDir(const char* path)
	{
		Context& ctx = Context::get_context();
		ctx.add_include_dir(path);
	}

	void AddBuiltInHeader(const char* name, const char* content)
	{
		Context& ctx = Context::get_context();
		ctx.add_built_in_header(name, content);
	}

	void AddCodeBlock(const char* code)
	{
		Context& ctx = Context::get_context();
		ctx.add_code_block(code);
	}

	void AddInlcudeFilename(const char* fn)
	{
		Context& ctx = Context::get_context();
		ctx.add_inlcude_filename(fn);
	}

	void AddConstantObject(const char* name, const DeviceViewable& obj)
	{
		Context& ctx = Context::get_context();
		ctx.add_constant_object(name, obj);
	}

	std::string AddStruct(const char* struct_body)
	{
		Context& ctx = Context::get_context();
		return ctx.add_struct(struct_body);
	}

	void Wait()
	{
		Context::get_context(); // make sure initialization
		cuCtxSynchronize();
	}

	Kernel::Kernel(const std::vector<const char*>& param_names, const char* code_body) :
		m_param_names(param_names.size()), m_code_body(code_body)
	{
		for (size_t i = 0; i < param_names.size(); i++)
			m_param_names[i] = param_names[i];
	}

	bool Kernel::calc_optimal_block_size(const DeviceViewable** args, int& sizeBlock, unsigned sharedMemBytes)
	{
		Context& ctx = Context::get_context();
		std::vector<CapturedDeviceViewable> arg_map(m_param_names.size());
		for (size_t i = 0; i < m_param_names.size(); i++)
		{
			arg_map[i].obj_name = m_param_names[i].c_str();
			arg_map[i].obj = args[i];
		}
		return ctx.calc_optimal_block_size(arg_map, m_code_body.c_str(), sizeBlock, sharedMemBytes);
	}

	bool Kernel::calc_number_blocks(const DeviceViewable** args, int sizeBlock, int& numBlocks, unsigned sharedMemBytes)
	{
		Context& ctx = Context::get_context();
		std::vector<CapturedDeviceViewable> arg_map(m_param_names.size());
		for (size_t i = 0; i < m_param_names.size(); i++)
		{
			arg_map[i].obj_name = m_param_names[i].c_str();
			arg_map[i].obj = args[i];
		}
		return ctx.calc_number_blocks(arg_map, m_code_body.c_str(), sizeBlock, numBlocks, sharedMemBytes);
	}

	bool Kernel::launch(dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes)
	{
		Context& ctx = Context::get_context();
		std::vector<CapturedDeviceViewable> arg_map(m_param_names.size());
		for (size_t i = 0; i < m_param_names.size(); i++)
		{
			arg_map[i].obj_name = m_param_names[i].c_str();
			arg_map[i].obj = args[i];
		}
		return ctx.launch_kernel(gridDim, blockDim, arg_map, m_code_body.c_str(), sharedMemBytes);
	}

}