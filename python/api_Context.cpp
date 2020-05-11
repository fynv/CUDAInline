#include "api.h"
#include "Context.h"
using namespace CUInline;
#include <string>
#include <vector>

typedef std::vector<std::string> StrArray;
typedef std::vector<const DeviceViewable*> PtrArray;

void n_set_libnvrtc_path(const char* path)
{
	set_libnvrtc_path(path);
}

void n_set_verbose(unsigned verbose)
{
	SetVerbose(verbose != 0);
}

unsigned long long n_size_of(const char* cls)
{
	return SizeOf(cls);
}

void n_add_include_dir(const char* dir)
{
	AddIncludeDir(dir);
}

void n_add_built_in_header(const char* filename, const char* filecontent)
{
	AddBuiltInHeader(filename, filecontent);
}

void n_add_inlcude_filename(const char* fn)
{
	AddInlcudeFilename(fn);
}

void n_add_code_block(const char* line)
{
	AddCodeBlock(line);
}

void n_add_constant_object(const char* name, void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	AddConstantObject(name, *dv);
}

void n_wait()
{
	Wait();
}

void* n_kernel_create(void* ptr_param_list, const char* body)
{
	StrArray* param_list = (StrArray*)ptr_param_list;
	size_t num_params = param_list->size();
	std::vector<const char*> params(num_params);
	for (size_t i = 0; i < num_params; i++)
		params[i] = (*param_list)[i].c_str();
	Kernel* cptr = new Kernel(params, body);
	return cptr;
}

void n_kernel_destroy(void* cptr)
{
	Kernel* kernel = (Kernel*)cptr;
	delete kernel;
}

int n_kernel_num_params(void* cptr)
{
	Kernel* kernel = (Kernel*)cptr;
	return (int)kernel->num_params();
}


int n_kernel_calc_optimal_block_size(void* ptr_kernel, void* ptr_arg_list, unsigned sharedMemBytes)
{
	Kernel* kernel = (Kernel*)ptr_kernel;
	size_t num_params = kernel->num_params();
	PtrArray* arg_list = (PtrArray*)ptr_arg_list;

	size_t size = arg_list->size();
	if (num_params != size)
	{
		printf("Wrong number of arguments received. %d required, %d received.", (int)num_params, (int)size);
		return -1;
	}

	int sizeBlock;
	if (kernel->calc_optimal_block_size(arg_list->data(), sizeBlock, sharedMemBytes))
		return sizeBlock;
	else
		return -1;
}


int n_kernel_calc_number_blocks(void* ptr_kernel, void* ptr_arg_list, int sizeBlock, unsigned sharedMemBytes)
{
	Kernel* kernel = (Kernel*)ptr_kernel;
	size_t num_params = kernel->num_params();
	PtrArray* arg_list = (PtrArray*)ptr_arg_list;

	size_t size = arg_list->size();
	if (num_params != size)
	{
		printf("Wrong number of arguments received. %d required, %d received.", (int)num_params, (int)size);
		return -1;
	}

	int numBlocks;
	if (kernel->calc_number_blocks(arg_list->data(), sizeBlock, numBlocks, sharedMemBytes))
		return numBlocks;
	else
		return -1;
}

int n_kernel_launch(void* ptr_kernel, void* ptr_gridDim, void* ptr_blockDim, void* ptr_arg_list, int sharedMemBytes)
{
	Kernel* kernel = (Kernel*)ptr_kernel;
	size_t num_params = kernel->num_params();

	dim_type* gridDim = (dim_type*)ptr_gridDim;
	dim_type* blockDim = (dim_type*)ptr_blockDim;

	PtrArray* arg_list = (PtrArray*)ptr_arg_list;

	size_t size = arg_list->size();
	if (num_params != size)
	{
		printf("Wrong number of arguments received. %d required, %d received.", (int)num_params, (int)size);
		return -1;
	}

	if (kernel->launch(*gridDim, *blockDim, arg_list->data(), sharedMemBytes))
		return 0;
	else
		return -1;
}


