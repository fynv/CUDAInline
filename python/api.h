#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define PY_CUDAInline_API __declspec(dllexport)
#else
#define PY_CUDAInline_API 
#endif


extern "C"
{
	// utils
	PY_CUDAInline_API void* n_string_array_create(unsigned long long size, const char* const* strs);
	PY_CUDAInline_API unsigned long long n_string_array_size(void* ptr_arr);
	PY_CUDAInline_API void n_string_array_destroy(void* ptr_arr);
	PY_CUDAInline_API void* n_pointer_array_create(unsigned long long size, const void* const* ptrs);
	PY_CUDAInline_API unsigned long long n_pointer_array_size(void* ptr_arr);
	PY_CUDAInline_API void n_pointer_array_destroy(void* ptr_arr);
	PY_CUDAInline_API void* n_dim3_create(unsigned x, unsigned y, unsigned z);
	PY_CUDAInline_API void n_dim3_destroy(void* cptr);

	// Context
	PY_CUDAInline_API void n_set_libnvrtc_path(const char* path);	
	PY_CUDAInline_API int n_cudainline_try_init();
	PY_CUDAInline_API void n_set_verbose(unsigned verbose);
	PY_CUDAInline_API unsigned long long n_size_of(const char* cls);
	PY_CUDAInline_API void n_add_include_dir(const char* dir);
	PY_CUDAInline_API void n_add_built_in_header(const char* filename, const char* filecontent);
	PY_CUDAInline_API void n_add_inlcude_filename(const char* fn);
	PY_CUDAInline_API void n_add_code_block(const char* line);
	PY_CUDAInline_API void n_add_constant_object(const char* name, void* cptr);
	PY_CUDAInline_API void n_wait();

	PY_CUDAInline_API void* n_kernel_create(void* ptr_param_list, const char* body);
	PY_CUDAInline_API void n_kernel_destroy(void* cptr);
	PY_CUDAInline_API int n_kernel_num_params(void* cptr);
	PY_CUDAInline_API int n_kernel_calc_optimal_block_size(void* ptr_kernel, void* ptr_arg_list, unsigned sharedMemBytes);
	PY_CUDAInline_API int n_kernel_calc_number_blocks(void* ptr_kernel, void* ptr_arg_list, int sizeBlock, unsigned sharedMemBytes);
	PY_CUDAInline_API int n_kernel_launch(void* ptr_kernel, void* ptr_gridDim, void* ptr_blockDim, void* ptr_arg_list, int sharedMemBytes);

	// DeviceViewable
	PY_CUDAInline_API const char* n_dv_name_view_cls(void* cptr);
	PY_CUDAInline_API void n_dv_destroy(void* cptr);
	PY_CUDAInline_API void* n_dvint8_create(int v);
	PY_CUDAInline_API int n_dvint8_value(void* cptr);
	PY_CUDAInline_API void* n_dvuint8_create(unsigned v);
	PY_CUDAInline_API unsigned n_dvuint8_value(void* cptr);
	PY_CUDAInline_API void* n_dvint16_create(int v);
	PY_CUDAInline_API int n_dvint16_value(void* cptr);
	PY_CUDAInline_API void* n_dvuint16_create(unsigned v);
	PY_CUDAInline_API unsigned n_dvuint16_value(void* cptr);
	PY_CUDAInline_API void* n_dvint32_create(int v);
	PY_CUDAInline_API int n_dvint32_value(void* cptr);
	PY_CUDAInline_API void* n_dvuint32_create(unsigned v);
	PY_CUDAInline_API unsigned n_dvuint32_value(void* cptr);
	PY_CUDAInline_API void* n_dvint64_create(long long v);
	PY_CUDAInline_API long long n_dvint64_value(void* cptr);
	PY_CUDAInline_API void* n_dvuint64_create(unsigned long long v);
	PY_CUDAInline_API unsigned long long n_dvuint64_value(void* cptr);
	PY_CUDAInline_API void* n_dvfloat_create(float v);
	PY_CUDAInline_API float n_dvfloat_value(void* cptr);
	PY_CUDAInline_API void* n_dvdouble_create(double v);
	PY_CUDAInline_API double n_dvdouble_value(void* cptr);
	PY_CUDAInline_API void* n_dvbool_create(int v);
	PY_CUDAInline_API int n_dvbool_value(void* cptr);

	// DVBuffer
	PY_CUDAInline_API unsigned long long n_dvbufferlike_size(void* cptr);
	PY_CUDAInline_API void n_dvbufferlike_from_host(void* cptr, void* hdata);
	PY_CUDAInline_API void n_dvbufferlike_to_host(void* cptr, void* hdata, unsigned long long begin, unsigned long long end);
	PY_CUDAInline_API void* n_dvbuffer_create(unsigned long long size, void* hdata);
	PY_CUDAInline_API void* n_dvbuffer_from_dvs(void* ptr_dvs);
	PY_CUDAInline_API void* n_dvbuffer_range_create(unsigned long long size, void* data);
	PY_CUDAInline_API void* n_dvbuffer_range_from_dvbuffer(void* ptr_in, unsigned long long begin, unsigned long long end);

	// DVCombine
	PY_CUDAInline_API void* n_dvcombine_create(void* ptr_dvs, void* ptr_names, const char* operations);

}
