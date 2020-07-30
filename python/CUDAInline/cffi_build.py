import os
if os.path.exists('CUDAInline/cffi.py'):
	os.remove('CUDAInline/cffi.py')

import cffi
ffibuilder = cffi.FFI()
ffibuilder.set_source("CUDAInline.cffi", None)

ffibuilder.cdef("""    
// utils
void* n_string_array_create(unsigned long long size, const char* const* strs);
unsigned long long n_string_array_size(void* ptr_arr);
void n_string_array_destroy(void* ptr_arr);
void* n_pointer_array_create(unsigned long long size, const void* const* ptrs);
unsigned long long n_pointer_array_size(void* ptr_arr);
void n_pointer_array_destroy(void* ptr_arr);
void* n_dim3_create(unsigned x, unsigned y, unsigned z);
void n_dim3_destroy(void* cptr);

// Context
void n_set_libnvrtc_path(const char* path);
int n_cudainline_try_init();
void n_set_verbose(unsigned verbose);
unsigned long long n_size_of(const char* cls);
void n_add_include_dir(const char* dir);
void n_add_built_in_header(const char* filename, const char* filecontent);
void n_add_inlcude_filename(const char* fn);
void n_add_code_block(const char* line);
void n_add_constant_object(const char* name, void* cptr);
void n_wait();

void* n_kernel_create(void* ptr_param_list, const char* body, unsigned type_locked);
void n_kernel_destroy(void* cptr);
int n_kernel_num_params(void* cptr);
int n_kernel_calc_optimal_block_size(void* ptr_kernel, void* ptr_arg_list, unsigned sharedMemBytes);
int n_kernel_calc_number_blocks(void* ptr_kernel, void* ptr_arg_list, int sizeBlock, unsigned sharedMemBytes);
int n_kernel_launch(void* ptr_kernel, void* ptr_gridDim, void* ptr_blockDim, void* ptr_arg_list, int sharedMemBytes);

// DeviceViewable
const char* n_dv_name_view_cls(void* cptr);
void n_dv_destroy(void* cptr);
void* n_dvint8_create(int v);
int n_dvint8_value(void* cptr);
void* n_dvuint8_create(unsigned v);
unsigned n_dvuint8_value(void* cptr);
void* n_dvint16_create(int v);
int n_dvint16_value(void* cptr);
void* n_dvuint16_create(unsigned v);
unsigned n_dvuint16_value(void* cptr);
void* n_dvint32_create(int v);
int n_dvint32_value(void* cptr);
void* n_dvuint32_create(unsigned v);
unsigned n_dvuint32_value(void* cptr);
void* n_dvint64_create(long long v);
long long n_dvint64_value(void* cptr);
void* n_dvuint64_create(unsigned long long v);
unsigned long long n_dvuint64_value(void* cptr);
void* n_dvfloat_create(float v);
float n_dvfloat_value(void* cptr);
void* n_dvdouble_create(double v);
double n_dvdouble_value(void* cptr);
void* n_dvbool_create(int v);
int n_dvbool_value(void* cptr);

// DVBuffer
unsigned long long n_dvbufferlike_size(void* cptr);
void n_dvbufferlike_from_host(void* cptr, void* hdata);
void n_dvbufferlike_to_host(void* cptr, void* hdata, unsigned long long begin, unsigned long long end);
void* n_dvbuffer_create(unsigned long long size, void* hdata);
void* n_dvbuffer_from_dvs(void* ptr_dvs);
void* n_dvbuffer_range_create(unsigned long long size, void* data);
void* n_dvbuffer_range_from_dvbuffer(void* ptr_in, unsigned long long begin, unsigned long long end);

// DVCombine
void* n_dvcombine_create(void* ptr_dvs, void* ptr_names, const char* operations);
""")


ffibuilder.compile()
