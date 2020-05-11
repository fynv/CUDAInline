import CUDAInline as cui
import numpy as np

# interface with numpy
harr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
darr = cui.device_vector_from_numpy(harr)
print(darr.to_host())

# C data type
print(darr.name_view_cls())

harr2 = np.array([6,7,8,9,10], dtype='int32')
darr2 = cui.device_vector_from_numpy(harr2)

# kernel with auto parameters, launched twice with different types
kernel = cui.Kernel(['arr_in', 'arr_out', 'k'],
    '''
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= arr_in.size()) return;
    arr_out[idx] = arr_in[idx]*k;
    ''')

darr_out = cui.device_vector('float', 5)
kernel.launch(1,128, [darr, darr_out, cui.DVFloat(10.0)])
print (darr_out.to_host())

darr_out = cui.device_vector('int32_t', 5)
kernel.launch(1,128, [darr2, darr_out, cui.DVInt32(5)])
print (darr_out.to_host())

# create a vector from python list with C type specified
darr3 = cui.device_vector_from_list([3.0, 5.0, 7.0, 9.0 , 11.0], 'float')
print(darr3.to_host())

drange = darr3.range(1,4)
print(drange.to_host())
