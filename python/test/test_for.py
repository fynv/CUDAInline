import CUDAInline as cui
import numpy as np

# interface with numpy
harr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
darr = cui.device_vector_from_numpy(harr)

harr2 = np.array([6,7,8,9,10], dtype='int32')
darr2 = cui.device_vector_from_numpy(harr2)

# test launching non-templated for
forLoop = cui.For(['arr_in','arr_out','k'], "idx",
	'''
	arr_out[idx] = arr_in[idx]*k;
	''')

darr_out = cui.device_vector('float', 5)
forLoop.launch(0, 5, [darr, darr_out, cui.DVFloat(10.0)])
print (darr_out.to_host())

darr_out = cui.device_vector('int32_t', 5)
forLoop.launch_n(5, [darr2, darr_out, cui.DVInt32(5)])
print (darr_out.to_host())
