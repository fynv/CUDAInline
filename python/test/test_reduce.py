import CUDAInline as cui
import numpy as np

darr = cui.device_vector_from_list(range(1,1025), 'int32_t')

kernel = cui.Kernel(['dst', 'src', 'n'],
'''
    extern __shared__ decltype(dst)::value_t s_buf[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    decltype(dst)::value_t& here=s_buf[tid];
    if (i<n) here=src[i];
    __syncthreads();
    for (unsigned s = blockDim.x/2; s>0; s>>=1)
    {
    	if (tid < s && i+s<n)
    		here += s_buf[tid + s];
    	__syncthreads();
    }
    if (tid==0) dst[blockIdx.x] = here;
''')


BLOCK_SIZE = 256
size_shared = int(darr.elem_size()*BLOCK_SIZE);

dst  = darr
while dst.size()>1:
	src = dst
	n = src.size()
	blocks = int((n + BLOCK_SIZE - 1) / BLOCK_SIZE)
	dst = cui.device_vector("int32_t", blocks)
	kernel.launch(blocks, BLOCK_SIZE, [dst, src, cui.DVUInt32(n)], size_shared)

print(dst.to_host()[0])

