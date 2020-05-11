import numpy as np
from .Context import *
from .DeviceViewable import *
from .DVBuffer import *
from .DVCombine import DVCombine_Create

class DVVector(DeviceViewable):
    def __init__(self, elem_cls, elem_size, buf):
        self.m_elem_cls = elem_cls
        self.m_ref_type = elem_cls + '&'
        self.m_elem_size = elem_size
        self.m_size = DVUInt64(int(buf.size()/elem_size))
        self.m_buf = buf
        self.m_cptr = DVCombine_Create({'_size':  self.m_size, '_data': self.m_buf},
'''
    typedef {0} value_t;
    typedef {1} ref_t;
    __device__ size_t size() const
    {{
        return _size;
    }}
    __device__ ref_t operator [](size_t idx)
    {{
        return ((value_t*)_data)[idx];
    }}
'''.format(self.m_elem_cls, self.m_ref_type))

    def name_elem_cls(self):
        return self.m_elem_cls

    def elem_size(self):
        return self.m_elem_size

    def size(self):
        return self.m_size.value()

    def to_host(self, begin = 0, end = -1):
        if self.m_elem_cls=='int8_t':
            nptype = np.int8
        elif self.m_elem_cls=='uint8_t':
            nptype = np.uint8
        elif self.m_elem_cls=='int16_t':
            nptype = np.int16
        elif self.m_elem_cls=='uint16_t':
            nptype = np.uint16
        elif self.m_elem_cls=='int32_t':
            nptype = np.int32
        elif self.m_elem_cls=='uint32_t':
            nptype = np.uint32
        elif self.m_elem_cls=='int64_t':
            nptype = np.int64
        elif self.m_elem_cls=='uint64_t':
            nptype = np.uint64
        elif self.m_elem_cls=='float':
            nptype = np.float32
        elif self.m_elem_cls=='double':
            nptype = np.float64
        elif self.m_elem_cls=='bool':
            nptype = np.bool
        if end == -1:
            end = self.size()
        ret = np.empty(end - begin, dtype=nptype)
        self.m_buf.to_host(ret.__array_interface__['data'][0], begin*self.m_elem_size, end*self.m_elem_size)
        return ret

    def range(self, begin = 0, end = -1):
        if end == -1:
            end = self.size()
        buf = self.m_buf.range(begin*self.m_elem_size, end*self.m_elem_size)
        return DVVector(self.m_elem_cls, self.m_elem_size, buf)

def device_vector(elem_cls, size, ptr_host_data=None):
    elem_size = Size_Of(elem_cls)
    buf_size = elem_size*size
    buf = device_buffer(buf_size, ptr_host_data)
    return DVVector(elem_cls, elem_size, buf)

def device_vector_from_numpy(nparr):
    if nparr.dtype == np.int8:
        elem_cls = 'int8_t'
    elif nparr.dtype == np.uint8:
        elem_cls = 'uint8_t'
    elif nparr.dtype == np.int16:
        elem_cls = 'int16_t'
    elif nparr.dtype == np.uint16:
        elem_cls = 'uint16_t'
    elif nparr.dtype == np.int32:
        elem_cls = 'int32_t'
    elif nparr.dtype == np.uint32:
        elem_cls = 'uint32_t'       
    elif nparr.dtype == np.int64:
        elem_cls = 'int64_t'
    elif nparr.dtype == np.uint64:
        elem_cls = 'uint64_t'   
    elif nparr.dtype == np.int64:
        elem_cls = 'int64_t'
    elif nparr.dtype == np.uint64:
        elem_cls = 'uint64_t'   
    elif nparr.dtype == np.float32:
        elem_cls = 'float'
    elif nparr.dtype == np.float64:
        elem_cls = 'double'
    elif nparr.dtype == np.bool:
        elem_cls = 'bool'
    size = len(nparr)
    ptr_host_data = nparr.__array_interface__['data'][0]
    return device_vector(elem_cls, size, ptr_host_data)

def device_vector_from_list(lst, elem_cls):
    if elem_cls=='int8_t':
        nptype = np.int8
    elif elem_cls=='uint8_t':
        nptype = np.uint8
    elif elem_cls=='int16_t':
        nptype = np.int16
    elif elem_cls=='uint16_t':
        nptype = np.uint16
    elif elem_cls=='int32_t':
        nptype = np.int32
    elif elem_cls=='uint32_t':
        nptype = np.uint32
    elif elem_cls=='int64_t':
        nptype = np.int64
    elif elem_cls=='uint64_t':
        nptype = np.uint64
    elif elem_cls=='float':
        nptype = np.float32
    elif elem_cls=='double':
        nptype = np.float64
    elif elem_cls=='bool':
        nptype = np.bool
    nparr = np.array(lst, dtype=nptype)
    size = len(lst)
    ptr_host_data = nparr.__array_interface__['data'][0]
    return device_vector(elem_cls, size, ptr_host_data)

def device_vector_from_dvs(lst_dv):
    elem_cls = lst_dv[0].name_view_cls()
    elem_size = Size_Of(elem_cls)
    buf = device_buffer_from_dvs(lst_dv)
    return DVVector(elem_cls, elem_size, buf)

def device_vector_from_numba(nbarr):
    if nbarr.dtype == np.int8:
        elem_cls = 'int8_t'
    elif nbarr.dtype == np.uint8:
        elem_cls = 'uint8_t'
    elif nbarr.dtype == np.int16:
        elem_cls = 'int16_t'
    elif nbarr.dtype == np.uint16:
        elem_cls = 'uint16_t'
    elif nbarr.dtype == np.int32:
        elem_cls = 'int32_t'
    elif nbarr.dtype == np.uint32:
        elem_cls = 'uint32_t'       
    elif nbarr.dtype == np.int64:
        elem_cls = 'int64_t'
    elif nbarr.dtype == np.uint64:
        elem_cls = 'uint64_t'   
    elif nbarr.dtype == np.int64:
        elem_cls = 'int64_t'
    elif nbarr.dtype == np.uint64:
        elem_cls = 'uint64_t'   
    elif nbarr.dtype == np.float32:
        elem_cls = 'float'
    elif nbarr.dtype == np.float64:
        elem_cls = 'double'
    elif nbarr.dtype == np.bool:
        elem_cls = 'bool'
    size = nbarr.size
    elem_size = Size_Of(elem_cls)
    buf_size = elem_size*size
    buf = DVNumbaBuffer(nbarr, buf_size)
    return DVVector(elem_cls, elem_size, buf)
