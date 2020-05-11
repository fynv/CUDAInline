import os
import sys
import site
from .cffi import ffi

if os.name == 'nt':
    fn_cudainline = 'PyCUDAInline.dll'
elif os.name == "posix":
    fn_cudainline = 'libPyCUDAInline.so'

path_cudainline = os.path.dirname(__file__)+"/"+fn_cudainline

native = ffi.dlopen(path_cudainline)

