from .Native import native
if native.n_cudainline_try_init()==0:
	raise ImportError('cannot import CUDAInline')

from .Context import *
from .DeviceViewable import *
from .DVVector import *
