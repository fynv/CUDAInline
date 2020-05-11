from .Native import native
from .DeviceViewable import *
from .utils import *

def set_libnvrtc_path(path):
    native.n_set_libnvrtc_path(path.encode('utf-8'))

def Set_Verbose(verbose=True):
    native.n_set_verbose(verbose)

def Size_Of(clsname):
    return native.n_size_of(clsname.encode('utf-8'))

def Add_Include_Dir(path):
    native.n_add_include_dir(path.encode('utf-8'))

def Add_Built_In_Header(filename, filecontent):
    native.n_add_built_in_header(filename.encode('utf-8'), filecontent.encode('utf-8'))

def Add_Inlcude_Filename(filename):
    native.n_add_inlcude_filename(filename.encode('utf-8'))

def Add_Code_Block(code):
    native.n_add_code_block(code.encode('utf-8'))

def Add_Constant_Object(name, dv):
    native.n_add_constant_object(name.encode('utf-8'), dv.m_cptr)

def Wait():
    native.n_wait()

class Kernel:
    def __init__(self, param_names, body):
        o_param_names = StrArray(param_names)
        self.m_cptr = native.n_kernel_create(o_param_names.m_cptr, body.encode('utf-8'))

    def __del__(self):
        native.n_kernel_destroy(self.m_cptr)

    def num_params(self):
        return native.n_kernel_num_params(self.m_cptr)

    def calc_optimal_block_size(self, args, sharedMemBytes=0):
        arg_list = ObjArray(args)
        return native.n_kernel_calc_optimal_block_size(
            self.m_cptr, arg_list.m_cptr, sharedMemBytes)

    def calc_number_blocks(self, args, size_block, sharedMemBytes=0):
        arg_list = ObjArray(args)
        return native.n_kernel_calc_number_blocks(
            self.m_cptr, 
            arg_list.m_cptr, 
            size_block,
            sharedMemBytes)

    def launch(self, gridDim, blockDim, args, sharedMemBytes=0):
        d_gridDim = Dim3(gridDim)
        d_blockDim = Dim3(blockDim)
        arg_list = ObjArray(args)
        native.n_kernel_launch(
            self.m_cptr, 
            d_gridDim.m_cptr, 
            d_blockDim.m_cptr, 
            arg_list.m_cptr, 
            sharedMemBytes)

class For:
    def __init__(self, param_names, name_iter, body):
        self.m_param_names = StrArray(param_names)
        self.m_name_iter = name_iter
        self.m_code_body = body

    def num_params(self):
        return m_param_names.size()

    class InnerProcedural(DeviceViewable):
        def __init__(self, o_param_names, o_elems, name_iter, body):
            operations =  "    __device__ inline void inner(size_t {})\n    {{\n".format(name_iter)
            operations += body
            operations += "    }\n"
            self.m_cptr = native.n_dvcombine_create(o_elems.m_cptr, o_param_names.m_cptr, operations.encode('utf-8'))


    def launch(self, begin, end, args):        
        dvbegin = DVUInt64(begin)
        dvend = DVUInt64(end)
        o_args = ObjArray(args)
        func =  self.InnerProcedural(self.m_param_names, o_args, self.m_name_iter, self.m_code_body)
        kernel = Kernel(['begin', 'end', 'func'],
    '''
    size_t tid =  threadIdx.x + blockIdx.x*blockDim.x + begin;
    if(tid>=end) return;
    func.inner(tid);
    ''')
        sizeBlock = kernel.calc_optimal_block_size([dvbegin, dvend, func]);
        numBlocks = int((end - begin + sizeBlock - 1) / sizeBlock)
        kernel.launch(numBlocks, sizeBlock, [dvbegin, dvend, func])

    def launch_n(self, n, args):
        dv_n = DVUInt64(n)
        o_args = ObjArray(args)
        func =  self.InnerProcedural(self.m_param_names, o_args, self.m_name_iter, self.m_code_body)
        kernel = Kernel(['n', 'func'],
    '''
    size_t tid =  threadIdx.x + blockIdx.x*blockDim.x;
    if(tid>=n) return;
    func.inner(tid);
    ''')
        sizeBlock = kernel.calc_optimal_block_size([dv_n, func]);
        numBlocks = int((n + sizeBlock - 1) / sizeBlock)
        kernel.launch(numBlocks, sizeBlock, [dv_n, func])






