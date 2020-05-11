#include "Context.h"
#include "DVBuffer.h"
#include "DVCombine.h"

using namespace CUInline;

class Vector
{
public:
	const std::string& name_elem_cls() const { return m_elem_cls; }
	const std::string& name_ref_type() const { return m_ref_type; }
	size_t elem_size() const { return m_elem_size; }
	size_t size() const { return *(size_t*)m_size.view().data(); }

	Vector(const char* elem_cls, size_t size, void* hdata = nullptr)
		: m_elem_cls(elem_cls), 
		m_ref_type(m_elem_cls + "&"), 
		m_elem_size(SizeOf(elem_cls)), 
		m_size(size), 
		m_buf(size*m_elem_size, hdata),
		m_dv({ {"_size", &m_size}, {"_data", &m_buf} },
		(std::string("")+"\
	typedef " + m_elem_cls + " value_t;\n\
	typedef " + m_ref_type + " ref_t;\n\
	__device__ size_t size() const\n\
	{\n\
		return _size;\n\
	}\n\
	__device__ ref_t operator [](size_t idx)\n\
	{\n\
		return ((value_t*)_data)[idx];\n\
	}\n").c_str()){}

	void to_host(void* hdata, size_t begin = 0, size_t end = (size_t)(-1)) const
	{
		size_t _size = size();
		if (end == (size_t)(-1) || end > _size) end = _size;
		m_buf.to_host(hdata, begin*m_elem_size, end*m_elem_size);
	}

	const DVCombine& getDV() { return m_dv; }

private:
	std::string m_elem_cls;
	std::string m_ref_type;
	size_t m_elem_size;
	DVSizeT m_size;
	DVBuffer m_buf;
	DVCombine m_dv;
};

int main()
{
	Kernel ker(
		{ "arr_in", "arr_out", "k" },
		"    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
		"    if (idx >= arr_in.size()) return;\n"
		"    arr_out[idx] = arr_in[idx]*k;\n");

	float test_f[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0 };

	Vector dvec_in_f("float", 5, test_f);
	Vector dvec_out_f("float", 5);
	DVFloat k1(10.0);
	const DeviceViewable* args_f[] = { &dvec_in_f.getDV(), &dvec_out_f.getDV(), &k1 };
	ker.launch({ 1, 1, 1 }, { 128, 1, 1 }, args_f);
	dvec_out_f.to_host(test_f);
	printf("%f %f %f %f %f\n", test_f[0], test_f[1], test_f[2], test_f[3], test_f[4]);

	return 0;
}
