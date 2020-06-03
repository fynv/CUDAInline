#include "cuda_wrapper.h"
#include "DVBuffer.h"

namespace CUInline
{
	DVBufferLike::DVBufferLike()
	{
		m_size = 0;
		m_data = nullptr;
		m_name_view_cls = "void*";
	}

	void DVBufferLike::from_host(void* hdata)
	{
		if (m_size>0)
			cuMemcpyHtoD((CUdeviceptr)m_data, hdata, m_size);
	}

	void DVBufferLike::to_host(void* hdata, size_t begin, size_t end) const
	{
		if (end == (size_t)(-1) || end > m_size) end = m_size;
		size_t n = end - begin;
		if (n>0)
			cuMemcpyDtoH(hdata, (CUdeviceptr)((char*)m_data + begin), n);
	}

	ViewBuf DVBufferLike::view() const
	{
		ViewBuf buf(sizeof(void*));
		void **pview = (void**)buf.data();
		*pview = m_data;
		return buf;
	}

	DVBuffer::DVBuffer(size_t size, void* hdata)
	{
		TryInit();
		m_size = size;
		CUdeviceptr dptr;
		cuMemAlloc(&dptr, m_size);
		m_data = (void*)dptr;
		if (hdata)
			cuMemcpyHtoD(dptr, hdata, m_size);
		else
			cuMemsetD8(dptr, 0, m_size);
	}

	DVBuffer::~DVBuffer()
	{
		cuMemFree((CUdeviceptr)m_data);
	}

	DVBufferRange::DVBufferRange(size_t size, void* ddata)
	{
		m_size = size;
		m_data = ddata;
	}

	DVBufferRange::DVBufferRange(const DVBufferLike& vec, size_t begin, size_t end)
	{
		if (end == (size_t)(-1) || end > vec.size()) end = vec.size();
		m_size = end - begin;
		m_data = (char*)vec.data() + begin;		
	}

}
