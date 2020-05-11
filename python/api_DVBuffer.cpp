#include "api.h"
#include "DVBuffer.h"
using namespace CUInline;

typedef std::vector<const DeviceViewable*> PtrArray;

unsigned long long n_dvbufferlike_size(void* cptr)
{
	DVBufferLike* dvbuf = (DVBufferLike*)cptr;
	return dvbuf->size();
}

void n_dvbufferlike_from_host(void* cptr, void* hdata)
{
	DVBufferLike* dvbuf = (DVBufferLike*)cptr;
	dvbuf->from_host(hdata);
}

void n_dvbufferlike_to_host(void* cptr, void* hdata, unsigned long long begin, unsigned long long end)
{
	DVBufferLike* dvbuf = (DVBufferLike*)cptr;
	dvbuf->to_host(hdata, begin, end);
}

void* n_dvbuffer_create(unsigned long long size, void* hdata)
{
	return new DVBuffer(size, hdata);
}

void* n_dvbuffer_from_dvs(void* ptr_dvs)
{
	PtrArray* dvs = (PtrArray*)ptr_dvs;
	size_t num_items = dvs->size();
	if (num_items < 1) return nullptr;
	std::string elem_cls = (*dvs)[0]->name_view_cls();
	for (size_t i = 1; i < num_items; i++)
	{
		if ((*dvs)[i]->name_view_cls() != elem_cls)
			return nullptr;
	}
	size_t elem_size = SizeOf(elem_cls.c_str());
	std::vector<char> buf(elem_size*num_items);
	for (size_t i = 0; i < num_items; i++)
	{
		memcpy(buf.data() + elem_size * i, (*dvs)[i]->view().data(), elem_size);
	}
	return new DVBuffer(num_items*elem_size, buf.data());
}

void* n_dvbuffer_range_create(unsigned long long size, void* data)
{
	return new DVBufferRange(size, data);
}

void* n_dvbuffer_range_from_dvbuffer(void* ptr_in, unsigned long long begin, unsigned long long end)
{
	DVBufferLike* dvbuf = (DVBufferLike*)ptr_in;
	return new DVBufferRange(*dvbuf, begin, end);
}
