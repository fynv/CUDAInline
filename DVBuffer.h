#pragma once

#include "DeviceViewable.h"
#include "Context.h"

namespace CUInline
{
	class DVBufferLike : public DeviceViewable
	{
	public:
		size_t size() const { return m_size; }
		void* data() const { return m_data; }

		DVBufferLike();
		~DVBufferLike() {}

		void from_host(void* hdata);
		void to_host(void* hdata, size_t begin = 0, size_t end = (size_t)(-1)) const;
		virtual ViewBuf view() const;

	protected:
		size_t m_size;
		void* m_data;
	};

	class DVBuffer : public DVBufferLike
	{
	public:
		DVBuffer(size_t size, void* hdata = nullptr);
		~DVBuffer();
	};

	class DVBufferRange : public DVBufferLike
	{
	public:
		DVBufferRange(size_t size, void* ddata);
		DVBufferRange(const DVBufferLike& vec, size_t begin = 0, size_t end = (size_t)(-1));
	};

}
