#pragma once

#include "DeviceViewable.h"
#include "Context.h"

namespace CUInline
{
	class DVCombine : public DeviceViewable
	{
	public:
		DVCombine(const std::vector<CapturedDeviceViewable>& elem_map, const char* operations);
		virtual ViewBuf view() const;

	private:
		std::vector<ViewBuf> m_view_elems;
		std::vector<size_t> m_offsets;
	};
}
