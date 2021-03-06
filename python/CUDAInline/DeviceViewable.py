from .Native import ffi, native

class DeviceViewable:
    def name_view_cls(self):
        return ffi.string(native.n_dv_name_view_cls(self.m_cptr)).decode('utf-8')
    def __del__(self):
        native.n_dv_destroy(self.m_cptr)
    def value(self):
        s_type = self.name_view_cls()
        return '[Device-viewable object, type: %s]'%s_type

class DVInt8(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvint8_create(value)
    def value(self):
        return native.n_dvint8_value(self.m_cptr)

class DVUInt8(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvuint8_create(value)
    def value(self):
        return native.n_dvuint8_value(self.m_cptr)

class DVInt16(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvint16_create(value)
    def value(self):
        return native.n_dvint16_value(self.m_cptr)

class DVUInt16(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvuint16_create(value)
    def value(self):
        return native.n_dvuint16_value(self.m_cptr)

class DVInt32(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvint32_create(value)
    def value(self):
        return native.n_dvint32_value(self.m_cptr)

class DVUInt32(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvuint32_create(value)
    def value(self):
        return native.n_dvuint32_value(self.m_cptr)

class DVInt64(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvint64_create(value)
    def value(self):
        return native.n_dvint64_value(self.m_cptr)

class DVUInt64(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvuint64_create(value)
    def value(self):
        return native.n_dvuint64_value(self.m_cptr)

class DVFloat(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvfloat_create(value)
    def value(self):
        return native.n_dvfloat_value(self.m_cptr)

class DVDouble(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvdouble_create(value)
    def value(self):
        return native.n_dvdouble_value(self.m_cptr)

class DVBool(DeviceViewable):
    def __init__(self, value):
        self.m_cptr = native.n_dvbool_create(value)
    def value(self):
        return native.n_dvbool_value(self.m_cptr)!=0


