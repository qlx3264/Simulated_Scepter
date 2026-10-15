"""DirectML 适配器枚举与选择，不创建推理会话或访问游戏。"""

import ctypes as ct
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

from onnxruntime import get_available_providers


class Luid(ct.Structure):
    _fields_ = [("low", ct.c_uint32), ("high", ct.c_int32)]

    def __str__(self):
        return f"0x{self.high & 0xffffffff:08X}_0x{self.low:08X}"


class AdapterDesc(ct.Structure):
    _fields_ = [
        ("name", ct.c_wchar * 128),
        ("vendor", ct.c_uint32), ("device", ct.c_uint32),
        ("subsystem", ct.c_uint32), ("revision", ct.c_uint32),
        ("video_memory", ct.c_size_t), ("system_memory", ct.c_size_t),
        ("shared_memory", ct.c_size_t), ("luid", Luid), ("flags", ct.c_uint32),
    ]


class AdapterName(ct.Structure):
    # DISPLAYCONFIG_ADAPTER_NAME 的头部和设备接口路径。
    _fields_ = [
        ("type", ct.c_uint32), ("size", ct.c_uint32),
        ("luid", Luid), ("id", ct.c_uint32), ("path", ct.c_wchar * 128),
    ]


@dataclass(frozen=True)
class Adapter:
    index: int
    name: str
    luid: str
    key: str

    @property
    def label(self):
        return f"{self.name}（DXGI {self.index}，LUID {self.luid}）"


def guid(value):
    return (ct.c_ubyte * 16).from_buffer_copy(UUID(value).bytes_le)


def invoke(pointer, slot, result, types, *args):
    """按 Windows COM ABI 调用接口方法，槽位来自对应的 SDK 接口声明。"""
    table = ct.cast(pointer, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    method = ct.WINFUNCTYPE(result, ct.c_void_p, *types)(table[slot])
    return method(pointer, *args)


def check_result(result, operation):
    if result < 0:
        raise OSError(f"{operation} 失败：HRESULT 0x{result & 0xffffffff:08X}")


@contextmanager
def com_pointer():
    """保证工厂、列表和适配器在所有退出路径释放 COM 引用。"""
    pointer = ct.c_void_p()
    try:
        yield pointer
    finally:
        if pointer:
            invoke(pointer, 2, ct.c_uint32, [])  # IUnknown::Release


def list_adapters():
    """按 ORT device_id 对应的 DXGI 顺序枚举硬件，保留原始索引。

    Returns:
        Adapter 列表。key 优先使用设备接口路径，避免排序或重启改变编号；
        无法获取接口路径的设备使用本次系统会话的 LUID，不猜测同名设备身份。
    """
    create = ct.WinDLL("dxgi").CreateDXGIFactory1
    create.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    create.restype = ct.c_int32
    get_name = ct.WinDLL("user32").DisplayConfigGetDeviceInfo
    get_name.argtypes = [ct.POINTER(AdapterName)]
    get_name.restype = ct.c_int32
    adapters = []
    with com_pointer() as factory:
        check_result(create(guid("770aae78-f26f-4dba-a829-253c83d1b387"), ct.byref(factory)), "CreateDXGIFactory1")
        index = 0
        while True:
            with com_pointer() as pointer:
                status = invoke(factory, 12, ct.c_int32, [ct.c_uint32, ct.POINTER(ct.c_void_p)], index, ct.byref(pointer))
                if status & 0xffffffff == 0x887A0002:  # DXGI_ERROR_NOT_FOUND
                    break
                check_result(status, "EnumAdapters1")
                desc = AdapterDesc()
                check_result(invoke(pointer, 10, ct.c_int32, [ct.POINTER(AdapterDesc)], ct.byref(desc)), "GetDesc1")
                if not desc.flags & 2:  # DXGI_ADAPTER_FLAG_SOFTWARE
                    name = AdapterName(type=4, size=ct.sizeof(AdapterName), luid=desc.luid)
                    status = get_name(ct.byref(name))
                    key = name.path.casefold() if status == 0 and name.path else f"luid:{desc.luid}"
                    adapters.append(Adapter(index, desc.name, str(desc.luid), key))
            index += 1
    return adapters


def default_luid():
    """复用 ORT 1.20.1 的 DXCore 默认排序规则，返回首个硬件 GPU 的 LUID。

    显式 device_id 走 DXGI，默认走 DXCore，两套索引不可互用。
    此处只枚举；调用方按 LUID 映射到 DXGI 后显式创建会话，保证日志与选卡一致。
    """
    create = ct.WinDLL("dxcore").DXCoreCreateAdapterFactory
    create.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    create.restype = ct.c_int32
    with com_pointer() as factory:
        check_result(create(guid("78ee5945-c36e-4b13-a669-005dd11c0f06"), ct.byref(factory)), "DXCoreCreateAdapterFactory")
        # ORT 优先 GENERIC_ML；列表为空时才改用 CORE_COMPUTE。
        for attribute in ("b71b0d41-1088-422f-a27c-0250b7d3a988", "248e2800-a793-4724-abaa-23a6de1be090"):
            with com_pointer() as listing:
                check_result(invoke(factory, 3, ct.c_int32,
                                    [ct.c_uint32, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_void_p)],
                                    1, guid(attribute), guid("526c7776-40e9-459b-b711-f32ad76dfc28"), ct.byref(listing)),
                             "DXCore CreateAdapterList")
                count = invoke(listing, 4, ct.c_uint32, [])
                if not count:
                    continue
                if count > 1:
                    preference = ct.c_uint32(2)  # DXCoreAdapterPreference::HighPerformance
                    check_result(invoke(listing, 7, ct.c_int32, [ct.c_uint32, ct.POINTER(ct.c_uint32)], 1, ct.byref(preference)),
                                 "DXCore Sort(HighPerformance)")
                for index in range(count):
                    with com_pointer() as adapter:
                        check_result(invoke(listing, 3, ct.c_int32,
                                            [ct.c_uint32, ct.c_void_p, ct.POINTER(ct.c_void_p)],
                                            index, guid("f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e"), ct.byref(adapter)),
                                     "DXCore GetAdapter")
                        hardware = ct.c_bool()
                        check_result(invoke(adapter, 6, ct.c_int32, [ct.c_uint32, ct.c_size_t, ct.c_void_p],
                                            11, ct.sizeof(hardware), ct.byref(hardware)), "DXCore IsHardware")
                        graphics = invoke(adapter, 4, ct.c_bool, [ct.c_void_p], guid("0c9ece4d-2f6e-4f01-8c96-e89e331b47b1"))
                        if hardware.value and graphics:
                            luid = Luid()
                            check_result(invoke(adapter, 6, ct.c_int32, [ct.c_uint32, ct.c_size_t, ct.c_void_p],
                                                0, ct.sizeof(luid), ct.byref(luid)), "DXCore InstanceLuid")
                            return str(luid)
                break
    raise OSError("DXCore 未找到可用的硬件 GPU，请在 OCR 设备设置中选择显卡或 CPU")


def select_device(choice="auto", gpu_id=None):
    """将用户选择解析为本次会话的适配器；None 表示 CPU。

    Args:
        choice: auto、cpu 或 list_adapters 提供的设备 key。
        gpu_id: 独立 OCR 调用可显式覆盖为本机 DXGI 索引；不用于持久化。

    Raises:
        ValueError: 选择格式错误，或保存的设备已不可用。
        OSError: 适配器枚举失败。
        RuntimeError: 用户指定 GPU 但当前运行时没有 DirectML。
    """
    if not isinstance(choice, str) or not choice:
        raise ValueError("OCR 设备必须为 auto、cpu 或设置界面提供的设备标识")
    if choice == "cpu":
        return None
    if gpu_id is not None and (type(gpu_id) is not int or gpu_id < 0):
        raise ValueError("gpu_id 必须是非负的 DXGI 适配器索引")
    if "DmlExecutionProvider" not in get_available_providers():
        if choice == "auto" and gpu_id is None:
            return None
        raise RuntimeError("当前 ONNX Runtime 不支持 DirectML，请选择 CPU 或安装项目声明的运行时")
    adapters = list_adapters()
    if gpu_id is not None:
        matches = [item for item in adapters if item.index == gpu_id]
    elif choice == "auto":
        luid = default_luid()
        matches = [item for item in adapters if item.luid == luid]
    else:
        matches = [item for item in adapters if item.key.casefold() == choice.casefold()]
    if not matches:
        raise ValueError("所选 OCR 设备已不可用，请打开进阶设置中的 OCR 设备设置重新选择")
    if len(matches) > 1:
        raise ValueError("OCR 设备标识不唯一，请重新选择设备")
    return matches[0]
