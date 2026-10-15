"""隔离日志与桌面副作用，验证公开 OCR 入口的设备选择及会话配置。"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tool.onnxocr import device

# 只替换日志入口，不回滚整个 sys.modules，以免卸载 cv2 等原生扩展后重复初始化。
original_log = sys.modules.get("tool.log")
sys.modules["tool.log"] = SimpleNamespace(CUS_LOGGER=Mock())
try:
    from tool.onnxocr import onnx_paddleocr, predict_base
finally:
    if original_log is None:
        sys.modules.pop("tool.log")
    else:
        sys.modules["tool.log"] = original_log


class OcrDeviceTests(unittest.TestCase):
    def setUp(self):
        self.fast = device.Adapter(0, "独立显卡", "0x00000000_0x00000011", "pci:fast")
        self.slow = device.Adapter(3, "集成显卡", "0x00000000_0x00000022", "pci:slow")
        self.enterContext(patch.object(device, "list_adapters", return_value=[self.fast, self.slow]))
        self.default = self.enterContext(patch.object(device, "default_luid", return_value=self.slow.luid))
        self.providers = self.enterContext(patch.object(device, "get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]))
        self.settings = self.enterContext(patch.object(onnx_paddleocr, "load_settings", return_value={"ocr_device": "pci:fast"}))
        self.sessions = []
        self.factory = self.enterContext(patch.object(predict_base, "InferenceSession", side_effect=self.make_session))
        self.logger = self.enterContext(patch.object(predict_base, "CUS_LOGGER"))

    def make_session(self, model, *, providers, sess_options=None):
        session = Mock()
        session.get_providers.return_value = [item[0] if isinstance(item, tuple) else item for item in providers]
        session.get_inputs.return_value = [SimpleNamespace(name="input")]
        session.get_outputs.return_value = [SimpleNamespace(name="output")]
        self.sessions.append(session)
        return session

    def assert_gpu(self, ocr, index):
        self.assertEqual(ocr.text_detector.device.index, index)
        self.assertIs(ocr.text_detector.device, ocr.text_recognizer.device)
        for call, session in zip(self.factory.call_args_list, self.sessions):
            self.assertEqual(call.kwargs["providers"][0], ("DmlExecutionProvider", {"device_id": str(index)}))
            self.assertIn("CPUExecutionProvider", call.kwargs["providers"])
            self.assertFalse(call.kwargs["sess_options"].enable_mem_pattern)
            session.disable_fallback.assert_called_once()

    def test_global_selection_applies_to_detector_and_recognizer(self):
        ocr = onnx_paddleocr.ONNXPaddleOcr()
        self.assert_gpu(ocr, 0)
        self.assertEqual(self.factory.call_count, 2)
        for call in self.logger.debug.call_args_list:
            self.assertIn(self.fast.name, call.args[-1])
            self.assertIn(self.fast.luid, call.args[-1])

    def test_old_settings_use_dxcore_order_instead_of_dxgi_zero(self):
        self.settings.return_value = {}
        self.assert_gpu(onnx_paddleocr.ONNXPaddleOcr(), 3)
        self.default.assert_called_once()

    def test_device_path_survives_dxgi_index_change(self):
        moved = device.Adapter(5, self.fast.name, "new-luid", self.fast.key)
        with patch.object(device, "list_adapters", return_value=[self.slow, moved]):
            self.assert_gpu(onnx_paddleocr.ONNXPaddleOcr(), 5)

    def test_explicit_gpu_id_overrides_global_selection(self):
        self.assert_gpu(onnx_paddleocr.ONNXPaddleOcr(gpu_id=3), 3)
        self.settings.assert_not_called()

    def test_per_call_device_overrides_global_selection(self):
        self.assert_gpu(onnx_paddleocr.ONNXPaddleOcr(ocr_device="pci:slow"), 3)
        self.settings.assert_not_called()

    def test_cpu_flags_take_priority_and_do_not_enumerate(self):
        for kwargs in ({"cpu": True}, {"use_gpu": False}, {"cpu": True, "gpu_id": -1}):
            with self.subTest(kwargs=kwargs):
                self.factory.reset_mock()
                self.sessions.clear()
                with patch.object(device, "list_adapters", side_effect=AssertionError("CPU 不应访问 GPU")):
                    ocr = onnx_paddleocr.ONNXPaddleOcr(**kwargs)
                self.assertIsNone(ocr.text_detector.device)
                self.assertIsNone(ocr.text_recognizer.device)
                self.assertEqual(self.factory.call_count, 2)
                for call in self.factory.call_args_list:
                    self.assertEqual(call.kwargs["providers"], ["CPUExecutionProvider"])
        self.settings.assert_not_called()

    def test_global_cpu_selection_does_not_enumerate(self):
        self.settings.return_value = {"ocr_device": "cpu"}
        with patch.object(device, "list_adapters", side_effect=AssertionError("CPU 不应访问 GPU")):
            ocr = onnx_paddleocr.ONNXPaddleOcr()
        self.assertIsNone(ocr.text_detector.device)

    def test_missing_device_stops_before_creating_models(self):
        self.settings.return_value = {"ocr_device": "pci:removed"}
        with self.assertRaisesRegex(ValueError, "重新选择"):
            onnx_paddleocr.ONNXPaddleOcr()
        self.factory.assert_not_called()

    def test_invalid_gpu_id_is_rejected_without_cpu_fallback(self):
        for index in (-1, True, "0", 1.5, 99):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    onnx_paddleocr.ONNXPaddleOcr(gpu_id=index)
        self.factory.assert_not_called()

    def test_invalid_saved_value_is_rejected(self):
        for choice in (None, 1, "", {}):
            with self.subTest(choice=choice):
                self.settings.return_value = {"ocr_device": choice}
                with self.assertRaises(ValueError):
                    onnx_paddleocr.ONNXPaddleOcr()
        self.factory.assert_not_called()

    def test_cpu_only_runtime_preserves_auto_but_rejects_explicit_gpu(self):
        self.providers.return_value = ["CPUExecutionProvider"]
        self.settings.return_value = {}
        ocr = onnx_paddleocr.ONNXPaddleOcr()
        self.assertIsNone(ocr.text_detector.device)
        self.factory.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "不支持 DirectML"):
            onnx_paddleocr.ONNXPaddleOcr(gpu_id=0)
        self.factory.assert_not_called()

    def test_dml_initialization_fallback_is_reported_as_failure(self):
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        self.factory.side_effect = None
        self.factory.return_value = session
        with self.assertRaisesRegex(RuntimeError, "已回退 CPU"):
            onnx_paddleocr.ONNXPaddleOcr()
        self.logger.debug.assert_not_called()

    def test_native_enumeration_error_is_not_hidden(self):
        with patch.object(device, "list_adapters", side_effect=OSError("DXGI failed")):
            with self.assertRaisesRegex(OSError, "DXGI failed"):
                onnx_paddleocr.ONNXPaddleOcr()
        self.factory.assert_not_called()

    def test_luid_selection_and_duplicate_identity(self):
        fallback = device.Adapter(2, "另一块显卡", "LUID", "luid:LUID")
        with patch.object(device, "list_adapters", return_value=[fallback]):
            self.assertEqual(device.select_device("luid:luid"), fallback)
        with patch.object(device, "list_adapters", return_value=[self.fast, self.fast]):
            with self.assertRaisesRegex(ValueError, "不唯一"):
                device.select_device(self.fast.key)

    def test_com_reference_is_released_when_enumeration_raises(self):
        with patch.object(device, "invoke") as release:
            with self.assertRaisesRegex(OSError, "test"):
                with device.com_pointer() as pointer:
                    pointer.value = 123
                    raise OSError("test")
        release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
