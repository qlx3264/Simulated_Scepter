"""OCR 设备选择与保存，复用全局设置和配置对话框。"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QLabel

from tool.gui.settings_dialog import SettingsDialog
from tool.onnxocr.device import list_adapters, select_device
from tool.settings import load_settings, update_settings


class OcrSettingsDialog(SettingsDialog):
    def __init__(self, parent=None):
        super().__init__("OCR 设备设置", parent)
        self.resize(680, 320)
        self.content_layout.addWidget(QLabel("选择文字识别使用的设备：", self))
        self.device_combo = QComboBox(self)
        self.device_combo.setMinimumContentsLength(20)
        self.device_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.device_combo.addItem("自动选择（沿用运行时规则）", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.content_layout.addWidget(self.device_combo)
        self.device_detail = QLabel(self)
        self.device_detail.setWordWrap(True)
        self.device_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.content_layout.addWidget(self.device_detail)
        hint = QLabel("检测与文字识别共用设备。保存后重启程序生效。", self)
        hint.setWordWrap(True)
        self.content_layout.addWidget(hint)

        try:
            for adapter in list_adapters():
                self.device_combo.addItem(f"{adapter.name}（DXGI {adapter.index}）", adapter.key)
        except OSError as error:
            self.device_detail.setText(f"无法枚举显卡：{error}。仍可选择 CPU。")

        choice = load_settings().get("ocr_device", "auto")
        index = self.device_combo.findData(choice)
        if index < 0:
            self.device_combo.addItem("已保存的设备（当前不可用）", choice)
            index = self.device_combo.count() - 1
        self.device_combo.setCurrentIndex(index)
        self.device_combo.currentIndexChanged.connect(self.show_device)
        self.show_device()

    def show_device(self):
        """显示所选设备的当前身份；无法解析时保留选择供用户修正。"""
        try:
            adapter = select_device(self.device_combo.currentData())
        except (OSError, ValueError, RuntimeError) as error:
            self.device_detail.setText(str(error))
            return
        if adapter is None:
            self.device_detail.setText("重启后使用：CPU")
        else:
            self.device_detail.setText(f"重启后使用：{adapter.label}")

    def persist(self):
        choice = self.device_combo.currentData()
        try:
            select_device(choice)
        except (OSError, RuntimeError) as error:
            raise ValueError(str(error)) from error
        update_settings({"ocr_device": choice})
