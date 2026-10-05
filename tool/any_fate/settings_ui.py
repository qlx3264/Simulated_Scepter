"""寰宇命途配置界面。"""

from pathlib import Path

from route import PATHS
from tool.gui.runtime_settings import RuntimeSettings
from tool.gui.settings_dialog import SettingsDialog as BaseDialog
from tool.settings import load_settings, update_settings
from tool.simul.config import config


class SettingsDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__("寰宇命途配置", parent)
        data = load_settings()
        self.runtime = RuntimeSettings(data, self)
        self.content_layout.addWidget(self.runtime)
        self.section = ui = self.add_section(Path(PATHS["ui"]) / "AnyFateSettings.ui")
        ui.Any_fate_combo.addItems(config.fates)
        ui.Any_fate_combo.setCurrentText(data.get("any_fate", "巡猎"))

    def persist(self):
        update_settings(self.runtime.collect() | {"any_fate": self.section.Any_fate_combo.currentText()})
