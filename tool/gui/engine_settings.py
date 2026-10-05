"""点击配置入口时才导入对应内核的配置界面。"""

from importlib import import_module

SETTINGS_MODULES = {
    "Simulated": "tool.simul.settings_ui",
    "Divergent": "tool.diver.settings_ui",
    "AnyFate": "tool.any_fate.settings_ui",
    "IronBlood": "tool.iron_blood.settings_ui",
    "Currency": "tool.currency.settings_ui",
    "FingerSnap": "tool.finger_snap.settings_ui",
}


def create_settings_dialog(engine, parent=None):
    return import_module(SETTINGS_MODULES[engine]).SettingsDialog(parent)
