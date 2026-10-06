"""通过完整主窗口验证运行文字下的布局，隔离热键、协议弹窗和启动清理。"""

import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication, QDialog, QLayout, QMessageBox, QWidget

from test.core_fixture import copy_core
from tool import GLOBAL
from tool.registry import KernelRegistry
from tool.settings import load_settings, update_settings


class RuntimeGuiLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # QApplication 必须先创建，主窗口导入会初始化 Qt 日志信号。
        import new_gui

        cls.module = new_gui

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        config = Path(self.folder.name)
        source = Path(self.module.PATHS["example"]) / "settings_example.json"
        (config / "settings.json").write_bytes(source.read_bytes())
        shutil.copytree(Path(self.module.PATHS["root"]) / "actions", config / "actions")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(self.module.PATHS, {"root": str(config), "config": str(config), "core": str(copy_core(config / "core")), "backup": str(config / "backup")}))
        self.stack.enter_context(patch.object(self.module.MainWindow, "setup_keyboard_listener"))
        self.stack.enter_context(patch.object(self.module.MainWindow, "check_first_launch"))
        self.stack.enter_context(patch.object(self.module.QTimer, "singleShot"))
        # 主窗口会重绑全局日志信号；销毁后恢复原引用，避免后续测试向已释放的 Qt 窗口发信号。
        for name in ("PRINT_TO_UI", "IMAGE_TO_UI", "DIALOG"):
            self.stack.enter_context(patch.object(GLOBAL, name, None, create=True))
        self.window = self.module.MainWindow()
        self.addCleanup(self.dispose_window)

    def dispose_window(self):
        # closeEvent 会退出整个进程，因此释放窗口时不调用 close。
        self.window.hide()
        self.window.tray_icon.hide()
        GLOBAL.PRINT_TO_UI = GLOBAL.IMAGE_TO_UI = GLOBAL.DIALOG = None
        self.window.deleteLater()
        self.app.sendPostedEvents(None, QEvent.DeferredDelete)

    def show_window(self, width=930):
        self.window.resize(width, 760)
        self.window.show()
        self.app.processEvents()

    def set_mode(self, debug):
        update_settings({"debug": debug})
        self.window.opt = load_settings()
        self.window.update_dependent_controls_state()

    def test_filled_window_keeps_log_width_when_debug_controls_open(self):
        self.window.set_FPS(144.5)
        self.window.set_find_path_state("正在匹配交易地图，筛选目标节点并规划下一步路径")
        self.window.set_kill_num("40 / 40")
        self.window.Label_RunningState.setText("任务序列线程状态: 运行中")
        for count in range(20):
            self.window.print_to_ui(
                f"第 {count + 1} 条运行日志：已完成地图识别，正在检查交易、休整及奖励节点，继续计算下一步路径。",
                color="555555", time=True,
            )
        for width in (930, 1500):
            with self.subTest(width=width):
                self.set_mode(False)
                self.show_window(width)
                normal_width = self.window.TextBrowser.width()
                self.set_mode(True)
                self.show_window(width)
                self.assertGreaterEqual(self.window.TextBrowser.width(), normal_width)
                self.assertGreater(self.window.TextBrowser.width(), self.window.Tab1.width() * 0.65)
                self.assertGreater(self.window.engine_combo.width(), 0)
                self.assertGreater(self.window.script_combo.width(), 0)

    def test_long_status_text_and_script_name_do_not_enlarge_controls(self):
        self.set_mode(True)
        self.show_window()
        right = self.window.findChild(QLayout, "Lay_Tab1_Right")
        original = (self.window.TextBrowser.width(), right.geometry().width())
        self.window.set_find_path_state("正在搜索可用路径与目标节点" * 30)
        self.window.set_kill_num("当前击杀数及累计统计" * 30)
        self.window.script_combo.addItem("long_script_name_" * 30, "long.json")
        self.window.script_combo.setCurrentIndex(self.window.script_combo.count() - 1)
        self.show_window()
        self.assertEqual((self.window.TextBrowser.width(), right.geometry().width()), original)

    def test_engine_and_script_controls_use_separate_rows(self):
        self.set_mode(True)
        self.show_window()
        self.assertLessEqual(abs(self.window.engine_combo.geometry().center().y()
                                 - self.window.engine_settings_btn.geometry().center().y()), 1)
        self.assertLessEqual(abs(self.window.script_combo.geometry().center().y()
                                 - self.window.run_script_btn.geometry().center().y()), 1)
        self.assertGreater(self.window.script_combo.y(), self.window.engine_combo.geometry().bottom())
        self.assertGreater(self.window.engine_combo.width(), 150)
        self.assertGreater(self.window.script_combo.width(), 100)

    def test_main_window_keeps_only_general_configuration_controls(self):
        self.assertFalse(hasattr(self.window, "homeTab"))
        self.assertFalse(hasattr(self.window, "CurrencyWarTab"))
        for widget in self.window.AbyssTab.findChildren(QWidget):
            self.assertFalse(widget.objectName().startswith(("Simul_", "Diver_", "Iron_blood_", "Any_fate_", "Finger_snap_", "Currency_")))
        self.assertIsNone(self.window.findChild(QWidget, "debug_checkbox2"))

    def test_general_settings_save_preserves_configured_debug_mode(self):
        for debug in (False, True):
            self.set_mode(debug)
            with patch.object(QMessageBox, "information"):
                self.window.general_settings_save_btn.click()
            self.assertEqual(load_settings()["debug"], debug)
            self.assertIsNone(self.window.findChild(QWidget, "debug_checkbox2"))

    def test_empty_module_directory_disables_run_and_diagnostics(self):
        self.dispose_window()
        with patch.object(self.module, "KernelRegistry", return_value=KernelRegistry(Path(self.folder.name) / "missing")):
            self.window = self.module.MainWindow()
        self.assertEqual(self.window.engine_combo.count(), 0)
        self.assertFalse(self.window.kernel_rows)
        for widget in (self.window.run_script_btn, self.window.engine_settings_btn,
                       self.window.test_btn, self.window.print_btn, self.window.calibrate_btn):
            self.assertFalse(widget.isEnabled())
        self.assertFalse(self.window.restore_action.isEnabled())

    def test_gears_open_matching_configuration_in_both_modes_without_starting_tasks(self):
        dialog = Mock()
        dialog.exec_.return_value = QDialog.Rejected
        with patch.object(self.window.registry, "create_settings", return_value=dialog) as factory, \
                patch.object(self.window, "start_task") as start:
            self.set_mode(False)
            self.show_window()
            for engine, (run, button) in self.window.kernel_rows.items():
                self.assertTrue(button.isVisible())
                self.assertFalse(button.icon().isNull())
                button.click()
                factory.assert_called_with(engine, self.window)
            self.set_mode(True)
            self.assertTrue(self.window.engine_settings_btn.isVisible())
            for spec in self.window.registry.runnable():
                self.window.engine_combo.setCurrentIndex(self.window.engine_combo.findData(spec.id))
                self.window.engine_settings_btn.click()
                factory.assert_called_with(spec.id, self.window)
            start.assert_not_called()

    def test_common_selection_uses_dynamic_factory_and_survives_reopening(self):
        self.set_mode(True)
        index = self.window.engine_combo.findData("Common")
        self.assertGreaterEqual(index, 0)
        self.window.engine_combo.setCurrentIndex(index)
        self.assertEqual(load_settings()["script_engine"], "Common")
        with patch.object(self.window, "start_task", side_effect=lambda task: task()), \
                patch.object(self.window.registry, "create_engine") as factory, \
                patch.object(self.module, "run_action_script") as runner:
            self.window.run_script_btn.click()
            factory.assert_called_once_with("Common", script=True)
            runner.assert_called_once_with(factory.return_value, self.window.script_combo.currentData())
        self.dispose_window()
        self.window = self.module.MainWindow()
        self.assertEqual(self.window.engine_combo.currentData(), "Common")


if __name__ == "__main__":
    unittest.main()
