"""验证键鼠操作不会把光标送到游戏窗口之外。

2026-10-11 12:18 前台丢失事故中，`core/any_fate/engine.py:1096` 传入的归一化 y
在减去 0.1139 后变成负数，换算成屏幕坐标后落到游戏窗口下边界以下，
点击因此送给了压在下面的自动化工具窗口，游戏前台被抢走、任务卡死。
这里用真实队列固定住这次事故的坐标，确认越界操作被跳过。
"""

import unittest
from unittest.mock import patch

from tool.key_mouse_manager import KeyMouseManager

# 游戏窗口右下角与客户区尺寸，与 test_common_runtime.py:52 的实机参数一致
WINDOW_RIGHT = 1940
WINDOW_BOTTOM = 1110
CLIENT_WIDTH = 1920
CLIENT_HEIGHT = 1080
# 事故当次点击的请求坐标，换算后为 (1716, 1125)，在下边界 1110 之下 15 像素
INCIDENT_X = 0.1167
INCIDENT_Y = -0.013900000000000023


class KeyMouseBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.manager = KeyMouseManager()
        self.manager.set_screen_params(WINDOW_RIGHT, WINDOW_BOTTOM, CLIENT_WIDTH, CLIENT_HEIGHT)
        self.addCleanup(self.manager.stop)
        self.logger = self.enterContext(patch("tool.key_mouse_manager.CUS_LOGGER"))
        self.cursor = self.enterContext(patch("tool.key_mouse_manager.win32api.SetCursorPos"))
        self.click = self.enterContext(patch("tool.key_mouse_manager.pyautogui.click"))
        self.drag_to = self.enterContext(patch("tool.key_mouse_manager.pyautogui.dragTo"))
        self.scroll = self.enterContext(patch("tool.key_mouse_manager.pyautogui.scroll"))
        self.manager.start()

    def click_at(self, x, y):
        self.manager.click(x, y)
        self.manager.wait()

    def test_incident_click_below_window_is_skipped(self):
        """事故坐标越界时既不移动光标也不点击，并记录一条说明原因的警告。"""
        self.click_at(INCIDENT_X, INCIDENT_Y)
        self.cursor.assert_not_called()
        self.click.assert_not_called()
        self.logger.warning.assert_called_once()
        # 警告以参数化形式给出请求坐标与换算结果，最后两项即落到下边界外的 (1716, 1125)
        self.assertEqual(self.logger.warning.call_args.args[1], "点击")
        self.assertEqual(self.logger.warning.call_args.args[-2:], (1716, 1125))

    def test_repeated_out_of_range_click_warns_once(self):
        """同一偏离重复发生时不反复刷屏。"""
        for _ in range(3):
            self.click_at(INCIDENT_X, INCIDENT_Y)
        self.cursor.assert_not_called()
        self.logger.warning.assert_called_once()

    def test_click_outside_other_edges_is_skipped(self):
        """窗口上边界、左右边界之外同样跳过。"""
        for x, y in ((-0.05, 0.5), (1.05, 0.5), (1940, 540)):
            with self.subTest(x=x, y=y):
                self.click_at(x, y)
        self.cursor.assert_not_called()
        self.click.assert_not_called()

    def test_click_on_window_boundary_is_allowed(self):
        """y=0 对应窗口下边界，仍在窗口内。"""
        self.click_at(INCIDENT_X, 0.0)
        self.cursor.assert_called_once_with((1716, 1110))

    def test_ratio_click_inside_window_still_clicks(self):
        self.click_at(INCIDENT_X, 0.5)
        self.cursor.assert_called_once_with((1716, 570))
        self.click.assert_called_once()

    def test_actual_click_inside_window_still_clicks(self):
        self.click_at(215, 896)
        self.cursor.assert_called_once_with((235, 926))
        self.click.assert_called_once()

    def test_scroll_inside_window_still_scrolls(self):
        self.manager.scroll(direct=-1, x=0.5, y=0.5)
        self.manager.wait()
        self.cursor.assert_called_once_with((980, 570))
        self.scroll.assert_called_once_with(-120)

    def test_scroll_outside_window_is_skipped(self):
        self.manager.scroll(direct=-1, x=INCIDENT_X, y=INCIDENT_Y)
        self.manager.wait()
        self.cursor.assert_not_called()
        self.scroll.assert_not_called()

    def test_drag_inside_window_still_drags(self):
        self.manager.drag(0.5, 0.5, 0.6, 0.6)
        self.manager.wait()
        self.cursor.assert_called_once_with((980, 570))
        self.drag_to.assert_called_once()

    def test_drag_outside_window_is_skipped(self):
        """终点越界时起点也不移动，避免拖拽中途丢开左键。"""
        self.manager.drag(0.5, 0.5, INCIDENT_X, INCIDENT_Y)
        self.manager.wait()
        self.cursor.assert_not_called()
        self.drag_to.assert_not_called()


if __name__ == "__main__":
    unittest.main()
