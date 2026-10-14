"""「秘闻」事件与事件选项选择的点击目标回归测试。

背景：select_secret / select_event 原来的 else 分支会盲点 `click(self.tx, self.ty)` 与
`click(0.1167, ty - 0.1139)`，而这两个坐标取自进入函数时的 `self.tx/self.ty`——它是上一次
check() 留下的位置（即使那次匹配失败也会写入），与当前界面元素无关，既可能误触别的控件，
也可能换算到游戏窗口之外、点到别的窗口上抢走游戏前台。

现在的行为（`select_secret` 的点击优先级）：
对话条右下角的金色 ▼（`arrow_dialog`）→「选择」栏里的白色 →（`arrow`）→ 命途选项卡
（命中则快速两击 + 「确认」图片兜底 0.5 s）→ 都不在时退回点「秘闻」文字。
每个分支都点自己刚匹配到的位置，不再使用残留坐标。
- select_event：事件选项、休息区都没匹配上时什么都不点。
"""

import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tool.utils.image_tool import load_all_images_from_directory

# 内核在导入期就会读图像记忆切片（例如 tool/utils/predict.py），需要先装载图像缓存
ROOT = Path(__file__).resolve().parents[1]
load_all_images_from_directory(str(ROOT / "resource" / "imgs"))

from core.any_fate.engine import AnyFateUniverse
from core.simulated.engine import SimulatedUniverse

# 进入函数时 self.tx/self.ty 的残留值（旧实现会照这个坐标盲点）
STALE_POINT = (0.5000, 0.5000)
# 命途图标（选项卡左上角）、「确认」按钮、对话箭头的匹配位置
FATE_POINT = (0.2865, 0.6250)
CONFIRM_POINT = (0.1119, 0.5398)
ARROW_POINT = (0.3044145833333333, 0.15833333333333333)
# 对话条右下角金色 ▼（arrow_dialog）的匹配位置
DIALOG_POINT = (0.178125, 0.3250)
# 选项卡 → 它下方「确定」按钮的手量固定偏移
CONFIRM_OFFSET = (0.1167, 0.1139)


class _FakeClock:
    """按固定步长前进的假时钟，保证 0.5 秒轮询窗口能确定性结束。"""

    def __init__(self, step=0.1):
        self.now = 1000.0
        self.step = step

    def time(self):
        self.now += self.step
        return self.now


class _StubEngine:
    """只提供内核方法所需接口的替身，check() 命中时顺带更新 tx/ty（与真实现一致）。"""

    def __init__(self, matched, text_results=None):
        self.tx, self.ty = STALE_POINT
        self.matched = matched
        self.event_prior = ["选择事件"]
        self.fate = "存护"
        self.get_screen = Mock()
        self.check = Mock(side_effect=self._check)
        self.click_text = Mock(side_effect=text_results or [False, False])

    def _check(self, path, x, y, **kwargs):
        if path in self.matched:
            self.tx, self.ty = self.matched[path]
            return True
        return False


class SelectSecretTest(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.enterContext(patch("core.any_fate.engine.key_mouse_manager", self.manager))
        self.enterContext(patch("core.any_fate.engine.CUS_LOGGER"))
        self.clock = _FakeClock()
        self.enterContext(
            patch("core.any_fate.engine.time", types.SimpleNamespace(time=self.clock.time))
        )

    def _run(self, matched):
        engine = _StubEngine(matched)
        AnyFateUniverse.select_secret(engine)
        return engine

    def _clicks(self):
        return [call.args for call in self.manager.click.call_args_list]

    def test_dialog_chevron_wins_over_bar_arrow_and_fate_icon(self):
        """点击优先级第一位是金色 ▼：它和「选择」栏箭头、命途选项卡同时在场时只点它。"""
        engine = self._run(
            {"arrow_dialog": DIALOG_POINT, "arrow": ARROW_POINT, "fate3": FATE_POINT}
        )

        self.assertEqual(self._clicks(), [DIALOG_POINT])
        self.assertEqual([call.args[0] for call in engine.check.call_args_list], ["arrow_dialog"])

    def test_bar_arrow_clicked_when_chevron_absent(self):
        """金色 ▼ 不在场时才点「选择」栏里的白色 →，并且不再往下扫命途选项卡。"""
        engine = self._run({"arrow": ARROW_POINT, "fate3": FATE_POINT})

        self.assertEqual(self._clicks(), [ARROW_POINT])
        self.assertEqual(
            [call.args[0] for call in engine.check.call_args_list], ["arrow_dialog", "arrow"]
        )

    def test_secret_text_is_clicked_as_last_resort(self):
        """两个对话推进图标都没匹配上时，退回点「秘闻」文字（点画面任意位置都会把剧情推下去）。"""
        engine = self._run({})

        self.manager.click.assert_not_called()
        engine.click_text.assert_called_once_with(text="秘闻", box=[197, 233, 887, 906])

    def test_fate_icon_clicked_then_fixed_offset_then_confirm(self):
        """命中命途选项卡：先点选项卡本体、再点固定偏移的「确定」，最后用「确认」兜底。"""
        self._run({"fate3": FATE_POINT, "confirm": CONFIRM_POINT})

        self.assertEqual(
            self._clicks(),
            [
                FATE_POINT,
                (CONFIRM_OFFSET[0], FATE_POINT[1] - CONFIRM_OFFSET[1]),
                CONFIRM_POINT,
            ],
        )

    def test_fate_icon_clicked_even_if_confirm_never_appears(self):
        """「确认」按钮一直不出现时只留下那两下快速点击，且轮询会结束。"""
        engine = self._run({"fate1": FATE_POINT})

        self.assertEqual(
            self._clicks(),
            [FATE_POINT, (CONFIRM_OFFSET[0], FATE_POINT[1] - CONFIRM_OFFSET[1])],
        )
        self.assertGreaterEqual(
            [call.args[0] for call in engine.check.call_args_list].count("confirm"), 1
        )


class SelectEventTest(unittest.TestCase):
    """core/simulated/engine.py 的 select_event（any_fate 也会 super() 调到它）。"""

    def setUp(self):
        self.manager = Mock()
        self.enterContext(patch("core.simulated.engine.key_mouse_manager", self.manager))
        self.enterContext(patch("core.simulated.engine.CUS_LOGGER"))
        self.clock = _FakeClock()
        self.enterContext(
            patch("core.simulated.engine.time", types.SimpleNamespace(time=self.clock.time))
        )

    def _run(self, matched, text_results):
        engine = _StubEngine(matched, text_results)
        SimulatedUniverse.select_event(engine)
        return engine

    def test_no_option_matched_clicks_nothing(self):
        """事件选项、休息区都没匹配上时一下都不点。"""
        self._run({}, [False, False])

        self.manager.click.assert_not_called()

    def test_option_matched_but_confirm_missing_clicks_nothing(self):
        """事件选项匹配上了但「确认」没出现：同样不再补盲点。"""
        self._run({}, [True, False])

        self.manager.click.assert_not_called()

    def test_confirm_is_clicked_when_it_appears(self):
        """「确认」出现时仍然点它自己的匹配位置。"""
        self._run({"confirm": CONFIRM_POINT}, [True, False])

        self.assertEqual([call.args for call in self.manager.click.call_args_list], [CONFIRM_POINT])


if __name__ == "__main__":
    unittest.main()
