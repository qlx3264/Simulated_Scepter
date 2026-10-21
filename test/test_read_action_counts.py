"""``AnyFateUniverse.read_action_counts`` 的回归测试。

背景：作弊功能要通关寰宇蝗灾才解锁。没解锁的存档进骰面界面时，界面上只剩
「重投次数」一项、整行居中显示，正常那两个点位（作弊在左下、重投在右下）
都认不到数字，旧逻辑于是每轮都认不到次数、也读不到 0，永远卡在骰面。
现在两边都没认到时再在居中那个位置认一次：认到了就提示玩家没解锁作弊，
按 0 次作弊 + 识别到的重投次数继续往下走；认不到就回到原来的路径。

另外认次数时先只认、不落样本（``save_unmatched=False``）：居中点位认到了就说明
那两个正常点位本来就是空白，不该往模板目录里塞两张空图；只有三个点位都认不到，
才回去按老规矩各存一份，方便以后补模板。
"""
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from importing import load_img

# core.any_fate.engine 在导入期会读取图像内存切片，必须先预热图片缓存。
load_img()

from core.any_fate.engine import AnyFateUniverse  # noqa: E402
from tool.utils.ocr_num import (  # noqa: E402
    match_cheat_count_in_region,
    match_roll_count_in_region,
)


class _StubAnyFate(AnyFateUniverse):
    """只保留 read_action_counts 的真实逻辑，OCR 和日志全部打桩。"""

    def __init__(self):
        self.screen = "screen"


class ReadActionCountsTest(unittest.TestCase):
    def setUp(self):
        self.roll = self.enterContext(
            patch("core.any_fate.engine.match_roll_count_in_region", Mock(return_value=None)))
        self.cheat = self.enterContext(
            patch("core.any_fate.engine.match_cheat_count_in_region", Mock(return_value=None)))
        self.locked = self.enterContext(
            patch("core.any_fate.engine.match_locked_roll_count_in_region", Mock(return_value=None)))
        self.logger = self.enterContext(patch("core.any_fate.engine.CUS_LOGGER", Mock()))
        self.engine = _StubAnyFate()

    def _messages(self, level):
        return "".join(str(call.args[0]) for call in getattr(self.logger, level).call_args_list)

    def test_reads_both_counts_on_the_normal_dice_screen(self):
        """正常界面：两个点位都认到了，就不去居中位置多认一次，也不动模板目录。"""
        self.roll.return_value = 2
        self.cheat.return_value = 1

        self.assertEqual(self.engine.read_action_counts(), (2, 1))

        self.roll.assert_called_once_with("screen", save_unmatched=False)
        self.cheat.assert_called_once_with("screen", save_unmatched=False)
        self.locked.assert_not_called()
        self.assertIn("当前重投次数", self._messages("debug"))
        self.assertIn("当前作弊次数", self._messages("debug"))
        self.assertNotIn("尚未解锁作弊功能", self._messages("error"))

    def test_locked_cheat_is_read_from_the_centered_position(self):
        """两边都没认到、居中位置认到数字：提示没解锁作弊，作弊按 0 次算。

        居中点位认到了就说明那两个点位本来就是空白，不该再回去落两张空样本。
        """
        self.locked.return_value = 3

        self.assertEqual(self.engine.read_action_counts(), (3, 0))

        self.roll.assert_called_once_with("screen", save_unmatched=False)
        self.cheat.assert_called_once_with("screen", save_unmatched=False)
        self.locked.assert_called_once_with("screen", save_unmatched=False)
        self.assertIn("尚未解锁作弊功能", self._messages("error"))
        self.assertIn("当前重投次数: 3", self._messages("debug"))

    def test_unrecognized_counts_stay_none_and_only_then_save_samples(self):
        """三个点位都认不到：返回 (None, None)，这时才回到老逻辑各存一份样本。"""
        self.assertEqual(self.engine.read_action_counts(), (None, None))

        self.locked.assert_called_once_with("screen", save_unmatched=False)
        self.assertEqual(self.roll.call_args_list[-1], call("screen"))
        self.assertEqual(self.cheat.call_args_list[-1], call("screen"))
        self.assertEqual(self.roll.call_count, 2)
        self.assertEqual(self.cheat.call_count, 2)
        self.assertNotIn("尚未解锁作弊功能", self._messages("error"))
        self.assertEqual(self._messages("debug"), "")

    def test_one_recognized_count_skips_the_locked_probe(self):
        """只认到其中一个也不去认居中位置：那只说明另一个没认出来，不是没解锁。"""
        self.cheat.return_value = 1

        self.assertEqual(self.engine.read_action_counts(), (None, 1))

        self.roll.assert_called_once_with("screen", save_unmatched=False)
        self.locked.assert_not_called()
        self.assertIn("当前作弊次数", self._messages("debug"))
        self.assertNotIn("当前重投次数", self._messages("debug"))


class SaveUnmatchedTest(unittest.TestCase):
    """认不到数字时到底落不落样本，交给调用方决定。"""

    def setUp(self):
        # 全黑的空白区域：任何点位都认不到数字
        self.image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.save = self.enterContext(
            patch("tool.utils.ocr_num._save_unmatched_roll_count", Mock(return_value="/tmp/x.png")))
        self.logger = self.enterContext(patch("tool.utils.ocr_num.CUS_LOGGER", Mock()))

    def test_probe_does_not_save_an_unmatched_sample(self):
        """先探一遍的时候不落样本、也不打告警。"""
        self.assertIsNone(match_roll_count_in_region(self.image, save_unmatched=False))

        self.save.assert_not_called()
        self.logger.warning.assert_not_called()

    def test_default_still_saves_an_unmatched_sample(self):
        """没人说要先探的时候还是照旧落样本，补模板靠它。"""
        self.assertIsNone(match_cheat_count_in_region(self.image))

        self.save.assert_called_once()
        self.assertIn("未识别作弊次数", str(self.logger.warning.call_args))


if __name__ == "__main__":
    unittest.main()
