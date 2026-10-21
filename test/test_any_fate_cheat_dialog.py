"""作弊选项窗（铁血战士、弹指一挥共用）的点选回归测试。

作弊选项窗是点掷骰界面的「作弊」按钮后弹出的模态窗：标题「作弊」+ 可滚动的骰面效果列表
（每行 = 「【命途·效果名】」标题 + 下面一行说明文字）+ 底部「取消」「确认」。
本文件锁定下面这些行为（都是实机踩过的坑）：

1. 只认以「【」开头的行标题。说明文字里会出现别的效果名（比如浇灌那行写着「立即为当前位面
   所有带有【慈怀】效果的区域的…」），按说明文字匹配会选错行（issue #297：想选「慈怀」，
   实际确认了「可憎」）。
2. 列表用滚轮翻，不用拖拽：拖拽松手有惯性，列表在截图之后还在滑，按截图坐标点击就会点偏。
3. 点完必须看到那一行的高亮（整条卡片变紫）才算选中，没高亮就按当前位置重点，最多五次；
   点几次都没高亮就放弃这一屏，宁可这一轮不作弊，也不要确认一个没选中的骰面。
4. 「确认」按当前帧 OCR 到的「确认」文字位置点击，并且取框内最下面那一块——两种模式的作弊
   窗高度不同，写死 y 会点空；标题里也可能出现「确认」两个字。
5. 窗口还在不在，每一步都现查一次（OCR 标题那一小块），不能缓存。
6. 选不中时按 esc 关窗（不消耗作弊次数），并清掉 self.cheat_target。

测试全部用 stub：ts.find_with_box 按区域返回 OCR 结果，键鼠操作记录下来供断言，
不连游戏、不真点击、不真等待。
"""

import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from tool.utils.image_tool import load_all_images_from_directory

ROOT = Path(__file__).resolve().parents[1]
# 内核在 import 时会读图像内存切片，所以要先加载图片再导入内核
load_all_images_from_directory(str(ROOT / "resource" / "imgs"))

from core.any_fate.engine import AnyFateUniverse  # noqa: E402


def mirror(point, xx=1920, yy=1080):
    """真实像素坐标 → 点击用的镜像比例坐标（和 click_event_option 的换算一致）。"""
    return 1 - point[0] / xx, 1 - point[1] / yy


def center(box):
    return (box[0] + box[1]) / 2, (box[2] + box[3]) / 2


def line(text, box):
    return {"raw_text": text, "box": list(box)}


# 底部按钮
CONFIRM = line("确认", [1168, 1223, 811, 841])
# 标题里也带「确认」两个字的情形：按钮在它下面，点的时候必须挑最下面那个
CONFIRM_TITLE = line("确认丢弃祝福「无根据颂歌」？", [900, 1200, 720, 750])
CHEAT_BUTTON = line("作弊", [1261, 1321, 761, 792])

# 丰饶（弹指一挥）作弊列表里的行
CIHUAI_TITLE = line("【丰饶·慈怀】", [470, 700, 828, 854])
CIHUAI_DESC = line("选择1个区域，为其附加【慈怀】效果", [560, 1440, 860, 887])
GUIXIN_TITLE = line("【丰饶·归心】", [470, 700, 690, 716])
# 说明文字里带「慈怀」、但标题是别的效果：不能被匹配到
JIAOGUAN_DESC = line(
    "立即为当前位面所有带有【慈怀】效果的区域的1个相邻区域附加【慈怀】效果", [560, 1440, 700, 727])


class _StubCheatEngine(AnyFateUniverse):
    """只保留作弊窗的真实逻辑，OCR 和键鼠全部换成可断言的假实现。"""

    def __init__(self, list_frames, dialog_open=False, cheat_button=True,
                 confirm_visible=True, confirm_closes=True, select_works=True,
                 select_works_after=0, cheat_target="慈怀", highlight_fraction=1.0,
                 confirm_lines=None):
        # 不走真实 __init__：这里不需要游戏连接
        self.xx, self.yy = 1920, 1080
        self.screen = None
        self._stop = False
        self.cheat_target = cheat_target
        self.confirm_lines = list(confirm_lines) if confirm_lines else [CONFIRM]
        self.dialog_open = dialog_open
        self.cheat_button_visible = cheat_button
        self.confirm_visible = confirm_visible
        self.confirm_closes = confirm_closes
        self.select_works = select_works
        self.select_works_after = select_works_after
        self.select_attempts = 0
        self.selected_box = None
        # 实机上被选中那一行的卡片也不是整块均匀变紫，这里按比例涂
        self.highlight_fraction = highlight_fraction
        self.frames = [list(frame) for frame in list_frames]
        self.frame_index = 0
        self.clicks = []
        self.scrolls = []
        self.presses = []
        self.ts = Mock()
        self.ts.find_with_box = Mock(side_effect=self._ocr)

    # ---- 假屏幕 ----
    def get_screen(self):
        screen = np.zeros((self.yy, self.xx, 3), dtype=np.uint8)
        if self.selected_box is not None:
            # 行卡片左右边界，和引擎里判定高亮时用的数值一致
            left, right = [int(value * self.xx / 1920) for value in (470, 1450)]
            right = left + max(int((right - left) * self.highlight_fraction), 1)
            title_top = self.selected_box[2]
            top = max(int(title_top - 8 * self.yy / 1080), 0)
            bottom = max(int(title_top + 88 * self.yy / 1080), 0)
            # BGR；换算成 HSV 落在引擎判定的紫色区间里
            screen[top:bottom, left:right] = (200, 80, 120)
        self.screen = screen
        return screen

    # ---- 假 OCR ----
    def _frame_lines(self):
        return list(self.frames[min(self.frame_index, len(self.frames) - 1)])

    def _confirm_lines(self):
        if not (self.dialog_open and self.confirm_visible):
            return []
        return list(self.confirm_lines)

    def _ocr(self, box=None, **kwargs):
        box = box or [0, self.xx, 0, self.yy]
        if box == self.scaled_box([420, 1500, 240, 960]):
            # 先取当前帧再推进：否则第 0 帧永远用不到
            items = self._frame_lines()
            self.frame_index += 1
        elif box == self.scaled_box([900, 1520, 700, 1030]):
            items = self._confirm_lines()
        else:
            items = []
        return [item for item in items
                if box[0] <= item["box"][0] and item["box"][1] <= box[1]
                and box[2] <= item["box"][2] and item["box"][3] <= box[3]]

    # ---- 假 click_text ----
    def click_text(self, text, delay=0, box=None, after_delay=0, click=True,
                   find_all=False, warning=True, ocr_line=True, need_fresh=True,
                   allow_fail=False):
        # 窗口顶部的「作弊」标题：只探测、不点击
        if text == "作弊" and box == [923, 996, 230, 270]:
            return self.dialog_open
        # 掷骰界面上的「作弊」按钮：点了就开窗
        if text == "作弊" and box == [1261, 1321, 761, 792]:
            if not self.cheat_button_visible:
                return False
            self.dialog_open = True
            return True
        return False

    # ---- 键鼠 ----
    @staticmethod
    def _near_point(px, py, box, tolerance=1.0):
        cx, cy = center(box)
        return abs(px - cx) <= tolerance and abs(py - cy) <= tolerance

    def _all_rows(self):
        rows = []
        for frame in self.frames:
            rows.extend(frame)
        return rows

    def _click(self, x, y):
        self.clicks.append((round(x, 6), round(y, 6)))
        px, py = (1 - x) * self.xx, (1 - y) * self.yy
        for item in self._confirm_lines():
            if self._near_point(px, py, item["box"]):
                if self.confirm_closes:
                    self.dialog_open = False
                return
        for item in self._all_rows():
            if self._near_point(px, py, item["box"]):
                self.select_attempts += 1
                if self.select_works and self.select_attempts > self.select_works_after:
                    self.selected_box = item["box"]
                return

    def _scroll(self, direct, x=0.5, y=0.5):
        self.scrolls.append((direct, round((1 - x) * self.xx), round((1 - y) * self.yy)))

    def _press(self, key, *args, **kwargs):
        self.presses.append(key)


class CheatDialogTest(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.manager.click.side_effect = lambda x, y: self.engine._click(x, y)
        self.manager.scroll.side_effect = lambda direct, x=0.5, y=0.5: self.engine._scroll(direct, x, y)
        self.manager.press.side_effect = lambda key, *a, **k: self.engine._press(key)
        self.manager.wait = Mock()
        self.manager.sleep = Mock()
        self.enterContext(patch("core.any_fate.engine.key_mouse_manager", self.manager))
        self.logger = Mock()
        self.enterContext(patch("core.any_fate.engine.CUS_LOGGER", self.logger))
        # 作弊窗有淡出动画，等 0.15/0.2/0.3 秒都不用真等
        self.enterContext(patch("core.any_fate.engine.time", types.SimpleNamespace(sleep=Mock())))

    def make_engine(self, list_frames, **kwargs):
        self.engine = _StubCheatEngine(list_frames, **kwargs)
        return self.engine

    def assertClicked(self, *boxes):
        self.assertEqual(len(self.engine.clicks), len(boxes))
        for clicked, box in zip(self.engine.clicks, boxes):
            expected = mirror(center(box))
            self.assertAlmostEqual(clicked[0], expected[0], places=6)
            self.assertAlmostEqual(clicked[1], expected[1], places=6)

    def warnings(self):
        return "".join(str(call.args[0]) for call in self.logger.warning.call_args_list)

    def debugs(self):
        return "".join(str(call.args[0]) for call in self.logger.debug.call_args_list)

    def errors(self):
        return "".join(str(call.args[0]) for call in self.logger.error.call_args_list)


# ---------------------------------------------------------------- 列表识别
class CheatOptionEntriesTest(CheatDialogTest):
    def test_cheat_option_entries_keep_only_row_titles(self):
        engine = self.make_engine([[CIHUAI_TITLE, CIHUAI_DESC]], dialog_open=True)
        entries = engine.cheat_option_entries()
        # 说明文字在 OCR 这一步就被丢掉，只剩标题
        self.assertEqual([entry["raw_text"] for entry in entries], [CIHUAI_TITLE["raw_text"]])
        self.assertEqual(entries[0]["box"], CIHUAI_TITLE["box"])

    def test_find_cheat_option_matches_a_title_substring(self):
        engine = self.make_engine([[GUIXIN_TITLE, CIHUAI_TITLE]], dialog_open=True)
        self.assertEqual(engine.find_cheat_option(("慈怀",)), CIHUAI_TITLE)

    def test_find_cheat_option_returns_none_without_keywords(self):
        engine = self.make_engine([[CIHUAI_TITLE]], dialog_open=True)
        self.assertIsNone(engine.find_cheat_option(()))
        self.assertIsNone(engine.find_cheat_option(("",)))
        # 没有关键字就不用去 OCR
        engine.ts.find_with_box.assert_not_called()

    def test_find_cheat_option_returns_none_when_the_row_is_absent(self):
        engine = self.make_engine([[GUIXIN_TITLE]] * 3, dialog_open=True)
        self.assertIsNone(engine.find_cheat_option(("慈怀",)))

    def test_description_lines_are_never_matched(self):
        # 这行说明文字里带「慈怀」，但它是浇灌那行的说明，不是慈怀行
        engine = self.make_engine([[JIAOGUAN_DESC]] * 3, dialog_open=True)
        self.assertIsNone(engine.find_cheat_option(("慈怀",)))


# ---------------------------------------------------------------- 点击与高亮
class CheatOptionClickTest(CheatDialogTest):
    def test_click_cheat_option_clicks_the_text_center(self):
        engine = self.make_engine([[CIHUAI_TITLE]], dialog_open=True)
        selected = engine.click_cheat_option(CIHUAI_TITLE)
        self.assertEqual(selected, CIHUAI_TITLE["raw_text"])
        self.assertClicked(CIHUAI_TITLE["box"])

    def test_click_cheat_option_refuses_points_outside_the_list(self):
        engine = self.make_engine([[CIHUAI_TITLE]], dialog_open=True)
        # 框在列表上方（y 40~66），换算出来的点击位置不在列表区域内
        self.assertIsNone(engine.click_cheat_option(line("【丰饶·慈怀】", [470, 700, 40, 66])))
        self.assertEqual(engine.clicks, [])

    def test_cheat_option_selected_reads_the_highlight(self):
        engine = self.make_engine([[CIHUAI_TITLE]], dialog_open=True)
        engine.selected_box = CIHUAI_TITLE["box"]
        self.assertTrue(engine.cheat_option_selected(CIHUAI_TITLE))

    def test_cheat_option_selected_needs_the_highlight(self):
        engine = self.make_engine([[CIHUAI_TITLE]], dialog_open=True)
        self.assertFalse(engine.cheat_option_selected(CIHUAI_TITLE))

    def test_cheat_option_selected_tolerates_a_partial_highlight(self):
        # 实机上被选中那行的紫色占比约 0.41~0.51，不是整块卡片都变紫
        engine = self.make_engine([[CIHUAI_TITLE]], dialog_open=True, highlight_fraction=0.45)
        engine.selected_box = CIHUAI_TITLE["box"]
        self.assertTrue(engine.cheat_option_selected(CIHUAI_TITLE))

    def test_cheat_option_selected_rejects_a_trace_of_highlight(self):
        # 鼠标停在行上也有淡淡的紫色描边，不能算选中
        engine = self.make_engine([[CIHUAI_TITLE]], dialog_open=True, highlight_fraction=0.05)
        engine.selected_box = CIHUAI_TITLE["box"]
        self.assertFalse(engine.cheat_option_selected(CIHUAI_TITLE))


# ---------------------------------------------------------------- 找目标行
class SelectCheatOptionTest(CheatDialogTest):
    def test_select_finds_the_row_without_scrolling(self):
        engine = self.make_engine([[CIHUAI_TITLE, CIHUAI_DESC]] * 3, dialog_open=True)
        self.assertEqual(engine.select_cheat_option(("慈怀",)), CIHUAI_TITLE["raw_text"])
        # 第一屏就认出来了，一次都不用滚
        self.assertEqual(engine.scrolls, [])
        self.assertClicked(CIHUAI_TITLE["box"])

    def test_select_scrolls_down_by_three_rows(self):
        engine = self.make_engine(
            [[GUIXIN_TITLE], [GUIXIN_TITLE]] + [[CIHUAI_TITLE, CIHUAI_DESC]] * 7, dialog_open=True)
        self.assertEqual(engine.select_cheat_option(("慈怀",)), CIHUAI_TITLE["raw_text"])
        self.assertEqual(len(engine.scrolls), 2)
        left, right, top, bottom = engine.scaled_box([420, 1500, 240, 960])
        for direct, x, y in engine.scrolls:
            # 一次滚 15 格约三行，方向只能往下
            self.assertEqual(direct, -15)
            # 滚轮落在列表区域中间，不是屏幕中心
            self.assertAlmostEqual(x, (left + right) / 2)
            self.assertAlmostEqual(y, (top + bottom) / 2)
        self.assertClicked(CIHUAI_TITLE["box"])

    def test_select_gives_up_after_three_screens(self):
        engine = self.make_engine([[GUIXIN_TITLE]] * 10, dialog_open=True)
        self.assertIsNone(engine.select_cheat_option(("慈怀",)))
        # 三屏扫完就收场：每次滚 15 格
        self.assertEqual([direct for direct, _, _ in engine.scrolls], [-15, -15, -15])
        self.assertEqual(engine.clicks, [])
        self.assertIn("未找到目标骰面效果", self.warnings())

    def test_select_stops_when_the_dialog_closes(self):
        engine = self.make_engine([[CIHUAI_TITLE]] * 3, dialog_open=False)
        self.assertIsNone(engine.select_cheat_option(("慈怀",)))
        self.assertEqual(engine.clicks, [])
        self.assertEqual(engine.scrolls, [])
        self.assertIn("作弊选项窗已经不在当前画面上了", self.warnings())

    def test_select_retries_the_click_at_most_five_times(self):
        # 点了但高亮一直不出现：同一屏最多点五次，点不到就报错收场，不再换屏幕
        engine = self.make_engine([[CIHUAI_TITLE]] * 10, dialog_open=True, select_works=False)
        self.assertIsNone(engine.select_cheat_option(("慈怀",)))
        self.assertEqual(len(engine.clicks), 5)
        for clicked in engine.clicks:
            expected = mirror(center(CIHUAI_TITLE["box"]))
            self.assertAlmostEqual(clicked[0], expected[0], places=6)
            self.assertAlmostEqual(clicked[1], expected[1], places=6)
        # 目标行一屏就认到了，所以一次都不滚
        self.assertEqual(engine.scrolls, [])
        self.assertIn("未检测到高亮", self.warnings())
        self.assertIn("重试五次仍未点击成功", self.errors())

    def test_select_stops_when_the_row_disappears_during_retries(self):
        # 点过之后目标行滑出视野、OCR 认不到了：重试次数照样要减，
        # 不能一直原地重认（认不到就永远不点，也就永远减不到 0）
        frames = [[CIHUAI_TITLE]] + [[GUIXIN_TITLE]] * 9
        engine = self.make_engine(frames, dialog_open=True, select_works=False)
        self.assertIsNone(engine.select_cheat_option(("慈怀",)))
        self.assertEqual(engine.clicks, [])
        self.assertEqual(engine.scrolls, [])
        self.assertIn("又认不到目标行", self.warnings())
        self.assertIn("重试五次仍未点击成功", self.errors())

    def test_select_retries_until_the_click_takes_effect(self):
        engine = self.make_engine([[CIHUAI_TITLE]] * 10, dialog_open=True, select_works_after=1)
        self.assertEqual(engine.select_cheat_option(("慈怀",)), CIHUAI_TITLE["raw_text"])
        # 第一次点击没选中，按当前位置再点一次就成了
        self.assertClicked(CIHUAI_TITLE["box"], CIHUAI_TITLE["box"])
        self.assertIn("未检测到高亮", self.warnings())

    def test_select_does_nothing_when_stopped(self):
        engine = self.make_engine([[CIHUAI_TITLE]] * 3, dialog_open=True)
        engine._stop = True
        self.assertIsNone(engine.select_cheat_option(("慈怀",)))
        self.assertEqual(engine.clicks, [])


# ---------------------------------------------------------------- 确认
class ConfirmCheatDialogTest(CheatDialogTest):
    def test_confirm_clicks_the_ocr_text_position(self):
        engine = self.make_engine([[CIHUAI_TITLE]] * 2, dialog_open=True)
        self.assertTrue(engine.confirm_cheat_dialog())
        self.assertClicked(CONFIRM["box"])
        self.assertFalse(engine.dialog_open)
        # 点完窗收起，下一轮现查发现已经关了
        self.assertIn("作弊选项窗已关闭", self.debugs())

    def test_confirm_returns_true_when_the_dialog_was_already_closed(self):
        engine = self.make_engine([[CIHUAI_TITLE]] * 2, dialog_open=False)
        self.assertTrue(engine.confirm_cheat_dialog())
        self.assertEqual(engine.clicks, [])
        self.assertIn("作弊选项窗已关闭", self.debugs())

    def test_confirm_fails_when_the_confirm_text_is_missing(self):
        engine = self.make_engine([[CIHUAI_TITLE]] * 2, dialog_open=True, confirm_visible=False)
        self.assertFalse(engine.confirm_cheat_dialog())
        self.assertEqual(engine.clicks, [])
        self.assertIn("没识别到「确认」按钮", self.warnings())

    def test_confirm_picks_the_lowest_confirm_text(self):
        # 标题里也带「确认」两个字，只能点下面那个按钮
        engine = self.make_engine([[CIHUAI_TITLE]] * 2, dialog_open=True,
                                  confirm_lines=[CONFIRM_TITLE, CONFIRM])
        self.assertTrue(engine.confirm_cheat_dialog())
        self.assertEqual(len(engine.clicks), 1)
        clicked = engine.clicks[0]
        expected = mirror(center(CONFIRM["box"]))
        self.assertAlmostEqual(clicked[0], expected[0], places=6)
        self.assertAlmostEqual(clicked[1], expected[1], places=6)


# ---------------------------------------------------------------- 完整流程
class CheatFlowTest(CheatDialogTest):
    def test_cheat_selects_then_confirms(self):
        engine = self.make_engine([[CIHUAI_TITLE, CIHUAI_DESC]] * 3,
                                  dialog_open=True, cheat_target="慈怀")
        engine.cheat()
        self.assertClicked(CIHUAI_TITLE["box"], CONFIRM["box"])
        # 成功了就把目标清掉，免得下一轮又去作弊
        self.assertIsNone(engine.cheat_target)
        self.assertEqual(engine.presses, [])
        self.assertIn("已在作弊界面选择", "".join(
            str(call.args[0]) for call in self.logger.debug.call_args_list))

    def test_cheat_escapes_and_clears_target_when_the_row_is_missing(self):
        engine = self.make_engine([[GUIXIN_TITLE]] * 10, dialog_open=True, cheat_target="慈怀")
        engine.cheat()
        self.assertEqual(engine.clicks, [])
        # 选不中就按 esc 关窗（不消耗作弊次数），并把目标清掉
        self.assertEqual(engine.presses, ["esc"])
        self.assertIsNone(engine.cheat_target)
        self.assertIn("关掉作弊选项窗", self.warnings())

    def test_cheat_keeps_the_target_when_confirm_fails(self):
        engine = self.make_engine([[CIHUAI_TITLE]] * 3, dialog_open=True,
                                  confirm_visible=False, cheat_target="慈怀")
        engine.cheat()
        self.assertClicked(CIHUAI_TITLE["box"])
        # 点中了但没确认成功：保留目标，等下一轮再确认
        self.assertEqual(engine.cheat_target, "慈怀")
        self.assertIn("已点中，但没确认成功", self.warnings())

    def test_cheat_without_target_gives_up(self):
        engine = self.make_engine([[CIHUAI_TITLE]] * 3, dialog_open=True, cheat_target=None)
        engine.cheat()
        self.assertEqual(engine.clicks, [])
        self.assertEqual([direct for direct, _, _ in engine.scrolls], [-15, -15, -15])
        self.assertEqual(engine.presses, ["esc"])
        self.assertIn("关掉作弊选项窗", self.warnings())


if __name__ == "__main__":
    unittest.main()
