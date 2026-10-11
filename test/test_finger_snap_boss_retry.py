"""弹指一挥识图在慈怀闪光下漏检 boss/head 的回归测试。"""
import unittest
from unittest.mock import Mock, call, patch

from importing import load_img

# core.finger_snap.engine 在导入期会读取灰度模板，必须先载入图像缓存。
load_img()

from core.finger_snap.engine import FingerSnap
from tool.utils.Error import NoBossError, NoMatchError

FLASH_FRAME = "慈怀闪光帧"
STABLE_FRAME = "闪光结束后的稳定帧"
BOSS_HEAD_MATCH = {"name": "head", "location": (1182, 385), "similarity": 0.71}
OTHER_MATCH = {"name": "battle", "location": (945, 301), "similarity": 0.76}
GRAPH_NODE = {"idx": 0, "name": "head", "cx": 1182.0, "cy": 385.0}


def fake_match(image, mode=1):
    """按截图内容返回识别结果：闪光帧漏检 boss/head，稳定帧能识别到。"""
    if image == STABLE_FRAME:
        return [dict(BOSS_HEAD_MATCH)]
    return [dict(OTHER_MATCH)]


class FakeAgent:
    """替代决策模型，只记录载入的节点。"""

    def __init__(self):
        self.loaded = None

    def load_map(self, nodes, edges, start_idx, infected, *args, **kwargs):
        self.loaded = (nodes, edges, start_idx, infected)
        return Mock()


class FakeEngine:
    """只提供 try_analysis_map 依赖的最小状态。

    截图按 frames 顺序由 get_screen 逐次返回，用于区分缓存截图与新抓截图。
    单次识图复用生产实现，测试只替换截图与识图、建图依赖。
    """

    _analysis_map_once = FingerSnap._analysis_map_once

    def __init__(self, frames):
        self.frames = list(frames)
        self.screen = self.frames[0]
        self.refreshed = 0
        self.debug = False
        self.countdown = 15
        self.plane_floor = 1
        self.fail_match_count = 0
        self._cheat_count = 0
        self._reroll_count = 0
        self.nodes = None
        self.edges = None
        self.start_nodes = None
        self.path = None
        self.next_node = None
        self.replace_idx = None
        self.countdown_agent = FakeAgent()

    def get_screen(self):
        self.refreshed += 1
        self.screen = self.frames[min(self.refreshed, len(self.frames) - 1)]
        return self.screen

    def click_text(self, **kwargs):
        return False

    def save_screen(self, **kwargs):
        return None


class BossRetryTest(unittest.TestCase):
    def patch_analysis(self, matches):
        """打桩识图与建图依赖，返回（识图桩，等待桩）。"""
        matcher = self.enterContext(
            patch("core.finger_snap.engine.match_multiple_targets",
                  side_effect=matches))
        self.enterContext(
            patch("core.finger_snap.engine.detect_corner_markers",
                  return_value=[]))
        self.enterContext(
            patch("core.finger_snap.engine.detect_infectable_nodes",
                  return_value=[]))
        self.enterContext(
            patch("core.finger_snap.engine.build_rightward_graph2",
                  return_value=([dict(GRAPH_NODE)], {0: [1]}, 0)))
        sleeper = self.enterContext(patch("core.finger_snap.engine.time.sleep"))
        return matcher, sleeper

    def test_flash_frame_missing_boss_is_retried_with_fresh_screen(self):
        engine = FakeEngine([FLASH_FRAME, STABLE_FRAME])
        matcher, sleeper = self.patch_analysis(fake_match)

        result = FingerSnap.try_analysis_map(engine, 1)

        self.assertTrue(result)
        # 第一次用缓存截图漏检 boss/head，第二次刷新截图后识别成功。
        self.assertEqual([call(FLASH_FRAME, 1), call(STABLE_FRAME, 1)],
                         matcher.call_args_list)
        self.assertEqual(1, engine.refreshed)
        self.assertEqual([call(0.3)], sleeper.call_args_list)
        self.assertEqual("head", engine.countdown_agent.loaded[0][0]["name"])

    def test_single_pass_on_flash_frame_reports_missing_boss(self):
        """闪光帧单次识图确实缺 boss/head，这正是修复前的中断路径。"""
        engine = FakeEngine([FLASH_FRAME])
        self.patch_analysis(fake_match)

        with self.assertRaises(NoBossError):
            FingerSnap._analysis_map_once(engine, 1)

    def test_missing_boss_after_all_attempts_still_raises(self):
        engine = FakeEngine([FLASH_FRAME])
        matcher, sleeper = self.patch_analysis(fake_match)

        with self.assertRaises(NoBossError):
            FingerSnap.try_analysis_map(engine, 1)

        self.assertEqual(3, matcher.call_count)
        self.assertEqual(2, sleeper.call_count)

    def test_empty_match_keeps_the_original_no_match_failure(self):
        engine = FakeEngine([FLASH_FRAME])
        matcher, sleeper = self.patch_analysis(lambda image, mode=1: [])

        with self.assertRaises(NoMatchError):
            FingerSnap.try_analysis_map(engine, 1)

        # 空匹配只按原逻辑刷新一次截图，不进入 boss/head 重试。
        self.assertEqual(2, matcher.call_count)
        self.assertEqual(0, sleeper.call_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
