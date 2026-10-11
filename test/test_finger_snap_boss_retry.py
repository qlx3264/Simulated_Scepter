"""弹指一挥识图在慈怀闪光下漏检 boss/head 的回归测试。

闪光期间的漏检在限定时间内刷新截图复核；窗口结束仍漏检时记错误日志并返回，
不抛异常中断流程。
"""
import unittest
from unittest.mock import Mock, call, patch

from importing import load_img

# core.finger_snap.engine 在导入期会读取灰度模板，必须先载入图像缓存。
load_img()

from core.finger_snap.engine import FingerSnap
from tool.utils.Error import NoMatchError

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


class FakeClock:
    """可推进的单调时钟。

    每抓一次截图代表一次识图开销，用于在不真实等待的情况下跑完识别窗口。
    """

    def __init__(self, seconds_per_screen):
        self.now = 0.0
        self.seconds_per_screen = seconds_per_screen

    def __call__(self):
        return self.now

    def elapse(self):
        self.now += self.seconds_per_screen


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
    识图与建图依赖由用例打桩，复核循环本身用生产实现。
    """

    def __init__(self, frames, clock=None):
        self.frames = list(frames)
        self.screen = self.frames[0]
        self.clock = clock
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
        if self.clock is not None:
            self.clock.elapse()
        self.refreshed += 1
        self.screen = self.frames[min(self.refreshed, len(self.frames) - 1)]
        return self.screen

    def click_text(self, **kwargs):
        return False

    def save_screen(self, **kwargs):
        return None


class BossRetryTest(unittest.TestCase):
    def patch_analysis(self, matches, clock=None):
        """打桩识图、建图依赖、日志与时钟，返回（识图桩，睡眠桩，日志桩）。"""
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
        logger = self.enterContext(patch("core.finger_snap.engine.CUS_LOGGER"))
        if clock is not None:
            self.enterContext(
                patch("core.finger_snap.engine.time.monotonic", clock))
        sleeper = self.enterContext(patch("core.finger_snap.engine.time.sleep"))
        return matcher, sleeper, logger

    def test_flash_frame_missing_boss_is_retried_with_fresh_screen(self):
        clock = FakeClock(seconds_per_screen=0.2)
        engine = FakeEngine([FLASH_FRAME, STABLE_FRAME], clock)
        matcher, sleeper, logger = self.patch_analysis(fake_match, clock)

        result = FingerSnap.try_analysis_map(engine, 1)

        self.assertTrue(result)
        # 第一次用缓存截图漏检 boss/head，第二次刷新截图后识别成功。
        self.assertEqual([call(FLASH_FRAME, 1), call(STABLE_FRAME, 1)],
                         matcher.call_args_list)
        self.assertEqual(1, engine.refreshed)
        # 复核靠反复识图完成，不阻塞主进程。
        self.assertFalse(sleeper.called)
        self.assertFalse(logger.error.called)
        self.assertEqual("head", engine.countdown_agent.loaded[0][0]["name"])

    def test_expired_window_still_rechecks_once_before_returning(self):
        """识别窗口已到期时也要复核一次，之后记错误日志并返回。"""
        clock = FakeClock(seconds_per_screen=10.0)  # 单次识图远超 3 秒窗口
        engine = FakeEngine([FLASH_FRAME], clock)
        matcher, sleeper, logger = self.patch_analysis(fake_match, clock)

        result = FingerSnap.try_analysis_map(engine, 1)

        self.assertFalse(result)
        # 首次用缓存截图 + 一次刷新截图的复核，没有阻塞等待。
        self.assertEqual(2, matcher.call_count)
        self.assertEqual(1, engine.refreshed)
        self.assertFalse(sleeper.called)
        self.assertEqual(1, logger.error.call_count)

    def test_missing_boss_after_window_returns_without_interrupting(self):
        """识别窗口内始终缺 boss/head 时记错误日志后返回，不抛异常中断流程。"""
        clock = FakeClock(seconds_per_screen=2.0)  # 每次识图耗时 2 秒
        engine = FakeEngine([FLASH_FRAME], clock)
        matcher, sleeper, logger = self.patch_analysis(fake_match, clock)

        result = FingerSnap.try_analysis_map(engine, 1)

        self.assertFalse(result)
        # 首次识图 + 窗口内的两次复核，全部由连续识图完成，没有阻塞等待。
        self.assertEqual(3, matcher.call_count)
        self.assertEqual(2, engine.refreshed)
        self.assertFalse(sleeper.called)
        self.assertEqual(1, logger.error.call_count)

    def test_empty_match_keeps_the_original_no_match_failure(self):
        clock = FakeClock(seconds_per_screen=0.2)
        engine = FakeEngine([FLASH_FRAME], clock)
        matcher, sleeper, _ = self.patch_analysis(
            lambda image, mode=1: [], clock)

        with self.assertRaises(NoMatchError):
            FingerSnap.try_analysis_map(engine, 1)

        # 空匹配只按原逻辑刷新一次截图，不进入 boss/head 的识别窗口。
        self.assertEqual(2, matcher.call_count)
        self.assertEqual(1, engine.refreshed)
        self.assertFalse(sleeper.called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
