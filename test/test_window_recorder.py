"""隔离窗口与编解码依赖，验证录制停止请求、录制文件处置与录制线程停止。"""

import glob
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

FAKE_WINDOW_RECT = (623, 5, 2565, 1141)
TEMP_PREFIX = ".recorder-test-"


def make_box(box_type, payload=b""):
    """按 ISO BMFF 结构拼一个 box：4 字节长度 + 4 字节类型 + 内容。"""
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + box_type + payload


FTYP_BOX = make_box(b"ftyp", b"isom")
MDAT_BOX = make_box(b"mdat", b"\x00" * 4)
# 完整视频以 moov 索引结束；缺少它的文件体积正常却无法播放。
MOOV_BOX = make_box(b"moov", b"mvhd\x00\x00\x00\x00")


class FakeVideoWriter:
    """记录写入与释放情况的最小 VideoWriter 替身。"""

    def __init__(self, path, *_args):
        self.path = path
        self.written = 0
        self.released = False
        with open(path, "wb") as file:
            file.write(FTYP_BOX)
            file.write(MDAT_BOX)

    def isOpened(self):
        return True

    def write(self, _frame):
        self.written += 1

    def release(self):
        self.released = True
        with open(self.path, "ab") as file:
            file.write(MOOV_BOX)


def load_recorder_module(cv2_stub, thread_class):
    """加载录制器模块，用替身隔离 OpenCV、截图与线程实现。"""
    module_names = (
        "cv2", "numpy", "PIL", "PIL.ImageGrab", "win32gui", "win32ui",
        "route", "tool.log", "tool.thread", "tool.utils.game_window",
    )
    saved = {name: sys.modules.get(name) for name in module_names}

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.array = lambda value, *args, **kwargs: value
    fake_numpy.uint8 = object
    fake_numpy.float32 = object

    fake_grab = types.ModuleType("PIL.ImageGrab")
    fake_grab.grab = Mock()

    game_window = types.ModuleType("tool.utils.game_window")
    game_window.CLOUD_WINDOW_KIND = "cloud"
    game_window.LOCAL_GAME_TITLE = "崩坏：星穹铁道"
    game_window.find_game_window = lambda **kwargs: None
    game_window.get_client_screen_rect = lambda hwnd: FAKE_WINDOW_RECT
    game_window.get_window_kind = lambda hwnd: "local"
    game_window.is_usable_game_window = lambda hwnd: True

    win32gui = types.ModuleType("win32gui")
    win32gui.FindWindow = lambda *args: 1234
    win32gui.GetWindowRect = lambda hwnd: FAKE_WINDOW_RECT
    win32gui.IsWindow = lambda hwnd: True
    win32gui.IsWindowVisible = lambda hwnd: True
    win32gui.ReleaseDC = lambda hwnd, dc: None
    win32gui.DeleteObject = lambda handle: None

    logger = types.ModuleType("tool.log")
    logger.CUS_LOGGER = Mock()

    thread_module = types.ModuleType("tool.thread")
    thread_module.ThreadWithException = thread_class

    for name in ("tool", "tool.utils"):
        module = sys.modules.setdefault(name, types.ModuleType(name))
        module.__path__ = []
    sys.modules["tool"].log = logger
    sys.modules["tool"].thread = thread_module
    sys.modules["tool.utils"].game_window = game_window

    sys.modules.update({
        "cv2": cv2_stub,
        "numpy": fake_numpy,
        "PIL": types.ModuleType("PIL"),
        "PIL.ImageGrab": fake_grab,
        "win32gui": win32gui,
        "win32ui": Mock(),
        "route": types.ModuleType("route"),
        "tool.log": logger,
        "tool.thread": thread_module,
        "tool.utils.game_window": game_window,
    })
    sys.modules["route"].PATHS = {"video": "."}

    try:
        path = Path(__file__).resolve().parents[1] / "tool" / "window_recorder.py"
        spec = importlib.util.spec_from_file_location("window_recorder_under_test", path)
        recorder_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recorder_module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    return recorder_module


class WindowRecorderTests(unittest.TestCase):
    def setUp(self):
        # 录制文件写到工作区根目录：新建子目录在部分环境下不可写，测试只需唯一前缀。
        self.test_dir = Path(__file__).resolve().parents[1]
        self.output_path = str(self.test_dir)
        self.cv2_stub = types.ModuleType("cv2")
        self.cv2_stub.VideoWriter = FakeVideoWriter
        self.cv2_stub.VideoWriter_fourcc = lambda *args: 0

        # 录制线程替身不执行线程体，is_alive 默认返回 False，启动后即视为已结束。
        self.thread_class = Mock(side_effect=lambda **kwargs: Mock(is_alive=Mock(return_value=False)))
        module = load_recorder_module(self.cv2_stub, self.thread_class)
        self.module = module
        # 真实上限为 5 秒，测试用极小值等价覆盖同一分支，避免拖慢测试。
        self.recorder = module.WindowRecorder(
            output_path=self.output_path,
            window_title="崩坏：星穹铁道",
            window_class_name="UnityWndClass",
            stop_thread_timeout=0.05,
        )

    def tearDown(self):
        for path in glob.glob(str(self.test_dir / f"{TEMP_PREFIX}*")):
            os.remove(path)

    @property
    def output_file(self):
        return self.recorder.output_file

    def write_recording(self, with_moov=True):
        """在测试目录写入一份模拟录制文件。"""
        self.recorder.output_file = os.path.join(
            self.output_path, f"{TEMP_PREFIX}第1次轮回-20261003_123515.mp4")
        with open(self.output_file, "wb") as file:
            file.write(FTYP_BOX)
            file.write(MDAT_BOX)
            if with_moov:
                file.write(MOOV_BOX)

    def test_stop_between_sessions_refuses_next_start_and_keeps_request(self):
        # 原缺陷：两次录制之间到达的停止请求被随后的启动清除，录制继续到进程退出，
        # 视频缺少 moov 索引而无法播放，也没有任何一方再删除该文件。
        self.recorder.stop_recording(delete_video=True)
        self.assertTrue(self.recorder.stop_event.is_set())

        self.recorder.start_recording(1)

        self.assertFalse(self.recorder.recording)
        self.assertTrue(self.recorder.stop_event.is_set())
        self.assertIsNone(self.recorder.out)
        self.assertEqual(self.thread_class.call_count, 0)
        self.assertFalse(hasattr(self.recorder, "output_file"))

    def test_stop_during_initialization_discards_writer_and_file(self):
        original_writer = self.cv2_stub.VideoWriter

        def stop_after_open(path, *args):
            self.recorder.stop_event.set()
            return original_writer(path, *args)

        self.cv2_stub.VideoWriter = stop_after_open
        self.recorder.start_recording(2)

        self.assertFalse(self.recorder.recording)
        self.assertIsNone(self.recorder.out)
        self.assertEqual(self.thread_class.call_count, 0)
        self.assertFalse(os.path.exists(self.output_file))

    def test_stop_removes_unplayable_recording(self):
        self.write_recording(with_moov=False)
        self.recorder.recording = True

        self.recorder.stop_recording()

        self.assertFalse(os.path.exists(self.output_file))

    def test_stop_keeps_playable_recording_and_updates_filename(self):
        self.recorder.stop_event.clear()
        self.write_recording()
        self.recorder.recording = True

        self.recorder.stop_recording(battle_count=2)

        self.assertTrue(os.path.exists(self.output_file))
        self.assertIn("2战-", self.output_file)

    def test_is_playable_reports_missing_index(self):
        self.write_recording(with_moov=False)
        self.assertFalse(self.recorder._is_playable())

        with open(self.output_file, "ab") as file:
            file.write(MOOV_BOX)

        self.assertTrue(self.recorder._is_playable())

    def test_stopping_thread_ends_process_without_touching_recording(self):
        # 录制线程卡在无法中断的调用里时，继续等待只会让视频一直增长且始终缺少
        # moov 索引；此时应结束进程，而不是删掉或重命名一个仍在写入的文件。
        self.write_recording(with_moov=False)
        self.recorder.recording = True
        self.recorder.recording_thread = Mock(is_alive=Mock(return_value=True))
        self.recorder._halt_process = Mock()
        self.recorder._halt_process.side_effect = SystemExit(1)

        with self.assertRaises(SystemExit):
            self.recorder.stop_recording(delete_video=True, battle_count=2)

        self.recorder.recording_thread.join.assert_called_once_with(
            timeout=self.recorder.stop_thread_timeout)
        self.recorder._halt_process.assert_called_once_with()
        self.assertTrue(os.path.exists(self.output_file))
        self.assertNotIn("2战-", self.output_file)

    def test_thread_stopped_in_time_does_not_end_process(self):
        self.write_recording()
        self.recorder.recording = True
        self.recorder.recording_thread = Mock(is_alive=Mock(return_value=False))
        self.recorder._halt_process = Mock()

        self.recorder.stop_recording()

        self.recorder._halt_process.assert_not_called()
        self.assertTrue(os.path.exists(self.output_file))

    def test_stop_timeout_defaults_to_five_seconds(self):
        recorder = self.module.WindowRecorder(output_path=self.output_path)

        self.assertEqual(recorder.stop_thread_timeout, 5.0)

    def test_halt_process_exits_with_failure_code(self):
        with patch.object(self.module.os, "_exit") as exit_process:
            self.recorder._halt_process()

        exit_process.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
