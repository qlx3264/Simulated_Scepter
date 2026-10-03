"""使用选定内核循环执行 JSON 动作脚本。"""

from diver import DivergentUniverse
from tool.diver.keyops import KeyController
from tool.GLOBAL import get_global_stop_flag, key_mouse_manager
from tool.public_ocr import load_actions


def run_script(engine, json_path):
    """循环识别并调用内核的动作执行器，停止或失败时释放运行资源。

    Args:
        engine: 已初始化的游戏内核，提供截图、OCR、动作执行与停止能力。
        json_path: JSON 动作脚本的路径。
    """
    engine.default_json = load_actions(json_path)
    if not engine.default_json:
        raise ValueError("动作脚本为空，无法运行")
    engine.default_json_path = json_path
    if get_global_stop_flag():
        return
    engine._stop = False
    try:
        # 差分内核的动作会使用辅助按键控制器，其线程需要在解除停止状态后创建。
        if isinstance(engine, DivergentUniverse):
            engine.keys = KeyController(engine)
        key_mouse_manager.start()
        while not engine._stop and not get_global_stop_flag():
            engine.ts.forward(engine.get_screen())
            if engine._stop or get_global_stop_flag():
                break
            engine.run_static()
    finally:
        if not engine._stop:
            engine.stop()
