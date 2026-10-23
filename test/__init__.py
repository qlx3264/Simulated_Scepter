"""测试包：隔离测试副作用，并让测试写入的日志与正常运行区分开。"""

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

# 必须在任何模块 import tool.log 之前设置：日志文件名在 import tool.log 时就已确定。
# tool.log 据此给测试日志加 test_ 前缀，避免测试产物混进正常运行产生的日志。
os.environ.setdefault("SIMS_TEST_LOG_PREFIX", "test_")

# 测试专用的用户目录：避免 config_path()、load_settings() 之类的写入落到真实 user 目录。
TEST_USER_DIR = Path(tempfile.mkdtemp(prefix="simsce-test-user-")).resolve()

from route import PATHS

PATHS["user"] = str(TEST_USER_DIR)
PATHS["backup"] = str(TEST_USER_DIR / "backup")


def _cleanup_test_user_dir():
    shutil.rmtree(TEST_USER_DIR, ignore_errors=True)


atexit.register(_cleanup_test_user_dir)

# 把被移入子包的模块目录加入导入路径，使测试与脚本模式的导入名一致。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool" / "divine_treasure"))
