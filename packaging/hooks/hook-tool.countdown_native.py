"""打包时编译采样内核，避免发行包意外退回 Python 热循环。"""

import runpy
from pathlib import Path

root = Path(__file__).resolve().parents[2]
build = runpy.run_path(str(root / "scripts" / "build_countdown_native.py"))
binaries = [(str(build["build_native"]()), "tool")]
datas = [(str(root / "native" / "countdown" / "LICENSE-MT19937"), "tool")]
