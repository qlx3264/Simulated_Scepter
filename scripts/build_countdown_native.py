"""构建并安装倒计时 Rust 采样内核；供开发命令和 PyInstaller hook 共用。"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_native() -> Path:
    cargo = shutil.which("cargo")
    if cargo is None:
        candidate = Path.home() / ".cargo" / "bin" / ("cargo.exe" if sys.platform == "win32" else "cargo")
        if not candidate.is_file():
            raise RuntimeError("未找到 Rust cargo，请安装 Rust 工具链后重新构建")
        cargo = str(candidate)
    crate = ROOT / "native" / "countdown"
    target = crate / "target"
    subprocess.run([cargo, "build", "--release", "--offline", "--locked",
                    "--manifest-path", str(crate / "Cargo.toml"),
                    "--target-dir", str(target)], check=True, cwd=ROOT)
    filename = ("countdown_native.dll" if sys.platform == "win32" else
                "libcountdown_native.dylib" if sys.platform == "darwin" else "libcountdown_native.so")
    destination = ROOT / "tool" / ("_" + filename.removeprefix("lib"))
    shutil.copy2(target / "release" / filename, destination)
    return destination


if __name__ == "__main__":
    print(build_native())
