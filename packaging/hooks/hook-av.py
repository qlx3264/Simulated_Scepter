"""PyInstaller 钩子：收集 PyAV 依赖的 FFmpeg 共享库。

PyAV 的 FFmpeg 库不在 `av` 包目录内，而是放在同级的 `av.libs` 里，
PyInstaller 6.10 没有内置钩子，不打进包会在运行时报找不到动态库。
"""

from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = collect_dynamic_libs("av")
