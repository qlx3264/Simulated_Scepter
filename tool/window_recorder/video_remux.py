"""把分片 MP4 无损转封装为可跳转的标准 MP4。

录制输出为分片 MP4（moov 在开头、数据分片），崩溃或强制结束时可播，但分片结构
没有集中的样本表，部分播放器（例如 Windows 自带的播放器）无法建立进度索引，
表现为拖不动进度条、不能加速播放。

这里用流复制（不重新编码）把分片「卷」回标准 MP4，恢复完整的样本表。转封装在
独立线程中执行，不阻塞调用方的主任务。
"""

import os
import threading
import time

import av

from tool.log import CUS_LOGGER

# 转换产物的临时后缀；成功后替换源文件，失败则删除
TEMP_SUFFIX = ".converting.mp4"

# 按文件串行化：同一路径的两次转换会共用同一个临时文件，并发会互相覆盖导致产物损坏。
# 键为文件的绝对路径，值为对应锁。
_path_locks = {}
_registry_lock = threading.Lock()


def _lock_for(path, blocking):
    """取得某个文件的转换锁。

    Args:
        path: 录像文件路径。
        blocking: 锁被占用时是等待还是立即返回 None。

    Returns:
        已持有的锁；blocking 为 False 且已被占用时返回 None。
    """
    key = os.path.abspath(path)
    with _registry_lock:
        lock = _path_locks.setdefault(key, threading.Lock())
    if lock.acquire(blocking=blocking):
        return lock
    return None


def _remux(source_path, target_path):
    """流复制转封装，返回复用的包数。

    Args:
        source_path: 分片 MP4 源文件。
        target_path: 输出文件，非分片 MP4。

    Returns:
        复用的包数量。

    Raises:
        OSError: 源文件无法读取或输出无法写入。
        av.FFmpegError: 解复用、复用失败（含源文件尾部损坏且无法恢复的情况）。
    """
    packets = 0
    source = av.open(source_path)
    try:
        video_streams = list(source.streams.video)
        if not video_streams:
            raise OSError(f"源文件不含视频轨道: {source_path}")

        output = av.open(target_path, mode="w", format="mp4")
        try:
            out_streams = [output.add_stream_from_template(stream)
                           for stream in video_streams]
            for stream, out_stream in zip(video_streams, out_streams):
                for packet in source.demux(stream):
                    if packet.dts is None:
                        continue
                    packet.stream = out_stream
                    output.mux(packet)
                    packets += 1
        finally:
            output.close()
    finally:
        # Windows 上必须先释放源文件句柄，之后才能 os.replace 覆盖它
        source.close()
    return packets


def _decode_ok(path):
    """验证文件能否完整解码，返回 (是否无错, 解出的帧数)。

    转封装是流复制，尾部残片会被带进产物。与其猜测截断点，不如解码一遍确认：
    只有解得干净才允许替换原文件。
    """
    frames = 0
    try:
        container = av.open(path)
    except Exception as e:
        CUS_LOGGER.debug(f"成品无法打开：{path}（{e}）")
        return False, 0
    try:
        for packet in container.demux(video=0):
            if packet.dts is None:
                continue
            for _ in packet.decode():
                frames += 1
    except Exception as e:
        CUS_LOGGER.debug(f"成品解码中断：{path}（{type(e).__name__}: {e}）")
        return False, frames
    finally:
        container.close()
    return True, frames


def convert_to_standard_mp4(path, check_source=True, check_output=True):
    """把分片 MP4 转封装为可跳转的标准 MP4，就地替换原文件。

    转换在临时文件上进行：成功则替换源文件（名字保持不变），失败则删除临时文件、
    保留原始文件。因此无论成败，调用方都不会丢掉录像。

    两个开关对应两种调用场景：

    - **自动转换（录制正常结束）**：写入器已正常收尾，文件结构完整、不含损坏帧，
      因此两个校验都可以关掉，不必解码全片。见 `stop_recording` 的调用。
    - **手动转换（UI 里转换已有文件）**：文件可能来自被强杀的进程，尾部有半个分片。
      此时 `check_source` 会先判定源文件是否完整：完整才按标准流程转换；不完整则
      由 `check_output` 决定是「跳过、保持原样」还是「强行封装、抢救可播部分」。

    Args:
        path: 待转换的录像文件路径。
        check_source: 是否先校验源文件完整；不完整时直接返回 False。
        check_output: 是否要求成品完整解码后才替换；False 表示接受尾部损坏。

    Returns:
        转换成功返回 True；源文件不存在、无需转换或转换失败返回 False。
    """
    if not os.path.exists(path):
        CUS_LOGGER.warning(f"待转换文件不存在：{path}")
        return False

    # 同一文件可能被两次任务结束先后触发（上次转换尚未完成）。共用临时文件会互相
    # 覆盖，因此这里只允许一个转换进行；第二个调用认为文件已交给正在进行的转换处理。
    lock = _lock_for(path, blocking=False)
    if lock is None:
        CUS_LOGGER.debug(f"该文件正在转换中，跳过重复调用：{path}")
        return True
    try:
        return _convert_locked(path, check_source=check_source, check_output=check_output)
    finally:
        lock.release()


def _convert_locked(path, check_source=True, check_output=True):
    """已持有该文件转换锁的前提下执行转换。"""
    if check_source:
        # 源文件尾部残缺时无法转成「干净」的标准 mp4，保持原样交给用户决定
        source_ok, _ = _decode_ok(path)
        if not source_ok:
            CUS_LOGGER.info(
                f"录像尾部不完整（可能被强制结束），保持分片格式不转换：{path}")
            return False

    target = path + TEMP_SUFFIX
    if os.path.exists(target):
        os.remove(target)

    try:
        packets = _remux(path, target)
    except Exception as e:
        CUS_LOGGER.warning(f"转封装失败，保留原文件：{path}（{type(e).__name__}: {e}）")
        _discard(target)
        return False

    if packets == 0:
        CUS_LOGGER.warning(f"转封装未复用任何数据，保留原文件：{path}")
        _discard(target)
        return False

    # 只有要求校验时才解码成品；自动转换的源文件已由写入器正常收尾，无需校验
    if check_output:
        ok, frames = _decode_ok(target)
        if not ok or frames == 0:
            CUS_LOGGER.warning(f"转封装成品无法完整解码，保留原文件：{path}")
            _discard(target)
            return False
        detail = f"{frames} 帧"
    else:
        detail = "未校验解码"

    if not _replace_with_retry(target, path):
        CUS_LOGGER.warning(f"替换录像文件失败，保留原文件：{path}")
        _discard(target)
        return False

    CUS_LOGGER.debug(
        f"已转封装为可跳转的标准 MP4：{path}（复用 {packets} 个包，{detail}）")
    return True


def _replace_with_retry(target, path, attempts=5, delay=0.2):
    """替换目标文件，失败时短暂重试。

    Windows 上文件句柄的释放可能滞后于 close()，紧接着替换会得到“拒绝访问”。
    """
    for attempt in range(attempts):
        try:
            os.replace(target, path)
            return True
        except OSError as e:
            if attempt == attempts - 1:
                CUS_LOGGER.warning(f"替换失败（已重试 {attempts} 次）：{e}")
                return False
            time.sleep(delay)
    return False


def _discard(path):
    """删除转换失败的产物；删不掉时只记录，不影响调用方。"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        CUS_LOGGER.warning(f"删除转换失败的文件失败：{path}（{e}）")


def convert_in_background(path, on_finished=None, check_source=True, check_output=True):
    """在独立线程中转封装，不阻塞调用方。

    Args:
        path: 待转换的录像文件路径。
        on_finished: 可选回调，接收转换结果（bool）；在转换线程中执行。
        check_source: 是否先校验源文件完整。
        check_output: 是否要求成品完整解码后才替换；两者语义见 convert_to_standard_mp4。

    Returns:
        已启动的线程对象。
    """
    def worker():
        try:
            converted = convert_to_standard_mp4(
                path, check_source=check_source, check_output=check_output)
        except Exception as e:
            CUS_LOGGER.error(f"转封装线程发生异常：{e}")
            converted = False
        if on_finished is not None:
            on_finished(converted)

    thread = threading.Thread(target=worker, name="录像转封装", daemon=True)
    thread.start()
    return thread
