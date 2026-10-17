"""Rust 批量采样桥接；Python 保留推荐编排和可独立复核的参考实现。"""

from __future__ import annotations

import ctypes as ct
import sys
import time
from functools import lru_cache, wraps
from pathlib import Path
from threading import get_ident


class Context(ct.Structure):
    _fields_ = [("mask", ct.c_uint64), ("cd", ct.c_int64),
                ("node", ct.c_uint32), ("cheat", ct.c_uint32), ("reroll", ct.c_uint32),
                ("phase", ct.c_uint8), ("observed", ct.c_uint8),
                ("locked", ct.c_uint8), ("action", ct.c_uint8)]


class Node(ct.Structure):
    _fields_ = [("future", ct.c_uint64), ("neighbors", ct.c_uint64), ("bonus", ct.c_double),
                ("steps", ct.c_uint32), ("edge_count", ct.c_uint32), ("edges", ct.c_uint8 * 64)]


class Stats(ct.Structure):
    _fields_ = [("count", ct.c_uint64), ("total", ct.c_double), ("total_sq", ct.c_double),
                ("wins", ct.c_uint64), ("target_count", ct.c_uint64),
                ("minimum", ct.c_double), ("maximum", ct.c_double)]


class Record(ct.Structure):
    _fields_ = [("key", Context), ("stats", Stats), ("mode", ct.c_uint32)]


class Policy(ct.Structure):
    _fields_ = [("key", Context), ("action", ct.c_uint32), ("mode", ct.c_uint32)]


class Visit(ct.Structure):
    _fields_ = [("context", Context), ("count", ct.c_uint64)]


class Job(ct.Structure):
    _fields_ = [("initial", Context), ("seed", ct.c_int64), ("budget", ct.c_uint64),
                ("epsilon_start", ct.c_double), ("epsilon_end", ct.c_double),
                ("target", ct.c_double), ("actions", ct.c_uint8 * 64),
                ("action_count", ct.c_uint32), ("kind", ct.c_uint32), ("has_target", ct.c_uint32)]


class Result(ct.Structure):
    _fields_ = [("stats", Stats * 64), ("completed", ct.c_uint64), ("trials", ct.c_uint64)]


Poll = ct.CFUNCTYPE(ct.c_int, ct.c_uint64)
PHASES = ("terminal", "effect", "target", "path", "settle")
PHASE_IDS = {phase: index for index, phase in enumerate(PHASES)}
STAT_FIELDS = tuple(name for name, _ in Stats._fields_)


@lru_cache(maxsize=1)
def load_library():
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    path = Path(__file__).with_name("_countdown_native" + suffix)
    if not path.is_file():
        return None
    library = ct.CDLL(str(path))
    signatures = {
        "abi": ([], ct.c_uint32),
        "layout": ([ct.c_uint32], ct.c_size_t),
        "error": ([], ct.c_char_p),
        "create": ([ct.POINTER(Node), ct.c_size_t, ct.c_uint64], ct.c_void_p),
        "destroy": ([ct.c_void_p], None),
        "load": ([ct.c_void_p, ct.POINTER(Record), ct.c_size_t], ct.c_int),
        "policies": ([ct.c_void_p, ct.POINTER(Policy), ct.c_size_t, ct.c_double, ct.c_uint32], ct.c_int),
        "run": ([ct.c_void_p, ct.POINTER(Job), ct.POINTER(Result), Poll], ct.c_int),
        "records": ([ct.c_void_p, ct.POINTER(ct.c_size_t)], ct.POINTER(Record)),
        "visits": ([ct.c_void_p, ct.POINTER(ct.c_size_t)], ct.POINTER(Visit)),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, "countdown_" + name)
        function.argtypes, function.restype = arguments, result
    layouts = (Context, Node, Stats, Record, Policy, Visit, Job, Result)
    if (library.countdown_abi() != 1 or any(
            library.countdown_layout(index) != ct.sizeof(layout)
            for index, layout in enumerate(layouts))):
        raise RuntimeError("倒计时 Rust 内核 ABI 不匹配，请重新运行 scripts/build_countdown_native.py")
    return library


def sampling_session(method):
    """一次外层调用只传输一次 Q 表；嵌套校准共享内核，结束后释放内存。"""
    @wraps(method)
    def wrapped(controller, *args, **kwargs):
        if getattr(controller, "_native_scope", False):
            if (controller._native_owner != get_ident()
                    or getattr(controller._native_session, "running", False)):
                raise RuntimeError("同一倒计时控制器不能并发采样或在采样回调中重入")
            return method(controller, *args, **kwargs)
        controller._native_scope = True
        controller._native_owner = get_ident()
        controller._native_session = None
        try:
            return method(controller, *args, **kwargs)
        finally:
            session = controller._native_session
            if session:
                session.close()
            controller._native_session = None
            controller._native_scope = False
    return wrapped


def get_session(controller, context):
    from tool.countdown_evaluator import MonteCarloController

    session = controller._native_session
    if session is not None:
        return session or None
    # 子类/实例覆盖继续使用 Python，避免绕过调用者扩展的规则。
    supported = (type(controller) is MonteCarloController
                 and not any(name in controller.__dict__ for name in
                             ("_rollout", "_advance", "_greedy_action", "_greedy_win_action"))
                 and len(controller.map.nodes) <= 64)
    state = context.state
    full_mask = sum(1 << node for node in controller.map.node_map)
    supported = (supported and state.infected >= 0 and not state.infected & ~full_mask
                 and abs(state.countdown) < 2 ** 52 - 4096
                 and 0 <= state.cheat_rem < 2 ** 32 and 0 <= state.reroll_rem < 2 ** 32
                 and abs(controller.config.seed) < 2 ** 62)
    library = load_library() if supported else None
    try:
        session = NativeSession(controller, library) if library is not None else False
    except OverflowError:
        # 已有 Q 表也可能来自 Python 支持的大整数状态，不能截断后导入。
        session = False
    controller._native_session = session
    controller.sampling_backend = "rust" if session else "python"
    return session or None


class NativeSession:
    def __init__(self, controller, library):
        self.controller = controller
        self.library = library
        self.handle = None
        self.running = False
        self.ids = sorted(controller.map.node_map)
        self.indices = {node: index for index, node in enumerate(self.ids)}
        self.full_mask = sum(1 << node for node in self.ids)
        self._dense_ids = self.ids == list(range(len(self.ids)))
        from tool.countdown_evaluator import _PATH_SCORE_BONUS_FIELD

        nodes = (Node * len(self.ids))()
        for index, node in enumerate(self.ids):
            edges = controller.map.edges[node]
            field = _PATH_SCORE_BONUS_FIELD.get(str(controller.map.node_map[node].get("name", "")))
            nodes[index] = Node(
                self.pack_mask(controller.map.future_masks[node]),
                self.pack_mask(controller.map.neighbor_masks[node]),
                getattr(controller.config, field) if field else 0.0,
                controller.map.longest_steps_from(node), len(edges),
                (ct.c_uint8 * 64)(*(self.indices[edge] for edge in edges)))
        self.handle = library.countdown_create(nodes, len(nodes), min(2, controller.config.min_visits))
        self.check(self.handle)
        try:
            for mode, table in enumerate((controller.q, controller.win_q)):
                records = (Record * len(table))()
                for index, (key, stats) in enumerate(table.items()):
                    records[index] = Record(self.pack_key(key, mode, True),
                                            Stats(*(getattr(stats, field) for field in STAT_FIELDS)), mode)
                self.check(library.countdown_load(self.handle, records, len(records)))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.handle:
            self.library.countdown_destroy(self.handle)
            self.handle = None

    def check(self, success):
        if not success:
            message = self.library.countdown_error().decode("utf-8", errors="replace")
            raise RuntimeError("倒计时 Rust 内核失败: " + message)

    def pack_mask(self, mask):
        if mask < 0 or mask & ~self.full_mask:
            raise OverflowError("感染掩码含有地图外的节点")
        if self._dense_ids:
            return mask
        return sum(1 << index for index, node in enumerate(self.ids) if mask & (1 << node))

    def unpack_mask(self, mask):
        if self._dense_ids:
            return mask
        return sum(1 << node for index, node in enumerate(self.ids) if mask & (1 << index))

    def pack_action(self, phase, action):
        if phase == 1:
            return 0 if action == "keep" else 7 if action == "reroll" else action[1]
        return self.indices[action]

    def unpack_action(self, phase, action):
        if phase == 1:
            return "keep" if action == 0 else "reroll" if action == 7 else ("cheat", action)
        return self.ids[action]

    def pack_context(self, context):
        state = context.state
        return self.checked_context(
            self.pack_mask(state.infected), state.countdown, self.indices[state.node_idx],
            state.cheat_rem, state.reroll_rem, PHASE_IDS[context.phase],
            context.observed_effect or 0, context.locked_effect or 0, 0)

    @staticmethod
    def checked_context(mask, cd, node, cheat, reroll, phase, observed, locked, action):
        if not (abs(cd) < 2 ** 52 - 4096 and 0 <= cheat < 2 ** 32 and 0 <= reroll < 2 ** 32):
            raise OverflowError("CD 或资源超出 Rust 采样范围")
        return Context(mask, cd, node, cheat, reroll, phase, observed, locked, action)

    def unpack_context(self, context):
        from tool.countdown_evaluator import CountdownState, DecisionContext

        return DecisionContext(PHASES[context.phase], CountdownState(
            self.ids[context.node], self.unpack_mask(context.mask), context.cd, context.cheat, context.reroll),
            context.observed or None, context.locked or None)

    def pack_key(self, key, win, q=False):
        phase = PHASE_IDS[key[0]]
        state = key[1]
        if phase == 4:
            return self.checked_context(self.pack_mask(state[1]), key[3] if win else 0, self.indices[state[0]],
                                        state[2], state[3], phase, key[2], 0, 0)
        offset = int(bool(win))
        return self.checked_context(
            self.pack_mask(state[1]), state[2] if win else 0, self.indices[state[0]],
            state[2 + offset], state[3 + offset], phase, key[2] or 0, int(key[3] or 0),
            self.pack_action(phase, key[4]) if q else 0)

    def unpack_key(self, key, win):
        state = (self.ids[key.node], self.unpack_mask(key.mask))
        resources = (key.cheat, key.reroll)
        if key.phase == 4:
            result = ("settle", state + resources, key.observed)
            return result + (key.cd,) if win else result
        state += (key.cd,) if win else ()
        return (PHASES[key.phase], state + resources,
                key.observed if key.phase == 1 else None,
                bool(key.locked) if key.phase == 3 else None,
                self.unpack_action(key.phase, key.action))

    def sync_policies(self):
        controller = self.controller
        entries = [(mode, key, action) for mode, table in enumerate(
            (controller.frozen_policy, controller.frozen_win_policy)) for key, action in table.items()]
        records = (Policy * len(entries))()
        for index, (mode, key, action) in enumerate(entries):
            packed = self.pack_key(key, mode)
            records[index] = Policy(packed, self.pack_action(packed.phase, action), mode)
        self.check(self.library.countdown_policies(
            self.handle, records, len(records), controller.win_target or 0.0, controller.win_target is not None))

    def sync_training(self):
        controller = self.controller
        count = ct.c_size_t()
        records = self.library.countdown_records(self.handle, ct.byref(count))
        self.check(records)
        for index in range(count.value):
            record = records[index]
            key = self.unpack_key(record.key, record.mode)
            stats = (controller.win_q if record.mode else controller.q)[key]
            for field in STAT_FIELDS:
                setattr(stats, field, getattr(record.stats, field))
        visits = self.library.countdown_visits(self.handle, ct.byref(count))
        for index in range(count.value):
            visit = visits[index]
            context = self.unpack_context(visit.context)
            entry = controller._sampled_contexts.setdefault(controller._win_context_key(context), [context, 0])
            entry[1] += visit.count

    def run(self, context, actions, budget, seed, kind, target, progress, cancelled, label):
        if not -(2 ** 63) <= seed < 2 ** 63 or not 0 < budget < 2 ** 64:
            raise OverflowError("Rust 采样种子或预算超出整数范围")
        self.sync_policies()
        packed = self.pack_context(context)
        config = self.controller.config
        job = Job(packed, seed, budget, config.epsilon_start, config.epsilon_end,
                  target or 0.0, (ct.c_uint8 * 64)(*(self.pack_action(packed.phase, action) for action in actions)),
                  len(actions), kind, target is not None)
        result = Result()
        total = budget * len(actions) if kind in (2, 3) else budget
        last_progress = time.perf_counter()
        callback_error = None

        @Poll
        def poll(completed):
            nonlocal last_progress, callback_error
            try:
                if cancelled and cancelled():
                    return 1
                if progress:
                    now = time.perf_counter()
                    if completed and now - last_progress >= 0.1:
                        progress(completed, total, label)
                        last_progress = now
                return 0
            except BaseException as error:
                callback_error = error
                return 1

        self.running = True
        try:
            success = self.library.countdown_run(self.handle, ct.byref(job), ct.byref(result), poll)
        finally:
            self.running = False
        if kind < 2:
            self.sync_training()
        else:
            self.controller.total_evaluation_trials += result.trials
        if callback_error is not None:
            # 已完成训练必须计入种子序列，不能因 UI 回调异常而重放。
            if kind < 2:
                self.controller.total_control_rollouts += result.completed
            raise callback_error
        try:
            self.check(success)
            if progress and result.completed == total:
                progress(result.completed, total, label)
        except BaseException:
            if kind < 2:
                self.controller.total_control_rollouts += result.completed
            raise
        return result

    @staticmethod
    def stats(value):
        from tool.countdown_evaluator import MCSampleStats

        return MCSampleStats(**{field: getattr(value, field) for field in STAT_FIELDS})
