"""完整 Rust 评估会话；仅交换输入、进度和最终统计，按需导出诊断快照。"""

from __future__ import annotations

import ctypes as ct
import inspect
import sys
import time
from functools import lru_cache, wraps
from pathlib import Path
from threading import get_ident
from weakref import finalize, ref


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


class Request(ct.Structure):
    _fields_ = [("initial", Context), ("seed", ct.c_int64),
                ("control", ct.c_uint64), ("evaluation", ct.c_uint64), ("min_visits", ct.c_uint64),
                ("epsilon_start", ct.c_double), ("epsilon_end", ct.c_double),
                ("target", ct.c_double), ("learning_target", ct.c_double),
                ("has_target", ct.c_uint32), ("has_learning_target", ct.c_uint32),
                ("mode", ct.c_uint32), ("win", ct.c_uint32), ("train_win", ct.c_uint32),
                ("first_actions", ct.c_uint8 * 6)]


class Report(ct.Structure):
    _fields_ = [("mean", Stats * 64), ("wins", Stats * 64), ("actions", ct.c_uint8 * 64),
                ("count", ct.c_uint32), ("recommended", ct.c_uint32), ("highest_win", ct.c_uint32),
                ("has_wins", ct.c_uint32), ("control", ct.c_uint64), ("evaluation", ct.c_uint64),
                ("total_control", ct.c_uint64), ("total_trials", ct.c_uint64),
                ("q_states", ct.c_uint64), ("cancelled", ct.c_uint32)]


Poll = ct.CFUNCTYPE(ct.c_int, ct.c_uint64, ct.c_uint64, ct.c_uint32)
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
    library.countdown_abi.restype = ct.c_uint32
    if library.countdown_abi() != 2:
        raise RuntimeError("倒计时 Rust 内核 ABI 不匹配，请重新构建并打包")
    signatures = {
        "abi": ([], ct.c_uint32),
        "layout": ([ct.c_uint32], ct.c_size_t),
        "error": ([], ct.c_char_p),
        "create": ([ct.POINTER(Node), ct.c_size_t, ct.c_uint64], ct.c_void_p),
        "destroy": ([ct.c_void_p], None),
        "load": ([ct.c_void_p, ct.POINTER(Record), ct.c_size_t], ct.c_int),
        "initialize": ([ct.c_void_p, ct.POINTER(Policy), ct.c_size_t, ct.c_uint64, ct.c_uint64,
                        ct.c_double, ct.c_uint32], ct.c_int),
        "execute": ([ct.c_void_p, ct.POINTER(Request), ct.POINTER(Report), Poll], ct.c_int),
        "choose": ([ct.c_void_p, ct.POINTER(Context), ct.c_uint32], ct.c_int),
        "records": ([ct.c_void_p, ct.POINTER(ct.c_size_t)], ct.POINTER(Record)),
        "snapshot_policies": ([ct.c_void_p, ct.POINTER(ct.c_size_t)], ct.POINTER(Policy)),
        "visits": ([ct.c_void_p, ct.POINTER(ct.c_size_t)], ct.POINTER(Visit)),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, "countdown_" + name)
        function.argtypes, function.restype = arguments, result
    layouts = (Context, Node, Stats, Record, Policy, Visit, Request, Report)
    if (any(
            library.countdown_layout(index) != ct.sizeof(layout)
            for index, layout in enumerate(layouts))):
        raise RuntimeError("倒计时 Rust 内核 ABI 不匹配，请重新运行 scripts/build_countdown_native.py")
    return library


def sampling_session(method):
    """把完整操作交给原生会话；Python 方法体仅用于参考/扩展控制器。"""
    signature = inspect.signature(method)
    @wraps(method)
    def wrapped(controller, *args, **kwargs):
        if getattr(controller, "_native_scope", False):
            if (controller._native_owner != get_ident()
                    or getattr(controller._native_session, "running", False)):
                raise RuntimeError("同一倒计时控制器不能并发采样或在采样回调中重入")
            return method(controller, *args, **kwargs)
        controller._native_scope = True
        controller._native_owner = get_ident()
        try:
            arguments = signature.bind(controller, *args, **kwargs)
            arguments.apply_defaults()
            values = dict(arguments.arguments)
            values.pop("self")
            context = values.get("context")
            if context is None:
                from tool.countdown_evaluator import DecisionContext
                context = DecisionContext("effect", values["state"], 1)
            session = get_session(controller, context)
            if session:
                return session.execute(method.__name__, values, context)
            return method(controller, *args, **kwargs)
        finally:
            controller._native_scope = False
    return wrapped


def get_session(controller, context):
    from tool.countdown_evaluator import MonteCarloController

    session = getattr(controller, "_native_session", None)
    # 子类/实例覆盖继续使用 Python，避免绕过调用者扩展的规则。
    supported = (type(controller) is MonteCarloController
                 and not any(name in controller.__dict__ for name in
                             ("_rollout", "_advance", "_greedy_action", "_greedy_win_action",
                              "evaluate_actions", "_refine_with_budget", "_calibrate_successors",
                              "evaluate_current_policy", "_sample_successor_contexts"))
                 and len(controller.map.nodes) <= 64)
    state = context.state
    full_mask = sum(1 << node for node in controller.map.node_map)
    supported = (supported and state.infected >= 0 and not state.infected & ~full_mask
                 and abs(state.countdown) < 2 ** 52 - 4096
                 and 0 <= state.cheat_rem < 2 ** 32 and 0 <= state.reroll_rem < 2 ** 32
                 and abs(controller.config.seed) < 2 ** 62)
    if session and supported:
        return session
    if session:
        session.snapshot()
        session.close()
        controller._native_session = None
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
        self.owner = ref(controller)
        self.library = library
        self.handle = None
        self.running = False
        self.q_states = len(controller.q)
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
        self.release = finalize(self, library.countdown_destroy, self.handle)
        try:
            for mode, table in enumerate((controller.q, controller.win_q)):
                records = (Record * len(table))()
                for index, (key, stats) in enumerate(table.items()):
                    records[index] = Record(self.pack_key(key, mode, True),
                                            Stats(*(getattr(stats, field) for field in STAT_FIELDS)), mode)
                self.check(library.countdown_load(self.handle, records, len(records)))
            policies = []
            for mode, table in enumerate((controller.frozen_policy, controller.frozen_win_policy)):
                for key, action in table.items():
                    packed = self.pack_key(key, mode)
                    flags = mode | (2 if mode and key in controller._frozen_win_hits else 0)
                    policies.append(Policy(packed, self.pack_action(packed.phase, action), flags))
            records = (Policy * len(policies))(*policies)
            self.check(library.countdown_initialize(self.handle, records, len(records),
                controller.total_control_rollouts, controller.total_evaluation_trials,
                controller.win_target or 0.0, controller.win_target is not None))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.handle:
            self.release()
            self.handle = None

    @property
    def controller(self):
        return self.owner()

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

    def snapshot(self):
        """显式诊断/回退快照。正常推荐不调用，也不导出中间状态。"""
        if self.running:
            raise RuntimeError("不能在采样中导出状态")
        controller = self.controller
        controller.q.clear()
        controller.win_q.clear()
        controller.frozen_policy.clear()
        controller.frozen_win_policy.clear()
        controller._frozen_win_hits.clear()
        controller._sampled_contexts.clear()
        count = ct.c_size_t()
        records = self.library.countdown_records(self.handle, ct.byref(count))
        self.check(records)
        for index in range(count.value):
            record = records[index]
            key = self.unpack_key(record.key, record.mode)
            (controller.win_q if record.mode else controller.q)[key] = self.stats(record.stats)
        policies = self.library.countdown_snapshot_policies(self.handle, ct.byref(count))
        self.check(policies)
        for index in range(count.value):
            policy = policies[index]
            win = policy.mode & 1
            key = self.unpack_key(policy.key, win)[:-1]
            (controller.frozen_win_policy if win else controller.frozen_policy)[key] = (
                self.unpack_action(policy.key.phase, policy.action))
            if policy.mode & 2:
                controller._frozen_win_hits.add(key)
        visits = self.library.countdown_visits(self.handle, ct.byref(count))
        for index in range(count.value):
            visit = visits[index]
            context = self.unpack_context(visit.context)
            controller._sampled_contexts[controller._win_context_key(context)] = [context, visit.count]

    def choose(self, context, win=False):
        if self.running:
            raise RuntimeError("同一倒计时控制器不能在采样回调中重入")
        packed = self.pack_context(context)
        action = self.library.countdown_choose(self.handle, ct.byref(packed), win)
        return None if action < 0 else self.unpack_action(packed.phase, action)

    def execute(self, method, values, context):
        from tool.countdown_evaluator import MCRecommendation

        controller = self.controller
        config = controller.config
        target = values.get("target")
        control, evaluation = config.control_rollouts, config.evaluation_rollouts
        modes = {"recommend": 0, "_refine_with_budget": 1, "evaluate_actions": 2,
                 "evaluate_current_policy": 3, "evaluate_unobserved_effect_policy": 4}
        mode = modes[method]
        first_actions = (ct.c_uint8 * 6)()
        if mode == 1:
            control = max(1, int(values["total"]))
        elif mode in (2, 3):
            budget = values.get("rollouts" if mode == 2 else "evaluation_rollouts")
            if budget is not None:
                evaluation = max(1, int(budget))
        elif mode == 4:
            from dataclasses import replace
            evaluation = max(1, int(values["rollouts"]))
            for effect in range(1, 7):
                action = values["first_actions"][effect]
                if action not in controller.legal_actions(replace(context, observed_effect=effect)):
                    raise ValueError(f"{action!r} 不是效果 {effect} 的合法动作")
                first_actions[effect - 1] = self.pack_action(1, action)
        if not (0 < control < 2 ** 63 and 0 < evaluation < 2 ** 63
                and 0 < config.min_visits < 2 ** 56):
            raise OverflowError("采样预算超出 Rust 整数范围")
        packed = self.pack_context(context)
        request = Request(packed, config.seed, control, evaluation, config.min_visits,
                          config.epsilon_start, config.epsilon_end, target or 0.0,
                          controller.win_target or 0.0, target is not None,
                          controller.win_target is not None, mode,
                          values.get("win_policy", False), values.get("train_win", False), first_actions)
        report = Report()
        progress, cancelled = values.get("progress"), values.get("cancelled")
        last_progress = time.perf_counter()
        callback_error = None

        @Poll
        def poll(done, total, stage):
            nonlocal last_progress, callback_error
            if callback_error is not None:
                return 1
            try:
                if cancelled and cancelled():
                    return 1
                now = time.perf_counter()
                if progress and ((stage == 0 and done == total) or now - last_progress >= 0.1):
                    label = ("当前状态控制采样", "冻结策略独立评价", "后继策略校准")[stage]
                    progress(done, total, values.get("label", label))
                    last_progress = now
                return 0
            except BaseException as error:
                callback_error = error
                return 1

        self.running = True
        try:
            success = self.library.countdown_execute(
                self.handle, ct.byref(request), ct.byref(report), poll)
        finally:
            self.running = False
        controller.total_control_rollouts = report.total_control
        controller.total_evaluation_trials = report.total_trials
        self.q_states = report.q_states
        if mode == 0:
            controller.win_target = target
        if callback_error is not None:
            raise callback_error
        self.check(success)
        if mode == 1:
            return report.control
        if mode == 4:
            return self.stats(report.mean[0])
        actions = [self.unpack_action(packed.phase, report.actions[i]) for i in range(report.count)]
        reports = {action: self.stats(report.mean[i]) for i, action in enumerate(actions)}
        if mode == 2:
            return reports, report.evaluation
        if report.cancelled:
            raise InterruptedError("蒙特卡洛推荐已取消")
        if not report.count:
            raise RuntimeError("当前状态没有完成任何候选评价")
        win_reports = ({action: self.stats(report.wins[i]) for i, action in enumerate(actions)}
                       if report.has_wins else None)
        return MCRecommendation(
            context, reports, self.unpack_action(packed.phase, report.recommended),
            self.unpack_action(packed.phase, report.highest_win),
            report.control if mode == 0 else int(values.get("control_rollouts", 0)),
            report.evaluation, win_reports)

    @staticmethod
    def stats(value):
        from tool.countdown_evaluator import MCSampleStats
        return MCSampleStats(**{field: getattr(value, field) for field in STAT_FIELDS})
