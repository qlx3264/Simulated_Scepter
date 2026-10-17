"""真实 Rust 库与 Python 参考实现的同种子差分验证；运行前构建内核。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

from test.test_countdown_backend import high_target_map, linear_map, rational_path_map
from tool import countdown_evaluator as backend
from tool.countdown_native import load_library


class PythonController(backend.MonteCarloController):
    """保留扩展控制器的 Python 执行路径，作为独立对照。"""


class NativeCountdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if load_library() is None:
            raise RuntimeError("请先运行 uv run python scripts/build_countdown_native.py")

    def pair(self, model, seed=20260802, budget=300):
        config = backend.MCConfig(control_rollouts=budget, evaluation_rollouts=budget, min_visits=2, seed=seed)
        return backend.MonteCarloController(model, config), PythonController(model, config)

    def assert_training_equal(self, native, reference):
        self.assertEqual(native.sampling_backend, "rust")
        self.assertEqual(reference.sampling_backend, "python")
        self.assertEqual(native.q, reference.q)
        self.assertEqual(native.win_q, reference.win_q)
        self.assertEqual(list(native._sampled_contexts.items()), list(reference._sampled_contexts.items()))
        self.assertEqual(native.total_control_rollouts, reference.total_control_rollouts)

    def test_training_all_effects_and_integer_seeds(self):
        model = high_target_map()
        for seed in (0, -31, 8_000_000_001):
            for effect in backend.ALL_EFFECTS:
                with self.subTest(seed=seed, effect=effect):
                    native, reference = self.pair(model, seed)
                    context = backend.DecisionContext("effect", model.initial_state(9, 3, 40), effect)
                    for controller in (native, reference):
                        controller.win_target = 80
                        controller._refine_with_budget(context, 300, train_win=True)
                    self.assert_training_equal(native, reference)
                    for win in (False, True):
                        self.assertEqual(native.evaluate_actions(context, 80, win_policy=win),
                                         reference.evaluate_actions(context, 80, win_policy=win))

    def test_full_recommendations_and_target_change(self):
        for model in (linear_map((1,)), rational_path_map(), high_target_map()):
            native, reference = self.pair(model, budget=2000)
            for effect, target in ((6, 80), (5, 75), (3, None)):
                with self.subTest(nodes=len(model.nodes), effect=effect, target=target):
                    context = backend.DecisionContext("effect", model.initial_state(9, 2, 40), effect)
                    self.assertEqual(native.recommend(context, target), reference.recommend(context, target))
                    self.assert_training_equal(native, reference)
                    self.assertEqual(native.frozen_policy, reference.frozen_policy)
                    self.assertEqual(native.frozen_win_policy, reference.frozen_win_policy)
                    self.assertEqual(native._frozen_win_hits, reference._frozen_win_hits)
                    self.assertEqual(native.total_evaluation_trials, reference.total_evaluation_trials)

    def test_sparse_node_ids_large_and_negative_cd(self):
        original = rational_path_map()
        ids = {node: node * 17 + 100 for node in original.node_map}
        nodes = [dict(node, idx=ids[node["idx"]]) for node in original.nodes]
        edges = {ids[node]: [ids[next_node] for next_node in values] for node, values in original.edges.items()}
        model = backend.CountdownMap(nodes, edges, ids[original.start_idx], (ids[3],))
        for cd in (-100_000, 100_000):
            with self.subTest(cd=cd):
                native, reference = self.pair(model)
                context = backend.DecisionContext("effect", model.initial_state(300, 260, cd), 5)
                for controller in (native, reference):
                    controller.win_target = cd + 10
                    controller._refine_with_budget(context, 300, train_win=True)
                self.assert_training_equal(native, reference)
                self.assertEqual(native.evaluate_actions(context, cd + 10, win_policy=True),
                                 reference.evaluate_actions(context, cd + 10, win_policy=True))

    def test_unobserved_first_effect_policy(self):
        model = high_target_map()
        native, reference = self.pair(model)
        actions = {effect: "keep" for effect in backend.ALL_EFFECTS}
        actions[6] = ("cheat", 3)
        state = model.initial_state(4, 2, 60)
        self.assertEqual(native.evaluate_unobserved_effect_policy(state, actions, 1001, 80),
                         reference.evaluate_unobserved_effect_policy(state, actions, 1001, 80))
        self.assertEqual(native.total_evaluation_trials, reference.total_evaluation_trials)

    @staticmethod
    def cancel_after(limit):
        checks = 0

        def cancelled():
            nonlocal checks
            checks += 1
            return checks > limit

        return cancelled

    def test_cancel_and_resume_counts_and_statistics(self):
        model = high_target_map()
        native, reference = self.pair(model)
        context = backend.DecisionContext("effect", model.initial_state(5, 2, 50), 5)
        for controller in (native, reference):
            controller.refine_current_state(context, cancelled=self.cancel_after(13))
        self.assert_training_equal(native, reference)
        for controller in (native, reference):
            controller.refine_current_state(context)
        self.assert_training_equal(native, reference)
        self.assertEqual(native.evaluate_actions(context, 80, cancelled=self.cancel_after(13)),
                         reference.evaluate_actions(context, 80, cancelled=self.cancel_after(13)))
        self.assertEqual(native.total_evaluation_trials, reference.total_evaluation_trials)
        self.assertEqual(native.evaluate_actions(context, 80), reference.evaluate_actions(context, 80))
        self.assertIsNone(native._native_session)

    def test_callback_exception_releases_handle_and_keeps_completed_training(self):
        model = linear_map()
        native, reference = self.pair(model)
        context = backend.DecisionContext("effect", model.initial_state(2, 1, 50), 5)
        checks = 0

        def fail():
            nonlocal checks
            checks += 1
            if checks > 3:
                raise ValueError("cancel callback failed")
            return False

        with self.assertRaisesRegex(ValueError, "cancel callback failed"):
            native.refine_current_state(context, cancelled=fail)
        reference.refine_current_state(context, cancelled=self.cancel_after(3))
        self.assert_training_equal(native, reference)
        self.assertIsNone(native._native_session)

    def test_progress_exception_after_mean_batch_keeps_all_completed_paths(self):
        model = linear_map()
        native, reference = self.pair(model, budget=30)
        context = backend.DecisionContext("effect", model.initial_state(2, 1, 50), 5)
        checks = 0

        def fail_during_win_batch():
            nonlocal checks
            checks += 1
            if checks > 18:
                raise ValueError("second batch failed")
            return False

        native.win_target = reference.win_target = 52
        with self.assertRaisesRegex(ValueError, "second batch failed"):
            native._refine_with_budget(context, 30, train_win=True, cancelled=fail_during_win_batch)
        reference._refine_with_budget(context, 30, train_win=True, cancelled=self.cancel_after(18))
        self.assert_training_equal(native, reference)
        self.assertEqual(native.total_control_rollouts, 18)
        for controller in (native, reference):
            controller._refine_with_budget(context, 30, train_win=True)
        self.assert_training_equal(native, reference)

        def fail_on_completed_batch(done, total, label):
            if done == total:
                raise ValueError("progress failed")

        with self.assertRaisesRegex(ValueError, "progress failed"):
            native.refine_current_state(context, progress=fail_on_completed_batch)
        reference.refine_current_state(context)
        self.assert_training_equal(native, reference)

    def test_arbitrary_precision_state_or_seed_never_truncates(self):
        model = linear_map()
        context = backend.DecisionContext("effect", model.initial_state(2, 1, 50), 5)
        for seed, cd in ((2 ** 80, 50), (1, 2 ** 80)):
            native, reference = self.pair(model, seed=seed, budget=20)
            large_context = replace(context, state=replace(context.state, countdown=cd))
            self.assertEqual(native.evaluate_actions(large_context, 52), reference.evaluate_actions(large_context, 52))
            self.assertEqual(native.sampling_backend, "python")
        native, reference = self.pair(model)
        # 既有胜率 Q 表含超范围 CD，即使当前 CD 正常，也不能悄悄窄化导入。
        large_context = replace(context, state=replace(context.state, countdown=2 ** 80))
        for controller in (native, reference):
            controller.win_q[controller._win_q_key(large_context, "keep")].append(50, 52)
        self.assertEqual(native.evaluate_actions(context, 52), reference.evaluate_actions(context, 52))
        self.assertEqual(native.sampling_backend, "python")

    def test_sampling_callback_cannot_reenter_the_native_handle(self):
        model = linear_map()
        native, _reference = self.pair(model)
        context = backend.DecisionContext("effect", model.initial_state(2, 1, 50), 5)

        def reenter():
            native.evaluate_actions(context)
            return False

        with self.assertRaisesRegex(RuntimeError, "重入"):
            native.refine_current_state(context, cancelled=reenter)
        self.assertEqual(native.total_control_rollouts, 0)
        self.assertIsNone(native._native_session)
        self.assertEqual(native.refine_current_state(context), 300)

    def test_missing_library_and_oversized_map_use_python(self):
        model = linear_map()
        native, reference = self.pair(model)
        context = backend.DecisionContext("effect", model.initial_state(2, 1, 50), 5)
        with patch("tool.countdown_native.load_library", return_value=None):
            self.assertEqual(native.recommend(context, 52), reference.recommend(context, 52))
            self.assertEqual(native.sampling_backend, "python")
        nodes = [{"idx": index, "cx": index * 100, "cy": 0} for index in range(65)]
        model = backend.CountdownMap(nodes, {i: [i + 1] if i < 64 else [] for i in range(65)}, 0, ())
        native, reference = self.pair(model, budget=20)
        context = replace(context, state=model.initial_state(2, 0, 50))
        self.assertEqual(native.evaluate_actions(context, 52), reference.evaluate_actions(context, 52))
        self.assertEqual(native.sampling_backend, "python")


if __name__ == "__main__":
    unittest.main()
