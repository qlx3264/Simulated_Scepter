"""在固定 26 节点地图上比较完整推荐和完整路径采样；不启动游戏或 GUI。"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test.test_countdown_backend import high_target_map
from tool import countdown_evaluator as backend
from tool.countdown_native import load_library


class PythonController(backend.MonteCarloController):
    """使用同一控制器的 Python 参考采样循环。"""


def benchmark(budget, path_samples):
    if load_library() is None:
        raise RuntimeError("请先运行 uv run python scripts/build_countdown_native.py")
    model = high_target_map()
    config = backend.MCConfig(control_rollouts=budget, evaluation_rollouts=budget,
                              min_visits=2, seed=20260802)
    state = model.initial_state(9, 3, 40)
    context = backend.DecisionContext("effect", state, 6)
    results, recommendations, path_stats = {}, [], []
    for name, controller_type in (("python", PythonController), ("rust", backend.MonteCarloController)):
        controller = controller_type(model, config)
        start = perf_counter()
        recommendation = controller.recommend(context, 80)
        recommendation_seconds = perf_counter() - start
        first_actions = {effect: controller._greedy_action(
            backend.DecisionContext("effect", state, effect)) for effect in backend.ALL_EFFECTS}
        start = perf_counter()
        stats = controller.evaluate_unobserved_effect_policy(state, first_actions, path_samples, 80)
        path_seconds = perf_counter() - start
        if controller.sampling_backend != name:
            raise AssertionError(f"请求 {name}，实际执行 {controller.sampling_backend}")
        results[name] = {
            "recommendation_seconds": recommendation_seconds,
            "actual_control_paths": recommendation.control_rollouts,
            "actual_evaluation_paths": recommendation.evaluation_rollouts,
            "frozen_policy_seconds": path_seconds,
            "frozen_policy_paths": stats.count,
            "frozen_policy_stats": asdict(stats),
            "recommended_action": recommendation.recommended_action,
            "highest_win_action": recommendation.highest_win_action,
            "root_win_reports": {str(action): asdict(report)
                                 for action, report in recommendation.win_reports.items()},
        }
        recommendations.append(recommendation)
        path_stats.append(stats)
        print(f"{name}: recommend={recommendation_seconds:.6f}s, "
              f"{stats.count} full paths={path_seconds:.6f}s", flush=True)
    if recommendations[0] != recommendations[1] or path_stats[0] != path_stats[1]:
        raise AssertionError("Python 与 Rust 结果不一致")
    results["identical_results"] = True
    results["scenario"] = {"nodes": len(model.nodes), "seed": config.seed,
                           "initial_cd": 40, "cheat": 9, "reroll": 3,
                           "observed_effect": 6, "target": 80,
                           "control_budget": budget, "evaluation_budget": budget}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--path-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.budget <= 0 or args.path_samples <= 0:
        parser.error("采样预算必须大于 0")
    results = benchmark(args.budget, args.path_samples)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Python/Rust recommendations and sample statistics are identical.")


if __name__ == "__main__":
    main()
