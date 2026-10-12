import importlib
import unittest
from pathlib import Path

import numpy as np

from tool.utils.image_tool import find_image_in_folder, load_all_images_from_directory

ROOT = Path(__file__).resolve().parents[1]
load_all_images_from_directory(str(ROOT / "resource" / "imgs"))

analysis_map = importlib.import_module("tool.utils.analysis_map")


def tint_bright(template, target):
    """把模板亮部（图标本体）换成蓝/金色、形状保持不变，用来复现「金模板在蓝信标上得分更高」。"""
    patch = template.copy()
    lum = template.max(axis=2).astype(np.float32)
    mask = lum > 100
    if target == "blue":
        colored = np.stack([lum, lum * 0.72, lum * 0.42], axis=2)
    else:
        colored = np.stack([lum * 0.42, lum * 0.72, lum], axis=2)
    patch[mask] = colored[mask].astype(np.uint8)
    return patch


def paste(canvas, template, x, y):
    canvas[y:y + template.shape[0], x:x + template.shape[1]] = template


class IsBlessingPigTest(unittest.TestCase):
    """按颜色区分祝福扑满（蓝）与奇物扑满（金）。"""

    def test_blue_marker_is_blessing(self):
        patch = tint_bright(find_image_in_folder("", "pig1"), "blue")
        canvas = np.zeros((60, 60, 3), np.uint8)
        paste(canvas, patch, 10, 10)
        marker = {"name": "pig1", "location": (10, 10), "size": (patch.shape[1], patch.shape[0]), "similarity": 1.0}
        self.assertTrue(analysis_map.is_blessing_pig(canvas, marker))

    def test_gold_marker_is_not_blessing(self):
        patch = tint_bright(find_image_in_folder("", "pig3"), "gold")
        canvas = np.zeros((60, 60, 3), np.uint8)
        paste(canvas, patch, 10, 10)
        marker = {"name": "pig3", "location": (10, 10), "size": (patch.shape[1], patch.shape[0]), "similarity": 1.0}
        self.assertFalse(analysis_map.is_blessing_pig(canvas, marker))

    def test_marker_out_of_frame_has_no_color(self):
        canvas = np.zeros((60, 60, 3), np.uint8)
        marker = {"name": "pig1", "location": (200, 200), "size": (20, 17), "similarity": 1.0}
        self.assertIsNone(analysis_map.is_blessing_pig(canvas, marker))


class PigMarkerColorTest(unittest.TestCase):
    """角标模板只看形状，扑满信标的 name 必须按实际颜色归一化为 pig1（祝福）/ pig3（奇物）。"""

    def test_pig_marker_name_is_corrected_by_color(self):
        # 蓝色信标故意用 pig3 的形状、金色信标故意用 pig1 的形状，模板匹配会把两者判反
        blue = tint_bright(find_image_in_folder("", "pig3"), "blue")
        gold = tint_bright(find_image_in_folder("", "pig1"), "gold")
        canvas = np.zeros((200, 400, 3), np.uint8)
        paste(canvas, blue, 60, 60)
        paste(canvas, gold, 240, 60)
        matches = [
            {"name": "battle", "location": (60, 60), "size": (blue.shape[1], blue.shape[0]), "similarity": 0.9},
            {"name": "elite", "location": (240, 60), "size": (gold.shape[1], gold.shape[0]), "similarity": 0.9},
        ]

        corners = analysis_map.detect_corner_markers(canvas, matches, mode=2)

        self.assertEqual(2, len(corners))
        self.assertEqual("pig1", matches[0]["corner_marker"]["name"])
        self.assertEqual("pig3", matches[1]["corner_marker"]["name"])


if __name__ == "__main__":
    unittest.main()
