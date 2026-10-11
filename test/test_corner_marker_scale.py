import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from tool.utils.analysis_map import detect_corner_markers
from tool.utils.image_tool import find_image_in_folder, load_all_images_from_directory

ROOT = Path(__file__).resolve().parents[1]
load_all_images_from_directory(str(ROOT / "resource" / "imgs"))

# 「进入位面」总览画面（mode=1）比「选择」画面（mode=2/3）的地图渲染小约 1.2 倍
OVERVIEW_SCALE = 1 / 1.2


def make_map(marker_name, scale):
    """合成一张以给定尺度绘制角标的地图截图，返回 (image, 该角标所在节点的匹配结果)。

    节点框取角标周围一圈，保证角标中心与节点中心距离在 max_dist 之内。
    """
    marker = find_image_in_folder('', marker_name)
    marker = cv.resize(marker, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    height, width = marker.shape[:2]
    x, y = 120, 150

    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[y:y + height, x:x + width] = marker

    node = {'name': 'battle', 'location': (x - 20, y - 20), 'size': (width + 40, height + 40),
            'similarity': 1.0}
    return image, node


class CornerMarkerScaleTest(unittest.TestCase):
    def test_overview_screen_detects_scaled_marker(self):
        image, node = make_map('pig1', OVERVIEW_SCALE)

        results = detect_corner_markers(image, [node], mode=1)

        self.assertEqual([r['name'] for r in results], ['pig1'])
        self.assertGreater(results[0]['similarity'], 0.9)

    def test_select_screen_detects_native_marker(self):
        image, node = make_map('pig1', 1.0)

        results = detect_corner_markers(image, [node], mode=2)

        self.assertEqual([r['name'] for r in results], ['pig1'])
        self.assertGreater(results[0]['similarity'], 0.9)

    def test_select_mode_ignores_overview_scaled_marker(self):
        """「进入位面」尺度的角标不应被「选择」画面的尺度匹配到。"""
        image, node = make_map('pig1', OVERVIEW_SCALE)

        results = detect_corner_markers(image, [node], mode=2)

        self.assertEqual(results, [])

    def test_mode_three_uses_select_screen_scale(self):
        image, node = make_map('reinforce1', 1.0)

        results = detect_corner_markers(image, [node], mode=3)

        self.assertEqual([r['name'] for r in results], ['reinforce1'])
        self.assertGreater(results[0]['similarity'], 0.9)

    def test_default_mode_keeps_native_scale(self):
        """不传 mode 时沿用修复前的原生尺度匹配。"""
        image, node = make_map('pig1', 1.0)

        results = detect_corner_markers(image, [node])

        self.assertEqual([r['name'] for r in results], ['pig1'])
        self.assertGreater(results[0]['similarity'], 0.9)


if __name__ == "__main__":
    unittest.main()
