"""货币战争备战栏识别、模板采集及补位规则，不操作真实键鼠。"""

import hashlib
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.currency.settings import load_currency_settings, save_currency_settings
from tool.log import CUS_LOGGER
from tool.storage import resource_path
from tool.utils.image_tool import find_image_by_name

# 坐标均为 1920×1080 游戏截图中的 [左, 右, 上, 下]，备战栏从左到右编号。
BENCH_BOXES = tuple((383 + 125 * i, 493 + 125 * i, 843, 977) for i in range(9))
# 仅保存头像中心的 40×40 特征，不包含边框、角标和星级。
PORTRAIT_BOX = (35, 75, 47, 87)
AREA_BOXES = {
    "front": ((676, 807, 328, 465), (821, 948, 328, 465),
              (968, 1095, 328, 465), (1109, 1240, 328, 465)),
    "back": ((533, 673, 598, 737), (678, 812, 598, 737),
             (822, 950, 598, 737), (966, 1095, 598, 737),
             (1105, 1238, 598, 737), (1244, 1384, 598, 737)),
}
# 画面上的人数是前台与后台之和，上限随对局进度变化，因此不能按前台落点数校验。
TEAM_CAPACITY = len(AREA_BOXES["front"]) + len(AREA_BOXES["back"])
# 截帧时整栏会和参考图错开几像素，匹配时在四周留出搜索余量。
MATCH_MARGIN = 4
# 缩小后的比对尺度：抵消个别帧里角色画的细微缩放，避免同一角色被采集两份模板。
MATCH_SCALE = 0.4
# 同一角色的比对分数；不同角色实测不超过 0.65，同一角色位移或缩放后仍在 0.95 以上。
MATCH_THRESHOLD = 0.9


@dataclass(frozen=True)
class BenchSlot:
    """一次截图中的备战格；template 是可用于优先级配置的模板文件名（无后缀）。"""

    index: int
    template: str
    kind: str  # character、item 或 empty。


@dataclass(frozen=True)
class Deployment:
    """从备战栏 source 格拖到 area 区域的 target 格，编号均从 0 开始。"""

    source: int
    area: str
    target: int
    outgoing: str | None = None  # None=补空位，空字符串=尚未识别的上阵角色，其余为被替换角色。


def match_score(search, template, method=cv2.TM_CCOEFF_NORMED):
    """在 search 内滑动 template，返回最相似分数（越大越相似）。"""
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return 0.0
    scores = cv2.matchTemplate(search, template, method)
    if method == cv2.TM_SQDIFF_NORMED:
        return 1.0 - float(scores.min())
    return float(scores.max())


def match_feature(patch):
    """缩小特征，让细微缩放和单像素抖动不再落到另一个模板摘要上。"""
    return cv2.resize(
        patch, None, fx=MATCH_SCALE, fy=MATCH_SCALE, interpolation=cv2.INTER_AREA
    )


def portrait_score(search, template, method=cv2.TM_CCOEFF_NORMED):
    """先按原尺寸对齐，再比对缩小后的特征。

    原尺寸定位像素位移，缩小后的特征抵消角色画在个别帧里的细微缩放，取两者较高分。
    """
    scores = cv2.matchTemplate(search, template, method)
    if method == cv2.TM_SQDIFF_NORMED:
        _, _, loc, _ = cv2.minMaxLoc(scores)
        best = 1.0 - float(scores.min())
    else:
        _, best, _, loc = cv2.minMaxLoc(scores)
    x, y = loc
    aligned = search[y:y + template.shape[0], x:x + template.shape[1]]
    return max(float(best), match_score(match_feature(aligned), match_feature(template), method))


def save_template(path, patch):
    """原子写入特征模板，编码或写入失败时保留原文件。"""
    success, encoded = cv2.imencode(".png", patch)
    if not success:
        raise OSError(f"备战模板编码失败：{path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".portrait-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded.tobytes())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class BattlePreparation:
    def __init__(self, catalog=None):
        self.catalog = Path(catalog) if catalog is not None else resource_path(
            __file__, "templates/bench"
        )
        self.catalog.mkdir(parents=True, exist_ok=True)
        self.templates = {}
        for path in sorted(self.catalog.glob("*.png")):
            if path.stem.split("_", 1)[0] not in ("character", "item", "empty"):
                raise ValueError(f"无法识别备战模板类型：{path}")
            tile = find_image_by_name(str(path.resolve()))
            if tile is None:
                raise ValueError(f"无法读取备战模板：{path}")
            if tile.shape == (134, 110, 3):
                left, right, top, bottom = PORTRAIT_BOX
                tile = tile[top:bottom, left:right].copy()
                if path.stem.startswith("character_"):
                    save_template(path, tile)  # 保留文件名，已有优先级继续匹配同一角色。
            if tile.shape != (40, 40, 3):
                raise ValueError(f"备战模板应为 40×40 的彩色特征图：{path}")
            self.templates[path.stem] = tile
        self.badges = []
        for name in ("back", "both"):
            badge = find_image_by_name(str(resource_path(__file__, f"templates/badges/{name}.png")))
            if badge is None:
                raise FileNotFoundError(f"缺少货币战争角色标识模板：{name}")
            self.badges.append(badge)
        self.area_tiles = {}
        for area, boxes in AREA_BOXES.items():
            self.area_tiles[area] = []
            for index in range(len(boxes)):
                path = resource_path(__file__, f"templates/areas/{area}_{index}.png")
                tile = find_image_by_name(str(path))
                if tile is None:
                    raise FileNotFoundError(f"缺少货币战争空位模板：{path}")
                self.area_tiles[area].append(tile)
        self.merge_duplicates()
        self.reset_members()

    def same_character(self, first, second):
        """判断两份特征是否来自同一角色；允许几像素位移和细微缩放。"""
        margin = MATCH_MARGIN
        search = cv2.copyMakeBorder(
            self.templates[first], margin, margin, margin, margin, cv2.BORDER_REPLICATE,
        )
        method = cv2.TM_SQDIFF_NORMED if first.startswith("empty_") else cv2.TM_CCOEFF_NORMED
        return portrait_score(search, self.templates[second], method) >= MATCH_THRESHOLD

    def merge_duplicates(self):
        """合并同一角色的重复特征，删除多余文件并把上阵优先级迁移到保留的模板。

        截帧时整栏的缩放和位置会有细微差别，同一角色的两次采集可能落到不同摘要上，
        这里按比对分数合并；先迁移配置再删除文件，迁移失败时保留两份模板。
        """
        duplicates = {}
        names = list(self.templates)
        for index, kept in enumerate(names):
            if kept in duplicates:
                continue
            for other in names[index + 1:]:
                if other in duplicates:
                    continue
                if other.split("_", 1)[0] != kept.split("_", 1)[0]:
                    continue
                if self.same_character(kept, other):
                    duplicates[other] = kept
        if not duplicates:
            return
        if not self.migrate_priorities(duplicates):
            return
        for removed in duplicates:
            (self.catalog / f"{removed}.png").unlink(missing_ok=True)
            self.templates.pop(removed, None)
        CUS_LOGGER.warning(
            "同一角色采集了多份备战模板，已合并：%s",
            "，".join(f"{removed} 归入 {kept}" for removed, kept in duplicates.items()),
        )

    def migrate_priorities(self, duplicates):
        """把重复模板上的上阵优先级和名字迁移到保留的模板；返回是否迁移成功。"""
        try:
            deployment = load_currency_settings()["deployment"]
            updated = {}
            changed = False
            for key in ("front_priority", "back_priority"):
                names = []
                for name in deployment[key]:
                    mapped = duplicates.get(name, name)
                    changed = changed or mapped != name
                    if mapped not in names:
                        names.append(mapped)
                updated[key] = names
            names = {}
            for template, name in deployment["names"].items():
                mapped = duplicates.get(template, template)
                changed = changed or mapped != template
                names.setdefault(mapped, name)
            updated["names"] = names
            if changed:
                save_currency_settings({"deployment": updated})
        except (OSError, ValueError) as error:
            CUS_LOGGER.warning("上阵优先级迁移失败，保留重复的备战模板：%s", error)
            return False
        return True

    def reset_members(self):
        """新对局或落位证据不完整时，撤销已上阵角色的身份记录。"""
        self.members = {area: [None] * len(boxes) for area, boxes in AREA_BOXES.items()}

    def read_members(self, vacancies, placed):
        """按本帧空位和 OCR 上阵人数同步身份，避免把漏识别底板算成额外角色。

        空底板没匹配上并不等于格子里有人：已确认身份已经占满 OCR 人数时，其余占位
        按空位处理。

        Returns:
            区域到身份列表的快照；None 表示空位，空字符串表示身份待确认。
        """
        confirmed = 0
        unknown = 0
        for area, entries in self.members.items():
            for index in range(len(entries)):
                if index in vacancies[area]:
                    entries[index] = None
                elif entries[index] is None:
                    entries[index] = ""
                if entries[index]:
                    confirmed += 1
                elif entries[index] == "":
                    unknown += 1
        if unknown and confirmed >= placed:
            for entries in self.members.values():
                for index in range(len(entries)):
                    if entries[index] == "":
                        entries[index] = None
        return {area: entries[:] for area, entries in self.members.items()}

    def confirm_move(self, move, before, after):
        """通过备战栏角色增减确认补位或交换，并记录目标格的角色身份。

        Returns:
            (是否确认, 换回备战栏的角色模板)。补空位时第二项为 None。
        """
        incoming = next(slot.template for slot in before if slot.index == move.source)
        old_counts = Counter(slot.template for slot in before if slot.kind == "character")
        new_counts = Counter(slot.template for slot in after if slot.kind == "character")
        removed = old_counts - new_counts
        added = new_counts - old_counts
        if removed != Counter({incoming: 1}):
            return False, None
        if move.outgoing is None:
            if added:
                return False, None
            outgoing = None
        else:
            if added.total() != 1:
                return False, None
            outgoing = next(iter(added))
        self.members[move.area][move.target] = incoming
        return True, outgoing

    def scan_bench(self, frame):
        """截取九格，匹配已知模板，保存未匹配格并返回识别结果。

        Args:
            frame: 1920×1080、BGR 三通道游戏截图。

        Returns:
            从左到右排列的九个 BenchSlot。只有带角色标识的格子可用于上阵。
        """
        if frame.shape != (1080, 1920, 3):
            raise ValueError("货币战争备战识别需要 1920×1080 的 BGR 截图")
        slots = []
        for index, (left, right, top, bottom) in enumerate(BENCH_BOXES):
            tile = frame[top:bottom, left:right].copy()
            badge_score = 0.0
            for badge in self.badges:
                score = float(cv2.minMaxLoc(cv2.matchTemplate(
                    tile[:42, 77:], badge, cv2.TM_CCOEFF_NORMED
                ))[1])
                badge_score = max(badge_score, score)
            x1, x2, y1, y2 = PORTRAIT_BOX
            margin = MATCH_MARGIN
            portrait = tile[y1:y2, x1:x2].copy()
            search = tile[y1 - margin:y2 + margin, x1 - margin:x2 + margin].copy()
            best_name, best_score = "", 0.0
            for name, known in self.templates.items():
                if badge_score >= 0.9 and not name.startswith("character_"):
                    continue
                # 纯色过渡格的归一化分母为零，完全相同的像素仍应复用模板。
                if np.array_equal(portrait, known):
                    best_name, best_score = name, 1.0
                    break
                # 空格的暗色底板只含细线，使用像素误差，避免细线位置造成相关系数突变。
                method = cv2.TM_SQDIFF_NORMED if name.startswith("empty_") else cv2.TM_CCOEFF_NORMED
                score = portrait_score(search, known, method)
                if score > best_score:
                    best_name, best_score = name, score
            if best_score < MATCH_THRESHOLD:
                if badge_score >= 0.9:
                    kind = "character"
                elif float(np.std(portrait)) < 12:
                    kind = "empty"
                else:
                    kind = "item"
                digest = hashlib.sha256(portrait.tobytes()).hexdigest()[:12]
                best_name = f"{kind}_{digest}"
                save_template(self.catalog / f"{best_name}.png", portrait)
                self.templates[best_name] = portrait
                CUS_LOGGER.debug("备战格%d采集模板：%s", index + 1, best_name)
            slots.append(BenchSlot(index, best_name, best_name.split("_", 1)[0]))
        return slots

    def read_areas(self, frame):
        """返回前台、后台的空位编号，frame 须与 scan_bench 的截图契约一致。"""
        vacancies = {}
        height, width = frame.shape[:2]
        for area, boxes in AREA_BOXES.items():
            vacancies[area] = []
            for index, (left, right, top, bottom) in enumerate(boxes):
                search = frame[max(0, top - MATCH_MARGIN):min(height, bottom + MATCH_MARGIN),
                               max(0, left - MATCH_MARGIN):min(width, right + MATCH_MARGIN)]
                score = match_score(search, self.area_tiles[area][index])
                if score >= MATCH_THRESHOLD:
                    vacancies[area].append(index)
        return vacancies


def plan_deployment(slots, vacancies, counts, front_priority, back_priority, members):
    """按前后台优先级分配角色；列表里的角色只进入对应区域的空位。

    counts 是 OCR 读取的 (上阵人数, 人数上限)，两者都按前台与后台之和计算：前台固定四
    个落点，后台六个，上限随对局进度变化。未配置优先级的角色只补剩余空位，前台优先；
    前台一个角色都没有时用后台列表补位，否则无法出战。满员后按同区优先级替换较低优先
    级的角色，低优先级角色不会被主动换入。道具不参与；未知上阵角色的替换需由执行方检查
    换回的头像，发现原角色优先级更高时恢复原位。
    """
    characters = [slot for slot in slots if slot.kind == "character"]
    placed, capacity = counts
    if not 0 <= placed <= capacity <= TEAM_CAPACITY:
        raise ValueError(f"上阵人数不合法：{placed}/{capacity}")
    priorities = {"front": front_priority, "back": back_priority}
    ranks = {
        area: {name: index for index, name in enumerate(priority)}
        for area, priority in priorities.items()
    }
    remaining = capacity - placed
    occupied = sum(len(boxes) - len(vacancies[area]) for area, boxes in AREA_BOXES.items())
    if occupied < placed:
        # 识别到的占位比 OCR 人数还少，说明有上阵角色被当成空位，本轮补位会拖到它身上。
        remaining = 0

    moves = []
    open_slots = {area: list(indexes) for area, indexes in vacancies.items()}
    pool = list(characters)
    front_empty = len(open_slots["front"]) == len(AREA_BOXES["front"])

    def listed(area):
        """该区列表里的备战角色，按列表顺序。"""
        return [slot for name in priorities[area] for slot in pool if slot.template == name]

    def unlisted():
        """没有配置优先级的备战角色。"""
        return [
            slot for slot in pool
            if slot.template not in ranks["front"] and slot.template not in ranks["back"]
        ]

    def place(area, candidates):
        """按顺序把候选角色放进该区空位，返回安排的数量。"""
        pairs = list(zip(candidates, open_slots[area][:remaining]))
        if not pairs:
            return 0
        moves.extend(Deployment(slot.index, area, target) for slot, target in pairs)
        used = {slot.index for slot, _ in pairs}
        pool[:] = [slot for slot in pool if slot.index not in used]
        open_slots[area] = open_slots[area][len(pairs):]
        return len(pairs)

    # 已配置角色只进入自己的区域，按列表顺序落到该区空位。
    remaining -= place("front", listed("front"))
    if remaining > 0 and front_empty and not any(move.area == "front" for move in moves):
        # 前台一个角色都没有时无法出战，用后台列表里的角色顶一个空位。
        remaining -= place("front", listed("back")[:1])
    remaining -= place("back", listed("back"))
    # 未配置角色只补剩余空位，前台优先。
    remaining -= place("front", unlisted())
    remaining -= place("back", unlisted())
    if moves:
        return moves

    # 满员后按同区优先级替换；未配置优先级的区域不主动换人。
    for area in ("front", "back"):
        rank = ranks[area]
        if not rank or not characters:
            continue
        incoming = next(
            (slot for name in priorities[area] for slot in characters if slot.template == name),
            None,
        )
        filled = [index for index, name in enumerate(members[area]) if name is not None]
        if incoming is None or not filled:
            continue
        target = max(filled, key=lambda index: rank.get(members[area][index], len(rank)))
        if rank[incoming.template] < rank.get(members[area][target], len(rank)):
            return [Deployment(incoming.index, area, target, members[area][target])]
    return []
