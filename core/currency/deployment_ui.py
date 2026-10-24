"""使用已采集头像配置前台、后台的角色上阵顺序。"""

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from core.currency.priority_ui import CurrencyPriorityListWidget
from core.currency.settings import load_currency_settings, save_currency_settings
from tool.gui.settings_dialog import SettingsDialog as BaseDialog
from tool.storage import config_path, resource_path


class DeploymentDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__("角色上阵优先级", parent)
        self.path = config_path(__file__)
        deployment = load_currency_settings(self.path)["deployment"]
        self.names = dict(deployment["names"])
        catalog = resource_path(__file__, "templates/bench")
        templates = {path.stem: path for path in sorted(catalog.glob("character_*.png"))}
        hint = QLabel("前台、后台列表里的角色只进入对应区域，越靠前越优先；未设优先级的角色只补剩余空位，前台优先。满员后按列表优先级替换较低优先级角色，道具暂不处理。“重命名”可以给头像起名字，配置和列表都会显示该名字。")
        hint.setWordWrap(True)
        self.content_layout.addWidget(hint)
        if not templates:
            hint = QLabel("尚未采集到角色头像。先在备战画面运行一次识别，再打开此窗口配置。")
            hint.setWordWrap(True)
            self.content_layout.addWidget(hint)
        columns = QHBoxLayout()
        self.content_layout.addLayout(columns)
        self.lists = {}
        for key, title in (
            ("available", "补位角色"),
            ("front_priority", "前台优先级"),
            ("back_priority", "后台优先级"),
        ):
            column = QVBoxLayout()
            columns.addLayout(column)
            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            column.addWidget(label)
            widget = CurrencyPriorityListWidget()
            widget.setViewMode(QListWidget.IconMode)
            widget.setFlow(QListWidget.LeftToRight)
            widget.setWrapping(True)
            widget.setResizeMode(QListWidget.Adjust)
            widget.setDragEnabled(True)
            widget.setAcceptDrops(True)
            widget.setDropIndicatorShown(False)
            widget.setDragDropMode(QAbstractItemView.DragDrop)
            widget.setDefaultDropAction(Qt.MoveAction)
            widget.setSelectionMode(QAbstractItemView.SingleSelection)
            widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
            widget.setIconSize(QSize(72, 88))
            widget.setGridSize(QSize(96, 122))
            widget.setMinimumSize(210, 280)
            self.lists[key] = widget
            column.addWidget(widget)
            for target, text in (
                ("available", "仅用于补位"),
                ("front_priority", "移到前台"),
                ("back_priority", "移到后台"),
            ):
                if target == key:
                    continue
                button = QPushButton(text)
                button.setObjectName(f"{key}_to_{target}")
                button.clicked.connect(
                    lambda checked=False, source=key, destination=target:
                    self.transfer(source, destination)
                )
                column.addWidget(button)
            rename = QPushButton("重命名")
            rename.setObjectName(f"{key}_rename")
            rename.clicked.connect(lambda checked=False, source=key: self.rename(source))
            column.addWidget(rename)

        configured = set(deployment["front_priority"] + deployment["back_priority"])
        for key, widget in self.lists.items():
            names = [name for name in templates if name not in configured] if key == "available" else deployment[key]
            for name in names:
                path = templates.get(name)
                icon = QIcon(str(path)) if path is not None else QIcon()
                item = QListWidgetItem(icon, self.label(name))
                item.setData(Qt.UserRole, name)
                item.setToolTip(self.tooltip(name))
                item.setSizeHint(QSize(90, 116))
                widget.addItem(item)
        available = self.screen().availableGeometry()
        self.resize(min(1000, available.width() - 80), min(680, available.height() - 100))

    def label(self, template):
        """列表里显示的名字；没起名字时用模板名尾部，稳定且能和配置文件对上。"""
        return self.names.get(template) or template.split("_", 1)[-1][:8]

    def tooltip(self, template):
        """悬停提示：名字和对应的模板文件名。"""
        name = self.names.get(template)
        return f"{name}\n{template}" if name else template

    def rename(self, key):
        """给该列表里选中的角色起一个可读名字，写入配置后在列表和设置里显示。"""
        item = self.lists[key].currentItem()
        if item is None:
            return
        template = item.data(Qt.UserRole)
        name, accepted = QInputDialog.getText(
            self, "角色名字", "名字：", text=self.names.get(template, ""),
        )
        if not accepted or not name.strip():
            return
        self.names[template] = name.strip()
        item.setText(self.label(template))
        item.setToolTip(self.tooltip(template))

    def transfer(self, source, target):
        """将选中的角色移入指定区域的优先级末尾。"""
        origin = self.lists[source]
        row = origin.currentRow()
        if row < 0:
            return
        item = origin.takeItem(row)
        self.lists[target].addItem(item)
        self.lists[target].setCurrentItem(item)

    def persist(self):
        deployment = {"names": dict(self.names)}
        for key in ("front_priority", "back_priority"):
            widget = self.lists[key]
            deployment[key] = [widget.item(i).data(Qt.UserRole) for i in range(widget.count())]
        save_currency_settings({"deployment": deployment}, path=self.path)
