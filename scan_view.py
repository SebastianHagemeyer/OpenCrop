"""Class-roster style display for Check scan results.

Replaces the old debug-log dump with a visual layout: a warning banner
when any page couldn't be matched to a student (with a one-click
Resolve... handle into the recovery dialog), and a table of student
rows with colored status pills below it.

The widget is purely a view — the Launcher owns the indexed page list
and feeds it via :meth:`set_pages`. Clicking Resolve fires
:attr:`resolve_clicked`; the Launcher hooks that up to the orphan
dialog.
"""

from __future__ import annotations

from collections import Counter

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scan_index import PageRecord, group_into_students

# Colors for the status pill column. RGB chosen to read on both light
# and dark Qmark themes (semi-transparent backgrounds let the row's
# alternating-base colour show through).
_STATUS_COLORS = {
    "decoded":      QColor(56, 142, 60, 200),    # green
    "preprocessed": QColor(56, 142, 60, 200),    # green (same — it decoded, just needed a fallback)
    "inferred":     QColor(25, 118, 210, 200),   # blue
    "override":     QColor(123, 31, 162, 200),   # purple
    "unknown":      QColor(211, 47, 47, 200),    # red — shouldn't appear in the roster (banner instead)
}

_STATUS_LABEL = {
    "decoded":      "decoded",
    "preprocessed": "decoded*",  # asterisk hints that it needed Otsu/rescale
    "inferred":     "inferred",
    "override":     "manual",
    "unknown":      "?",
}


class OrphanBanner(QFrame):
    """Warning banner shown when one or more pages have no student match."""

    resolve_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "OrphanBanner { background-color: rgba(255, 152, 0, 40); "
            "border: 1px solid rgba(255, 152, 0, 160); border-radius: 4px; }"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(12)

        self._label = QLabel()
        self._label.setWordWrap(True)
        font = QFont()
        font.setBold(True)
        self._label.setFont(font)
        row.addWidget(self._label, 1)

        self._btn = QPushButton("Resolve...")
        self._btn.clicked.connect(self.resolve_clicked.emit)
        row.addWidget(self._btn)

        self.hide()

    def update_pages(self, orphan_page_numbers: list[int]) -> None:
        n = len(orphan_page_numbers)
        if n == 0:
            self.hide()
            return
        page_list = ", ".join(str(p) for p in orphan_page_numbers)
        plural = "s" if n != 1 else ""
        self._label.setText(
            f"⚠ {n} page{plural} couldn't be matched to a student "
            f"(pages {page_list}).  Click Resolve to assign them by hand."
        )
        self.show()


class ScanResultView(QWidget):
    """Class-roster + orphan banner. Updated via :meth:`set_pages`."""

    resolve_clicked = Signal()
    # Emitted when the teacher right-clicks a roster row and picks the
    # corresponding action. The launcher does the actual work (opens
    # the page viewer / mutates the sidecar / repaints) so this widget
    # stays a pure view.
    view_pages_requested = Signal(str)             # folder_name
    skip_toggled = Signal(str, bool)               # folder_name, new_skipped_state

    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.banner = OrphanBanner()
        self.banner.resolve_clicked.connect(self.resolve_clicked.emit)
        outer.addWidget(self.banner)

        self._header_label = QLabel("Run Check scan to index this PDF.")
        self._header_label.setStyleSheet("color: palette(mid);")
        outer.addWidget(self._header_label)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Student", "Pages", "Status"])
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setSortingEnabled(False)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        outer.addWidget(self._tree, 1)

    def clear(self) -> None:
        self._tree.clear()
        self.banner.hide()
        self._header_label.setText("Run Check scan to index this PDF.")

    def set_pages(self, pages: list[PageRecord],
                  skipped: set[str] | None = None) -> None:
        """Re-render the view from a freshly indexed (or re-grouped) page list.

        ``skipped`` is the set of student folder_names the teacher has
        excluded from this scan; those rows render strikethrough and
        grey, and the header counts them separately.
        """
        skipped = skipped or set()
        # group_into_students mutates: it re-runs _infer_missing and
        # rebuilds groups. Pages without a name end up in groups with
        # student_class == "UNKNOWN" — those become orphans in the
        # banner, not roster rows.
        groups = group_into_students(pages)

        roster: list = []
        orphan_pages: list[int] = []
        for g in groups:
            if g.student_class == "UNKNOWN":
                orphan_pages.extend(p.pdf_page_number for p in g.pages)
            else:
                roster.append(g)

        # Class header: most common class across the roster (handles the
        # rare cross-class scan; otherwise it's just "10MATD").
        classes = Counter(g.student_class for g in roster)
        if classes:
            top_cls = classes.most_common(1)[0][0]
            extra = ""
            if len(classes) > 1:
                extras = sorted(c for c in classes if c != top_cls)
                extra = " + " + ", ".join(extras)
            page_count = sum(len(g.pages) for g in roster)
            n_skipped = sum(1 for g in roster if g.folder_name in skipped)
            skip_note = (
                f", <span style='color: palette(mid);'>{n_skipped} skipped</span>"
                if n_skipped else ""
            )
            self._header_label.setText(
                f"<b>{top_cls}{extra}</b> — {len(roster)} student"
                f"{'s' if len(roster) != 1 else ''}{skip_note}, "
                f"{page_count} page{'s' if page_count != 1 else ''} matched"
            )
            self._header_label.setStyleSheet("")
        else:
            self._header_label.setText("No students matched.")
            self._header_label.setStyleSheet("color: palette(mid);")

        self._tree.clear()
        roster.sort(key=lambda g: g.student_name.lower())
        rank = {"decoded": 0, "preprocessed": 0, "inferred": 1,
                "override": 2, "unknown": 3}
        for g in roster:
            page_nums = ", ".join(str(p.pdf_page_number) for p in g.pages)
            statuses = [p.qr_status for p in g.pages]
            # Pick the "worst" status to color the row by — override
            # outranks inferred outranks decoded. Keeps a manual fix
            # visibly distinct.
            worst = max(statuses, key=lambda s: rank.get(s, 0))
            label = _STATUS_LABEL.get(worst, worst)
            is_skipped = g.folder_name in skipped
            status_text = f"skipped — {label}" if is_skipped else label
            item = QTreeWidgetItem([g.student_name, page_nums, status_text])
            # Stash the folder_name so context-menu / double-click
            # handlers can identify the row without re-deriving it.
            item.setData(0, Qt.UserRole, g.folder_name)

            if is_skipped:
                grey = QBrush(QColor(140, 140, 140))
                for col in (0, 1, 2):
                    item.setForeground(col, grey)
                    f = item.font(col)
                    f.setStrikeOut(True)
                    f.setItalic(True)
                    item.setFont(col, f)
            else:
                color = _STATUS_COLORS.get(worst)
                if color:
                    # Tint the status cell so the colour reads at a glance
                    # without overwhelming the row.
                    item.setForeground(2, QBrush(color))
                    font = item.font(2)
                    font.setBold(True)
                    item.setFont(2, font)
            self._tree.addTopLevelItem(item)
        self._skipped_set = set(skipped)  # snapshot for the context menu

        self.banner.update_pages(sorted(set(orphan_pages)))

    # ---------- context menu / interactions ----------

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        folder = item.data(0, Qt.UserRole)
        if not folder:
            return
        is_skipped = folder in getattr(self, "_skipped_set", set())
        menu = QMenu(self._tree)
        view_action = QAction("View pages...", menu)
        view_action.triggered.connect(
            lambda: self.view_pages_requested.emit(folder)
        )
        menu.addAction(view_action)
        menu.addSeparator()
        toggle_label = (
            "Include in extraction" if is_skipped else "Skip from extraction"
        )
        toggle_action = QAction(toggle_label, menu)
        toggle_action.triggered.connect(
            lambda: self.skip_toggled.emit(folder, not is_skipped)
        )
        menu.addAction(toggle_action)
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        folder = item.data(0, Qt.UserRole)
        if folder:
            self.view_pages_requested.emit(folder)
