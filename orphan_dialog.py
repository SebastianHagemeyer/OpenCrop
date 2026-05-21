"""Recovery dialog for pages whose QR couldn't be decoded.

Shown after "Check scan" when one or more pages came back as ``unknown``
(torn paper, scribble over the QR, page printed without one, etc). The
teacher picks which student each orphan page belongs to; multiple pages
with the same name are combined into one packet in PDF order.

Selections are written to ``<pdf>.qrfix.json`` next to the scan via
``scan_index.save_sidecar_overrides``. Both ``index_pdf`` and any later
``extract`` invocation pick up the sidecar automatically, so once the
teacher resolves orphans for a scan the fix is sticky.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pymupdf
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from scan_index import PageRecord

# Thumbnail width in dialog rows. Height follows the page's aspect.
_THUMB_W = 220
# DPI for thumbnail render. 60 → ~500px wide for A4, downscaled to
# _THUMB_W; keeps the title text + QR area legible without burning
# memory on full-resolution renders for pages we'll throw away.
_THUMB_DPI = 60


def _load_roster_names(class_path: Path | None) -> list[str]:
    """Best-effort roster read. xlsx → first-name column.

    Returns [] if the file is missing, in an unexpected format, or if
    openpyxl isn't installed (which is the case in the MSIX build
    today). The combobox stays editable either way so the teacher can
    always type a name freehand.
    """
    if not class_path or not class_path.is_file():
        return []
    try:
        import openpyxl
    except ImportError:
        return []
    try:
        wb = openpyxl.load_workbook(class_path, read_only=True, data_only=True)
        ws = wb.active
        names: list[str] = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0 and row and isinstance(row[0], str) and row[0].strip().lower().startswith("name"):
                continue  # header
            if not row:
                continue
            cell = row[0]
            if cell is None:
                continue
            name = str(cell).strip()
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def _render_thumbnail(pdf_doc: pymupdf.Document, page_number: int) -> QPixmap:
    """Render a low-DPI thumbnail of the given 1-based PDF page."""
    page = pdf_doc[page_number - 1]
    zoom = _THUMB_DPI / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
    pm = QPixmap.fromImage(img)
    return pm.scaledToWidth(_THUMB_W, Qt.SmoothTransformation)


def _default_class(pages: list[PageRecord], hint: str = "") -> str:
    """Pick a sensible default class for the orphan rows.

    Most common class among already-decoded pages wins; falls back to
    the QMARK_CLASS_NAME hint, then empty string.
    """
    classes = [p.student_class for p in pages if p.student_class]
    if classes:
        return Counter(classes).most_common(1)[0][0]
    return hint.strip()


def _decoded_names(pages: list[PageRecord]) -> set[str]:
    return {p.student_name for p in pages if p.student_name and p.qr_status != "unknown"}


class _OrphanRow(QWidget):
    """One row: thumbnail + page label + student name combobox."""

    def __init__(self, page_number: int, thumb: QPixmap, name_choices: list[str]) -> None:
        super().__init__()
        self.page_number = page_number

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(12)

        thumb_lbl = QLabel()
        thumb_lbl.setPixmap(thumb)
        thumb_lbl.setFixedSize(thumb.size())
        thumb_lbl.setStyleSheet("border: 1px solid palette(mid);")
        row.addWidget(thumb_lbl)

        right = QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(QLabel(f"<b>PDF page {page_number}</b>"))
        right.addWidget(QLabel("Student first name:"))
        self.name_cb = QComboBox()
        self.name_cb.setEditable(True)
        self.name_cb.addItem("")  # blank = skip this page
        for n in name_choices:
            self.name_cb.addItem(n)
        self.name_cb.setMinimumWidth(220)
        right.addWidget(self.name_cb)
        right.addWidget(QLabel(
            "<i>Leave blank to keep this page as an orphan. Same name<br>"
            "on multiple pages combines them into one packet (in PDF order).</i>"
        ))
        right.addStretch(1)
        row.addLayout(right, 1)

    def chosen_name(self) -> str:
        return self.name_cb.currentText().strip()


class OrphanRecoveryDialog(QDialog):
    """Modal dialog. Returns selections via ``selections`` after exec()."""

    def __init__(
        self,
        pdf_path: Path,
        pages: list[PageRecord],
        class_hint: str = "",
        roster_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Recover orphan pages")
        self.resize(720, 640)

        self._pdf_path = pdf_path
        self._orphans = [p for p in pages if p.qr_status == "unknown"]
        self._all_pages = pages
        # {pdf_page_number: override_dict}; filled in on accept.
        self.selections: dict[int, dict] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        header = QLabel(
            f"<b>{len(self._orphans)} page(s) couldn't be matched to a student "
            "via QR.</b><br>"
            "Pick which student each one belongs to. Same first name on "
            "multiple pages combines those pages into one packet (in PDF "
            "page order)."
        )
        header.setWordWrap(True)
        outer.addWidget(header)

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class:"))
        self.class_edit = QLineEdit(_default_class(pages, class_hint))
        self.class_edit.setMaximumWidth(220)
        class_row.addWidget(self.class_edit)
        class_row.addStretch(1)
        outer.addLayout(class_row)

        # Roster minus already-decoded names = suggested choices.
        roster = _load_roster_names(roster_path)
        decoded = _decoded_names(pages)
        suggestions = [n for n in roster if n not in decoded]

        # Render thumbnails once (avoid re-rendering on dialog resize).
        doc = pymupdf.open(pdf_path)
        try:
            self._rows: list[_OrphanRow] = []
            scroll_inner = QWidget()
            scroll_layout = QVBoxLayout(scroll_inner)
            scroll_layout.setContentsMargins(4, 4, 4, 4)
            scroll_layout.setSpacing(8)
            for p in self._orphans:
                thumb = _render_thumbnail(doc, p.pdf_page_number)
                row = _OrphanRow(p.pdf_page_number, thumb, suggestions)
                scroll_layout.addWidget(row)
                self._rows.append(row)
            scroll_layout.addStretch(1)
        finally:
            doc.close()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_inner)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._apply_btn = buttons.addButton("Apply && save", QDialogButtonBox.AcceptRole)
        self._apply_btn.clicked.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _on_apply(self) -> None:
        cls = self.class_edit.text().strip()
        if not cls:
            self.class_edit.setFocus()
            return

        # First pass: collect (page_number, name) for any non-blank row.
        by_name: dict[str, list[int]] = {}
        for row in self._rows:
            name = row.chosen_name()
            if not name:
                continue
            by_name.setdefault(name, []).append(row.page_number)

        # Second pass: sort each name's pages by PDF order, then issue
        # 1-based packet positions.
        selections: dict[int, dict] = {}
        for name, page_nums in by_name.items():
            page_nums.sort()
            total = len(page_nums)
            for idx, pn in enumerate(page_nums, start=1):
                selections[pn] = {
                    "class": cls,
                    "name": name,
                    "page_in_packet": idx,
                    "pages_total": total,
                }
        self.selections = selections
        self.accept()
