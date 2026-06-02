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
import os

import numpy as np
import pymupdf
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
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


def _infer_packet_size(pages: list[PageRecord]) -> int:
    """Best guess at one student's packet length — the Fill-down default.

    The modal ``pages_total`` among QR-decoded packets in this scan is the
    most reliable signal. If nothing decoded (every page is an orphan),
    fall back to the worksheet PDF the dashboard handed us via
    ``QMARK_SHEET_PATH`` — its page count is exactly one packet. Returns a
    small default when neither is available; the teacher sets the spinner
    by hand in that case.
    """
    totals = [p.pages_total for p in pages if p.pages_total]
    if totals:
        return Counter(totals).most_common(1)[0][0]
    sheet = os.environ.get("QMARK_SHEET_PATH", "").strip()
    if sheet:
        try:
            doc = pymupdf.open(sheet)
            try:
                n = len(doc)
            finally:
                doc.close()
            if n > 0:
                return n
        except Exception:
            pass
    return 2


class _OrphanRow(QWidget):
    """One row: thumbnail + page label + student name + packet page spinner."""

    # Emitted when the row's "Fill down" button is clicked; the dialog
    # connects it to a handler bound to this row's index.
    fill_requested = Signal()

    def __init__(self, page_number: int, default_packet_page: int,
                 thumb: QPixmap, name_choices: list[str],
                 preset_name: str = "",
                 hint_text: str | None = None,
                 packet_max: int = 9,
                 enable_fill_down: bool = False) -> None:
        super().__init__()
        self.page_number = page_number
        self._preset_name = preset_name

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
        self.name_cb.addItem("")  # blank = clear / orphan this page
        # Surface the preset name at the top of the dropdown if it's
        # not already in the roster suggestions, so it stays a
        # one-click pick after the dialog reopens.
        if preset_name and preset_name not in name_choices:
            self.name_cb.addItem(preset_name)
        for n in name_choices:
            self.name_cb.addItem(n)
        self.name_cb.setMinimumWidth(220)
        if preset_name:
            self.name_cb.setCurrentText(preset_name)
        right.addWidget(self.name_cb)

        # Packet page picker: when the same student appears on multiple
        # rows, the spinbox tells us which page goes first inside the
        # packet. Default is PDF order (row 1 = packet page 1, row 2 = 2,
        # ...), so the existing behaviour is the default; the spinner is
        # only needed when pages were handed in out of order.
        page_row = QHBoxLayout()
        page_row.addWidget(QLabel("Packet page:"))
        self.packet_sb = QSpinBox()
        self.packet_sb.setRange(1, max(9, packet_max))
        self.packet_sb.setValue(default_packet_page)
        self.packet_sb.setMaximumWidth(70)
        page_row.addWidget(self.packet_sb)
        page_row.addStretch(1)
        if enable_fill_down:
            self.fill_btn = QPushButton("Fill down ↓")
            self.fill_btn.setToolTip(
                "Assign this name to this page and the following pages in "
                "scan order — up to 'Pages per packet' at the top — and "
                "number them packet page 1, 2, 3 … (this page becomes "
                "packet page 1)."
            )
            self.fill_btn.clicked.connect(self.fill_requested)
            page_row.addWidget(self.fill_btn)
        right.addLayout(page_row)

        if hint_text is None:
            hint_text = (
                "Leave name blank to keep this page as an orphan. Same name "
                "on multiple pages combines them into one packet, sorted by "
                "the packet-page values you set above."
            )
        hint_lbl = QLabel(f"<i>{hint_text}</i>")
        hint_lbl.setWordWrap(True)
        right.addWidget(hint_lbl)
        right.addStretch(1)
        row.addLayout(right, 1)

    def chosen_name(self) -> str:
        return self.name_cb.currentText().strip()

    def set_name(self, name: str) -> None:
        # Editable combo, so setCurrentText works even for a name that
        # isn't among the roster suggestions.
        self.name_cb.setCurrentText(name)

    def packet_page(self) -> int:
        return self.packet_sb.value()

    def set_packet_page(self, page: int) -> None:
        self.packet_sb.setValue(min(page, self.packet_sb.maximum()))

    def is_cleared(self) -> bool:
        """True when the user blanked a name that started non-blank.

        Used in edit mode to know which pages should drop their existing
        override rather than just be left untouched.
        """
        return bool(self._preset_name) and not self.chosen_name()


class OrphanRecoveryDialog(QDialog):
    """Modal dialog. Returns selections via ``selections`` after exec()."""

    def __init__(
        self,
        pdf_path: Path,
        pages: list[PageRecord],
        class_hint: str = "",
        roster_path: Path | None = None,
        pages_to_edit: list[int] | None = None,
        edit_title: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        edit_mode = pages_to_edit is not None
        self._edit_mode = edit_mode
        self.setWindowTitle(edit_title or "Recover orphan pages")
        self.resize(720, 640)

        self._pdf_path = pdf_path
        if edit_mode:
            wanted = set(pages_to_edit or [])
            # Preserve PDF order so packet pages and thumbnails line up
            # with what the teacher sees in the roster.
            self._target_pages = [p for p in pages if p.pdf_page_number in wanted]
        else:
            self._target_pages = [p for p in pages if p.qr_status == "unknown"]
        self._all_pages = pages
        # {pdf_page_number: override_dict}; filled in on accept.
        self.selections: dict[int, dict] = {}
        # PDF page numbers whose existing override should be cleared
        # (user blanked the name in edit mode). Set on accept.
        self.cleared_pages: set[int] = set()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        if edit_mode:
            n = len(self._target_pages)
            header = QLabel(
                f"<b>Editing {n} page(s) for "
                f"{edit_title.split(' — ')[-1] if edit_title else 'this student'}.</b><br>"
                "Change the name to reassign a page to a different student, "
                "or blank the name to send it back to the orphan list. The "
                "packet-page spinner controls the page's position inside "
                "the packet."
            )
        else:
            header = QLabel(
                f"<b>{len(self._target_pages)} page(s) couldn't be matched to a student "
                "via QR.</b><br>"
                "Pick which student each one belongs to. Same first name on "
                "multiple pages combines those pages into one packet (in PDF "
                "page order).<br>"
                "<b>Tip:</b> type a student's name on their first page, then "
                "click <b>Fill down ↓</b> to hand them that page and the next "
                "few in scan order — set how many in <b>Pages per packet</b>."
            )
        header.setWordWrap(True)
        outer.addWidget(header)

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class:"))
        # In edit mode, prefer the student's existing class so we don't
        # silently rewrite it when the teacher just tweaks a packet page.
        default_cls = ""
        if edit_mode:
            for p in self._target_pages:
                if p.student_class:
                    default_cls = p.student_class
                    break
        if not default_cls:
            default_cls = _default_class(pages, class_hint)
        self.class_edit = QLineEdit(default_cls)
        self.class_edit.setMaximumWidth(220)
        class_row.addWidget(self.class_edit)
        class_row.addStretch(1)
        outer.addLayout(class_row)

        # Fill-down config (orphan mode only): how many consecutive pages
        # one click claims for a student. Default to this scan's inferred
        # packet size so the common "all packets the same length" case
        # needs no adjustment.
        self.packet_size_sb = None
        if not edit_mode and self._target_pages:
            packet_cap = max(9, len(self._target_pages))
            fill_row = QHBoxLayout()
            fill_row.addWidget(QLabel("Pages per packet (for Fill down):"))
            self.packet_size_sb = QSpinBox()
            self.packet_size_sb.setRange(1, packet_cap)
            self.packet_size_sb.setValue(
                min(max(1, _infer_packet_size(pages)), packet_cap)
            )
            self.packet_size_sb.setMaximumWidth(70)
            fill_row.addWidget(self.packet_size_sb)
            fill_row.addStretch(1)
            outer.addLayout(fill_row)

        # Roster minus already-decoded names = suggested choices. In
        # edit mode, the current student is allowed too (otherwise the
        # combobox wouldn't include their own name to keep).
        roster = _load_roster_names(roster_path)
        decoded = _decoded_names(pages)
        if edit_mode:
            preset_names = {p.student_name for p in self._target_pages if p.student_name}
            decoded = decoded - preset_names
        suggestions = [n for n in roster if n not in decoded]

        # Render thumbnails once (avoid re-rendering on dialog resize).
        doc = pymupdf.open(pdf_path)
        try:
            self._rows: list[_OrphanRow] = []
            scroll_inner = QWidget()
            scroll_layout = QVBoxLayout(scroll_inner)
            scroll_layout.setContentsMargins(4, 4, 4, 4)
            scroll_layout.setSpacing(8)
            for idx, p in enumerate(self._target_pages, start=1):
                thumb = _render_thumbnail(doc, p.pdf_page_number)
                default_packet = (p.page_in_packet
                                  if edit_mode and p.page_in_packet else idx)
                preset_name = p.student_name if edit_mode and p.student_name else ""
                hint = None
                if edit_mode:
                    hint = (
                        "Blank the name to remove this page from the student "
                        "(it goes back to the orphan banner so you can reassign "
                        "it). Otherwise change the name to move it to a "
                        "different student, or tweak the packet-page spinner."
                    )
                row = _OrphanRow(
                    p.pdf_page_number, default_packet, thumb, suggestions,
                    preset_name=preset_name, hint_text=hint,
                    packet_max=max(9, len(self._target_pages)),
                    enable_fill_down=not edit_mode,
                )
                if not edit_mode:
                    # Default-arg binds the row's index at definition time so
                    # each button reports its own row, not the loop's last.
                    row.fill_requested.connect(
                        lambda i=len(self._rows): self._fill_down_from(i)
                    )
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

    def _fill_down_from(self, start_idx: int) -> None:
        """Claim a run of consecutive pages for the name on row ``start_idx``.

        Copies that row's name onto it and the following rows (scan order,
        as displayed), up to 'Pages per packet', numbering them packet page
        1, 2, 3 … So naming a student's cover page and clicking Fill down
        assigns their whole packet in one go. Stops cleanly at the end of
        the orphan list — a short tail just fills fewer pages.
        """
        if self.packet_size_sb is None:
            return
        if not (0 <= start_idx < len(self._rows)):
            return
        name = self._rows[start_idx].chosen_name()
        if not name:
            QMessageBox.information(
                self, "Name needed first",
                "Type the student's first name on this row, then click "
                "Fill down.",
            )
            return
        count = self.packet_size_sb.value()
        end_idx = min(start_idx + count, len(self._rows))
        for packet_page, i in enumerate(range(start_idx, end_idx), start=1):
            self._rows[i].set_name(name)
            self._rows[i].set_packet_page(packet_page)

    def _on_apply(self) -> None:
        cls = self.class_edit.text().strip()
        if not cls:
            self.class_edit.setFocus()
            return

        # Collect (name, packet_page, pdf_page) for every named row,
        # plus the set of pages the teacher actively blanked (only
        # meaningful in edit mode — orphan mode just leaves blanks be).
        by_name: dict[str, list[tuple[int, int]]] = {}
        cleared: set[int] = set()
        for row in self._rows:
            name = row.chosen_name()
            if not name:
                if row.is_cleared():
                    cleared.add(row.page_number)
                continue
            by_name.setdefault(name, []).append((row.packet_page(), row.page_number))

        # Validate: a student can't have two pages claiming the same
        # packet position. Surface that as an actionable warning rather
        # than silently overwriting the override.
        for name, entries in by_name.items():
            pps = [pp for pp, _ in entries]
            if len(pps) != len(set(pps)):
                QMessageBox.warning(
                    self,
                    "Duplicate packet page",
                    f"<b>{name}</b> has two pages set to the same packet "
                    "position. Each page in a student's packet needs a "
                    "unique packet-page number — adjust the spinner on "
                    "the conflicting rows.",
                )
                return

        # pages_total = max packet-page picked for that name. Allows a
        # single-page packet (everyone at 1) or a sparse one (1 and 3
        # with no 2) — the teacher knows their data better than we do.
        selections: dict[int, dict] = {}
        for name, entries in by_name.items():
            pages_total = max(pp for pp, _ in entries)
            for pp, pn in entries:
                selections[pn] = {
                    "class": cls,
                    "name": name,
                    "page_in_packet": pp,
                    "pages_total": pages_total,
                }
        self.selections = selections
        self.cleared_pages = cleared
        self.accept()
