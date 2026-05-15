"""PySide6 GUI for defining per-question regions on a reference student's pages.

Usage:
    python make_template.py [path/to/scan.pdf]

If no PDF is given, you'll be asked to pick one.

Workflow:
    1. The scan is indexed (QR-decoded, grouped into students).
    2. The first student is shown by default — change via the dropdown.
    3. Click + drag a rectangle over each question, in order. Q numbers
       auto-advance (Q01 → Q02 → ...).
    4. Use Prev / Next (or Left / Right arrows) to flip pages within the packet.
    5. Save Template writes a YAML keyed by student-relative page index, so it
       applies to every other student in the same exam.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import threading
from pathlib import Path

# When launched from qmark, the parent passes the marking scheme's question
# codes via QMARK_QUESTION_CODES so regions adopt the scheme's structure
# (e.g. "Q1,Q2,Q3,Q4") instead of always being numbered Q01..QNN. Empty
# list = standalone behaviour (auto-incrementing Q01..).
QUESTION_CODES = [
    c.strip() for c in os.environ.get("QMARK_QUESTION_CODES", "").split(",")
    if c.strip()
]


def _code_for(n: int) -> str:
    """Return the label for the n-th region (1-based).

    Uses the scheme's codes if provided and we haven't run out; otherwise
    falls back to zero-padded Q01/Q02/... for any extra regions.
    """
    if QUESTION_CODES and 1 <= n <= len(QUESTION_CODES):
        return QUESTION_CODES[n - 1]
    return f"Q{n:02d}"

import numpy as np
import pymupdf
import yaml
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from scan_index import (
    PageRecord, StudentGroup, _decode_qr, _parse_qr, _render_page,
    group_into_students,
)

INDEX_DPI = 200       # QR scan dpi (lower = faster startup)
DISPLAY_DPI = 180     # rendering dpi for the on-screen image
MIN_BBOX = 0.01       # ignore drags smaller than 1% of page (likely accidental)
ICON_PATH = Path(__file__).resolve().parent / "paper.ico"


def _writable_templates_dir(pdf_path: Path) -> Path:
    """Where to default the YAML template save dialog for `pdf_path`.

    Prefer the qmark Sheets folder when launched from the dashboard so
    the template sits beside the worksheet PDF it describes. Otherwise a
    templates/ folder next to the PDF (or the PDF's own dir), falling
    back to %LOCALAPPDATA%\\OpenCrop\\templates when nothing writable
    is reachable — that's the case when the PDF lives under
    Program Files\\WindowsApps inside an MSIX install.
    """
    qmark_sheets = os.environ.get("QMARK_SHEETS_DIR", "").strip()
    if qmark_sheets:
        try:
            d = Path(qmark_sheets)
            d.mkdir(parents=True, exist_ok=True)
            if os.access(str(d), os.W_OK):
                return d
        except OSError:
            pass
    natural = pdf_path.parent / "templates"
    candidate = natural if natural.exists() else pdf_path.parent
    if os.access(str(candidate), os.W_OK):
        return candidate
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    fallback = Path(base) / "OpenCrop" / "templates"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


class PageView(QWidget):
    """Custom widget: renders a page pixmap, overlays region rectangles, and emits
    normalized bbox coords when the user finishes dragging a new rectangle."""

    rect_drawn = Signal(float, float, float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setCursor(Qt.CrossCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(204, 204, 204))  # gray80-ish
        self.setPalette(pal)

        self._pixmap: QPixmap | None = None
        self._regions: list[dict] = []
        self._draw_start: QPoint | None = None
        self._draw_end: QPoint | None = None
        # Cache the smoothly-scaled pixmap keyed on (sw, sh) — scaling a
        # 1500x2000 source on every paintEvent (which fires on every
        # mouseMove during a drag) is the dominant cost in this widget.
        self._scaled_cache: QPixmap | None = None
        self._cache_dims: tuple[int, int] = (0, 0)

    def _invalidate_cache(self) -> None:
        self._scaled_cache = None
        self._cache_dims = (0, 0)

    def set_page(self, pixmap: QPixmap, regions: list[dict]) -> None:
        self._pixmap = pixmap
        self._regions = regions
        self._draw_start = None
        self._draw_end = None
        self._invalidate_cache()
        self.update()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._invalidate_cache()

    def _layout(self) -> tuple[float, int, int]:
        """Return (scale, scaled_w, scaled_h) for the current pixmap+widget size."""
        if self._pixmap is None:
            return 1.0, 0, 0
        cw = max(self.width(), 1)
        ch = max(self.height(), 1)
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return 1.0, 0, 0
        scale = min(cw / pw, ch / ph)
        return scale, max(1, int(pw * scale)), max(1, int(ph * scale))

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        if self._pixmap is None:
            return
        scale, sw, sh = self._layout()
        if self._scaled_cache is None or self._cache_dims != (sw, sh):
            self._scaled_cache = self._pixmap.scaled(
                sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._cache_dims = (sw, sh)
        painter.drawPixmap(0, 0, self._scaled_cache)

        pw, ph = self._pixmap.width(), self._pixmap.height()
        red = QPen(QColor("red"), 2)
        painter.setPen(red)
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        painter.setFont(font)
        for r in self._regions:
            x0, y0, x1, y1 = r["bbox"]
            cx0, cy0 = x0 * pw * scale, y0 * ph * scale
            cx1, cy1 = x1 * pw * scale, y1 * ph * scale
            painter.drawRect(int(cx0), int(cy0), int(cx1 - cx0), int(cy1 - cy0))
            painter.drawText(int(cx0 + 4), int(cy0 + 18), r["q"])

        if self._draw_start is not None and self._draw_end is not None:
            preview = QPen(QColor("blue"), 2, Qt.DashLine)
            painter.setPen(preview)
            x0 = min(self._draw_start.x(), self._draw_end.x())
            y0 = min(self._draw_start.y(), self._draw_end.y())
            x1 = max(self._draw_start.x(), self._draw_end.x())
            y1 = max(self._draw_start.y(), self._draw_end.y())
            painter.drawRect(x0, y0, x1 - x0, y1 - y0)

    def mousePressEvent(self, ev) -> None:
        if self._pixmap is None:
            return
        self._draw_start = ev.position().toPoint()
        self._draw_end = self._draw_start
        self.update()

    def mouseMoveEvent(self, ev) -> None:
        if self._draw_start is None:
            return
        self._draw_end = ev.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        if self._draw_start is None or self._pixmap is None:
            return
        scale, _, _ = self._layout()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        end = ev.position().toPoint()
        x0c, y0c = self._draw_start.x(), self._draw_start.y()
        x1c, y1c = end.x(), end.y()
        self._draw_start = None
        self._draw_end = None
        self.update()
        if scale <= 0 or pw == 0 or ph == 0:
            return
        nx0 = max(0.0, min(x0c, x1c) / scale / pw)
        ny0 = max(0.0, min(y0c, y1c) / scale / ph)
        nx1 = min(1.0, max(x0c, x1c) / scale / pw)
        ny1 = min(1.0, max(y0c, y1c) / scale / ph)
        if (nx1 - nx0) < MIN_BBOX or (ny1 - ny0) < MIN_BBOX:
            return
        self.rect_drawn.emit(nx0, ny0, nx1, ny1)


class TemplateEditor(QMainWindow):
    # Signals fired from the background indexing worker back onto the UI
    # thread. The token lets us ignore stale results from a previous load
    # if the user opened a new PDF while the first one was still being
    # indexed.
    # `done=False` is an incremental update; `done=True` is the final
    # emit after the whole PDF has been processed (legacy 2-segment QR
    # groups without pages_total only show up on the final emit).
    _groups_updated = Signal(int, str, list, bool, int, int)
    # (token, pdf_path, groups, done, page_num, total_pages)
    _index_failed = Signal(int, str)         # (token, error)

    def __init__(self, pdf_path: Path) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.setWindowTitle(f"Template editor — {self.pdf_path.name}")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.doc: pymupdf.Document | None = None
        self.groups: list[StudentGroup] = []
        self.current_group_idx = 0
        self.current_page_idx = 0
        self.regions: list[dict] = []
        # Packet-relative page indices (1-based) that the user marked as
        # "MC page" — extract.py renders each of these as a full-page
        # image per student so the MC grader can show what the student
        # filled in next to the answer cells.
        self.mc_pages: set[int] = set()
        self.next_q_num = 1
        self._index_token = 0


        self._build_ui()
        self._groups_updated.connect(self._on_groups_updated)
        self._index_failed.connect(self._on_index_failed)
        self._load_pdf(self.pdf_path)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        open_btn = QPushButton("Open PDF…")
        open_btn.clicked.connect(self._open_pdf_dialog)
        top.addWidget(open_btn)
        top.addWidget(QLabel("  Reference student: "))
        self.student_combo = QComboBox()
        self.student_combo.setMinimumWidth(220)
        self.student_combo.currentIndexChanged.connect(self._on_student_change)
        top.addWidget(self.student_combo)
        self.page_label = QLabel("")
        top.addSpacing(12)
        top.addWidget(self.page_label)
        top.addStretch(1)
        outer.addLayout(top)

        body = QHBoxLayout()
        outer.addLayout(body, 1)

        self.page_view = PageView()
        self.page_view.rect_drawn.connect(self._on_rect_drawn)
        body.addWidget(self.page_view, 1)

        side = QFrame()
        side.setFixedWidth(260)
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(8, 8, 8, 8)
        body.addWidget(side)

        side_lay.addWidget(QLabel("Defining:"))
        self.defining_label = QLabel(_code_for(1))
        big = QFont()
        big.setPointSize(14)
        big.setBold(True)
        self.defining_label.setFont(big)
        side_lay.addWidget(self.defining_label)

        side_lay.addSpacing(8)
        side_lay.addWidget(QLabel("Regions defined:"))
        self.region_list = QListWidget()
        side_lay.addWidget(self.region_list, 1)
        self.del_btn = QPushButton("Delete selected")
        self.del_btn.clicked.connect(self._delete_selected)
        side_lay.addWidget(self.del_btn)

        side_lay.addSpacing(8)
        self.mc_toggle_btn = QPushButton("Mark this page as MC")
        self.mc_toggle_btn.setCheckable(True)
        self.mc_toggle_btn.setToolTip(
            "Flag the current packet page as a multiple-choice answer "
            "sheet. Extract will render the whole page (no bboxes) so "
            "the MC grader can display it alongside the answer cells."
        )
        self.mc_toggle_btn.clicked.connect(self._toggle_mc_page)
        side_lay.addWidget(self.mc_toggle_btn)

        side_lay.addSpacing(8)
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev page")
        self.prev_btn.clicked.connect(self._prev_page)
        self.next_btn = QPushButton("Next page ▶")
        self.next_btn.clicked.connect(self._next_page)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        side_lay.addLayout(nav)

        self.save_btn = QPushButton("Save template…")
        self.save_btn.clicked.connect(self._save)
        side_lay.addWidget(self.save_btn)

        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._prev_page)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._next_page)

    # ---------- PDF loading / indexing ----------

    def _open_pdf_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF (*.pdf)")
        if not path:
            return
        self._load_pdf(Path(path))

    def _set_busy(self, busy: bool) -> None:
        """Disable interactive controls while indexing runs on a worker
        thread so the user can't drive the editor into a half-loaded state."""
        for w in (self.student_combo, self.prev_btn, self.next_btn,
                  self.save_btn, self.del_btn, self.mc_toggle_btn):
            w.setEnabled(not busy)

    def _load_pdf(self, path: Path) -> None:
        """Kick off incremental QR indexing on a worker thread.

        Rather than decoding every page before letting the user touch the
        UI, the worker streams: decode page → re-group → emit any newly
        complete student packets. The editor bootstraps as soon as the
        first packet is complete (typically ~1-2 s for a 2-page packet
        scanned in order), and the rest of the students populate the
        Reference combo as more pages come in. A generation token ignores
        stale emits if the user opens another PDF mid-index.
        """
        if self.doc is not None:
            self.doc.close()
            self.doc = None
        self.pdf_path = path
        self._index_token += 1
        token = self._index_token
        self.groups = []
        self.setWindowTitle(f"Template editor — {path.name}  (indexing…)")
        self._set_busy(True)
        self.page_label.setText("Indexing pages and decoding QR codes…")

        def work() -> None:
            try:
                doc = pymupdf.open(str(path))
                total = len(doc)
                raw_pages: list[PageRecord] = []
                last_sig: tuple = ()
                for page_num, page in enumerate(doc, start=1):
                    if token != self._index_token:
                        doc.close()
                        return
                    img = _render_page(page, INDEX_DPI)
                    text, status = _decode_qr(img)
                    cls = name = None
                    pip = tot = None
                    if text:
                        parsed = _parse_qr(text)
                        if parsed:
                            cls, name, pip, tot = parsed
                        else:
                            status = "unknown"
                    raw_pages.append(PageRecord(
                        page_num, cls, name, pip, tot, text, status,
                    ))
                    # Deep-copy so the grouping pass's _infer_missing
                    # doesn't poison raw_pages for the next iteration.
                    snapshot = [dataclasses.replace(r) for r in raw_pages]
                    grouped = group_into_students(snapshot)
                    complete = [
                        g for g in grouped
                        if g.student_class != "UNKNOWN"
                        and g.pages
                        and g.pages[0].pages_total is not None
                        and len(g.pages) >= g.pages[0].pages_total
                    ]
                    sig = tuple(
                        (g.folder_name, len(g.pages)) for g in complete
                    )
                    if sig != last_sig:
                        last_sig = sig
                        self._groups_updated.emit(
                            token, str(path), complete, False,
                            page_num, total,
                        )
                doc.close()
                # Final emit picks up legacy 2-segment groups (pages_total
                # was None throughout) that the streaming pass skipped.
                snapshot = [dataclasses.replace(r) for r in raw_pages]
                final = [g for g in group_into_students(snapshot)
                         if g.student_class != "UNKNOWN"]
                self._groups_updated.emit(
                    token, str(path), final, True, total, total,
                )
            except Exception as e:
                self._index_failed.emit(token, str(e))

        threading.Thread(target=work, daemon=True).start()

    def _on_groups_updated(self, token: int, path_str: str,
                           groups: list, done: bool,
                           page_num: int, total: int) -> None:
        if token != self._index_token:
            return
        path = Path(path_str)
        if not self.doc:
            # Still waiting on the first complete packet. Update the
            # indexing-progress hint and bootstrap if we got one.
            if groups:
                self._bootstrap_editor(path, groups)
            elif done:
                self.setWindowTitle(f"Template editor — {path.name}")
                self.page_label.setText("")
                self._set_busy(False)
                QMessageBox.critical(
                    self, "No students found",
                    "Could not decode any QR codes in this PDF.",
                )
            else:
                self.page_label.setText(
                    f"Indexing page {page_num} of {total}…"
                )
            return
        # Editor is already live — merge in any newly complete students
        # (or legacy 2-segment groups arriving on the final emit) and
        # update the reference combo without disturbing the user's
        # current selection.
        if len(groups) > len(self.groups) or done:
            current_name = (
                self.groups[self.current_group_idx].folder_name
                if self.groups and self.current_group_idx < len(self.groups)
                else None
            )
            self.groups = groups
            if current_name is not None:
                for i, g in enumerate(self.groups):
                    if g.folder_name == current_name:
                        self.current_group_idx = i
                        break
            self._refresh_student_combo()
        if done:
            self.setWindowTitle(f"Template editor — {path.name}")
        else:
            self.setWindowTitle(
                f"Template editor — {path.name}  "
                f"(indexing {page_num}/{total}…)"
            )

    def _bootstrap_editor(self, path: Path, groups: list) -> None:
        """First-page-ready handoff: open the rendering doc, render the
        reference packet, and let the user start drawing rectangles. Runs
        on the UI thread."""
        self.groups = groups
        self.doc = pymupdf.open(str(path))
        self.pdf_path = path
        self.current_group_idx = 0
        self.current_page_idx = 0
        self.regions.clear()
        self.mc_pages.clear()
        self.next_q_num = 1
        self._refresh_student_combo()
        self._refresh_region_list()
        self._set_busy(False)
        self._render_current_page()

    def _refresh_student_combo(self) -> None:
        self.student_combo.blockSignals(True)
        self.student_combo.clear()
        self.student_combo.addItems([g.folder_name for g in self.groups])
        if 0 <= self.current_group_idx < len(self.groups):
            self.student_combo.setCurrentIndex(self.current_group_idx)
        self.student_combo.blockSignals(False)

    def _on_index_failed(self, token: int, error: str) -> None:
        if token != self._index_token:
            return
        self.setWindowTitle(f"Template editor — {self.pdf_path.name}")
        self.page_label.setText("")
        self._set_busy(False)
        QMessageBox.critical(self, "Indexing failed", error)

    # ---------- State accessors ----------

    def _current_group(self) -> StudentGroup:
        return self.groups[self.current_group_idx]

    def _current_pdf_page_number(self) -> int:
        return self._current_group().pages[self.current_page_idx].pdf_page_number

    def _current_page_in_packet(self) -> int:
        return self.current_page_idx + 1

    # ---------- Rendering ----------

    def _render_current_page(self) -> None:
        if not self.groups or self.doc is None:
            return
        pdf_pg = self._current_pdf_page_number()
        page = self.doc[pdf_pg - 1]
        zoom = DISPLAY_DPI / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Copy through QImage so the QPixmap doesn't reference pix.samples after pix is freed.
        qimg = QImage(arr.data, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        page_in_packet = self._current_page_in_packet()
        is_mc_page = page_in_packet in self.mc_pages
        regions_for_page = (
            [] if is_mc_page
            else [r for r in self.regions if r["page"] == page_in_packet]
        )
        self.page_view.set_page(pixmap, regions_for_page)

        n_pages = len(self._current_group().pages)
        suffix = "  — MC page (whole-page capture)" if is_mc_page else ""
        self.page_label.setText(
            f"Packet page {page_in_packet} of {n_pages}  (PDF page {pdf_pg}){suffix}"
        )
        self.defining_label.setText(
            "MC (whole page)" if is_mc_page else _code_for(self.next_q_num)
        )
        # Reflect the current page's MC state without re-firing the toggle.
        self.mc_toggle_btn.blockSignals(True)
        self.mc_toggle_btn.setChecked(is_mc_page)
        self.mc_toggle_btn.blockSignals(False)

    # ---------- Region handling ----------

    def _on_rect_drawn(self, nx0: float, ny0: float, nx1: float, ny1: float) -> None:
        page_in_packet = self._current_page_in_packet()
        if page_in_packet in self.mc_pages:
            # MC pages are whole-page captures — bboxes don't apply here.
            return
        self.regions.append({
            "q": _code_for(self.next_q_num),
            "page": page_in_packet,
            "bbox": [round(nx0, 4), round(ny0, 4), round(nx1, 4), round(ny1, 4)],
        })
        self.next_q_num += 1
        self._refresh_region_list()
        self._render_current_page()

    def _toggle_mc_page(self, checked: bool) -> None:
        if not self.groups:
            return
        page_in_packet = self._current_page_in_packet()
        if checked:
            # Marking as MC drops any bboxes the user may already have
            # drawn on this page (whole-page capture supersedes regions),
            # then renumbers so question codes stay contiguous.
            before = len(self.regions)
            self.regions = [r for r in self.regions if r["page"] != page_in_packet]
            if len(self.regions) != before:
                self._renumber()
                self._refresh_region_list()
            self.mc_pages.add(page_in_packet)
        else:
            self.mc_pages.discard(page_in_packet)
        self._render_current_page()

    def _refresh_region_list(self) -> None:
        self.region_list.clear()
        for r in self.regions:
            self.region_list.addItem(f"{r['q']}  page {r['page']}")

    def _renumber(self) -> None:
        for i, r in enumerate(self.regions, start=1):
            r["q"] = _code_for(i)
        self.next_q_num = len(self.regions) + 1

    def _delete_selected(self) -> None:
        row = self.region_list.currentRow()
        if row < 0:
            return
        del self.regions[row]
        self._renumber()
        self._refresh_region_list()
        self._render_current_page()

    # ---------- Navigation ----------

    def _on_student_change(self, idx: int) -> None:
        if idx < 0:
            return
        self.current_group_idx = idx
        self.current_page_idx = 0
        self._render_current_page()

    def _prev_page(self) -> None:
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self._render_current_page()

    def _next_page(self) -> None:
        if self.current_page_idx < len(self._current_group().pages) - 1:
            self.current_page_idx += 1
            self._render_current_page()

    # ---------- Save ----------

    def _save(self) -> None:
        if not self.regions and not self.mc_pages:
            QMessageBox.warning(
                self, "Nothing to save",
                "Define some regions or mark at least one page as MC first.",
            )
            return
        pages_per_student = len(self._current_group().pages)
        initial_dir = _writable_templates_dir(self.pdf_path)
        initial_path = str(initial_dir / f"{self.pdf_path.stem}.yaml")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save template", initial_path, "YAML (*.yaml)"
        )
        if not path:
            return
        # Question codes may not start with 'Q' once QMARK_QUESTION_CODES
        # is in play (e.g. "Part_B_Q1"), so sort lexicographically on the
        # whole code rather than parsing an int suffix.
        regions_sorted = sorted(self.regions, key=lambda r: str(r["q"]))
        mc_pages_sorted = sorted(self.mc_pages)
        template = {
            "exam": self.pdf_path.stem,
            "reference_student": self._current_group().folder_name,
            "pages_per_student": pages_per_student,
            "questions": [
                {"q": r["q"], "page": r["page"], "bbox": r["bbox"]}
                for r in regions_sorted
            ],
        }
        if mc_pages_sorted:
            template["mc_pages"] = mc_pages_sorted
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(template, f, sort_keys=False)
        mc_suffix = (
            f" + {len(mc_pages_sorted)} MC page(s)" if mc_pages_sorted else ""
        )
        QMessageBox.information(
            self, "Saved",
            f"Wrote {len(regions_sorted)} regions{mc_suffix} to:\n{path}",
        )

    # ---------- Resize hook (re-render so scaling stays sharp) ----------

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        # PageView already redraws via its paintEvent on resize; nothing else to do.


def main() -> None:
    app = QApplication(sys.argv)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    if len(sys.argv) >= 2:
        pdf_path = Path(sys.argv[1])
    else:
        chosen, _ = QFileDialog.getOpenFileName(None, "Open PDF", "", "PDF (*.pdf)")
        if not chosen:
            sys.exit(0)
        pdf_path = Path(chosen)

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    win = TemplateEditor(pdf_path)
    win.resize(1280, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
