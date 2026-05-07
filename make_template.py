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

import sys
from pathlib import Path

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

from scan_index import StudentGroup, group_into_students, index_pdf

INDEX_DPI = 200       # QR scan dpi (lower = faster startup)
DISPLAY_DPI = 180     # rendering dpi for the on-screen image
MIN_BBOX = 0.01       # ignore drags smaller than 1% of page (likely accidental)
ICON_PATH = Path(__file__).resolve().parent / "paper.ico"


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

    def set_page(self, pixmap: QPixmap, regions: list[dict]) -> None:
        self._pixmap = pixmap
        self._regions = regions
        self._draw_start = None
        self._draw_end = None
        self.update()

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
        painter.drawPixmap(0, 0, self._pixmap.scaled(
            sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        ))

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
        self.next_q_num = 1

        self._build_ui()
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
        self.defining_label = QLabel("Q01")
        big = QFont()
        big.setPointSize(14)
        big.setBold(True)
        self.defining_label.setFont(big)
        side_lay.addWidget(self.defining_label)

        side_lay.addSpacing(8)
        side_lay.addWidget(QLabel("Regions defined:"))
        self.region_list = QListWidget()
        side_lay.addWidget(self.region_list, 1)
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._delete_selected)
        side_lay.addWidget(del_btn)

        side_lay.addSpacing(8)
        nav = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev page")
        prev_btn.clicked.connect(self._prev_page)
        next_btn = QPushButton("Next page ▶")
        next_btn.clicked.connect(self._next_page)
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)
        side_lay.addLayout(nav)

        save_btn = QPushButton("Save template…")
        save_btn.clicked.connect(self._save)
        side_lay.addWidget(save_btn)

        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._prev_page)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._next_page)

    # ---------- PDF loading / indexing ----------

    def _open_pdf_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF (*.pdf)")
        if not path:
            return
        self._load_pdf(Path(path))

    def _load_pdf(self, path: Path) -> None:
        self.setWindowTitle(f"Template editor — {path.name}  (indexing…)")
        QApplication.processEvents()
        if self.doc is not None:
            self.doc.close()
        self.doc = pymupdf.open(path)
        pages = index_pdf(path, dpi=INDEX_DPI)
        self.groups = [g for g in group_into_students(pages) if g.student_class != "UNKNOWN"]
        if not self.groups:
            QMessageBox.critical(self, "No students found", "Could not decode any QR codes in this PDF.")
            return
        self.pdf_path = path
        self.current_group_idx = 0
        self.current_page_idx = 0
        self.regions.clear()
        self.next_q_num = 1
        self.student_combo.blockSignals(True)
        self.student_combo.clear()
        self.student_combo.addItems([g.folder_name for g in self.groups])
        self.student_combo.setCurrentIndex(0)
        self.student_combo.blockSignals(False)
        self._refresh_region_list()
        self._render_current_page()
        self.setWindowTitle(f"Template editor — {path.name}")

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
        regions_for_page = [r for r in self.regions if r["page"] == page_in_packet]
        self.page_view.set_page(pixmap, regions_for_page)

        n_pages = len(self._current_group().pages)
        self.page_label.setText(
            f"Packet page {page_in_packet} of {n_pages}  (PDF page {pdf_pg})"
        )
        self.defining_label.setText(f"Q{self.next_q_num:02d}")

    # ---------- Region handling ----------

    def _on_rect_drawn(self, nx0: float, ny0: float, nx1: float, ny1: float) -> None:
        self.regions.append({
            "q": f"Q{self.next_q_num:02d}",
            "page": self._current_page_in_packet(),
            "bbox": [round(nx0, 4), round(ny0, 4), round(nx1, 4), round(ny1, 4)],
        })
        self.next_q_num += 1
        self._refresh_region_list()
        self._render_current_page()

    def _refresh_region_list(self) -> None:
        self.region_list.clear()
        for r in self.regions:
            self.region_list.addItem(f"{r['q']}  page {r['page']}")

    def _renumber(self) -> None:
        for i, r in enumerate(self.regions, start=1):
            r["q"] = f"Q{i:02d}"
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
        if not self.regions:
            QMessageBox.warning(self, "Nothing to save", "Define some regions first.")
            return
        pages_per_student = len(self._current_group().pages)
        templates_dir = self.pdf_path.parent / "templates"
        initial_dir = templates_dir if templates_dir.exists() else self.pdf_path.parent
        initial_path = str(initial_dir / f"{self.pdf_path.stem}.yaml")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save template", initial_path, "YAML (*.yaml)"
        )
        if not path:
            return
        regions_sorted = sorted(self.regions, key=lambda r: int(r["q"][1:]))
        template = {
            "exam": self.pdf_path.stem,
            "reference_student": self._current_group().folder_name,
            "pages_per_student": pages_per_student,
            "questions": [
                {"q": r["q"], "page": r["page"], "bbox": r["bbox"]}
                for r in regions_sorted
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(template, f, sort_keys=False)
        QMessageBox.information(self, "Saved", f"Wrote {len(regions_sorted)} regions to:\n{path}")

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
