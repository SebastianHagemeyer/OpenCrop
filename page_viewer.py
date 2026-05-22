"""Modal preview of every PDF page belonging to one student.

Triggered from the scan-result roster (right-click → "View pages..." or
double-click a row). Useful for verifying a recovered student is the
right person, or for spotting an obvious mis-assignment before extract.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# Bigger than the orphan dialog's thumbnail — this view is for
# verifying the student matches, so text and handwriting should be
# easy to read. Still rendered at a low DPI so opening it on a
# 6-page packet is instant rather than a 2-second render.
_PAGE_W = 320
_PAGE_DPI = 80


def _render_page(pdf_doc: pymupdf.Document, page_number: int, target_w: int = _PAGE_W) -> QPixmap:
    page = pdf_doc[page_number - 1]
    zoom = _PAGE_DPI / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = QImage(pix.samples, pix.width, pix.height, pix.stride,
                 QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaledToWidth(target_w, Qt.SmoothTransformation)


class PageViewerDialog(QDialog):
    """Show the PDF pages for one student side by side."""

    def __init__(self, pdf_path: Path, page_numbers: list[int],
                 title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{title} — pages")
        self.resize(min(380 * len(page_numbers) + 60, 1280), 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        header = QLabel(
            f"<b>{title}</b> &nbsp; · &nbsp; "
            f"{len(page_numbers)} page{'s' if len(page_numbers) != 1 else ''} "
            f"(PDF page{'s' if len(page_numbers) != 1 else ''} "
            f"{', '.join(str(p) for p in page_numbers)})"
        )
        outer.addWidget(header)

        # Render once and lay out in a scrollable horizontal strip so a
        # 6-page packet fits without resizing the window.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        row = QHBoxLayout(inner)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(12)

        doc = pymupdf.open(pdf_path)
        try:
            for pn in page_numbers:
                pm = _render_page(doc, pn)
                col = QVBoxLayout()
                col.setSpacing(4)
                lbl = QLabel()
                lbl.setPixmap(pm)
                lbl.setFixedSize(pm.size())
                lbl.setStyleSheet("border: 1px solid palette(mid);")
                col.addWidget(lbl)
                caption = QLabel(f"<i>PDF page {pn}</i>")
                caption.setAlignment(Qt.AlignHCenter)
                col.addWidget(caption)
                wrap = QWidget()
                wrap.setLayout(col)
                row.addWidget(wrap)
            row.addStretch(1)
        finally:
            doc.close()

        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)
