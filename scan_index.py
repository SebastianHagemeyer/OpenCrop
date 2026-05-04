"""Parse a scanned exam PDF: decode per-page QR codes and group pages into students.

QR format expected: ``<class>/<firstname>``  (e.g. ``10MATD/Ruby``).

When QR decoding fails on a page, the page is attributed to the most recent
student. After grouping, group sizes are rebalanced: if the typical group has
N pages and a group has N+k while the next has N-k, the trailing inferred pages
are moved forward. This recovers the common "first page of next student
failed to decode" pattern.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf


@dataclass
class PageRecord:
    pdf_page_number: int          # 1-based
    student_class: str | None
    student_name: str | None
    qr_raw: str | None
    qr_status: str                # "decoded" | "preprocessed" | "inferred" | "unknown"


@dataclass
class StudentGroup:
    student_class: str
    student_name: str
    pages: list[PageRecord] = field(default_factory=list)

    @property
    def folder_name(self) -> str:
        return f"{self.student_class}_{self.student_name}"


def _render_page(page: pymupdf.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _try_detect(img: np.ndarray) -> str | None:
    detector = cv2.QRCodeDetector()
    ok, decoded, _points, _ = detector.detectAndDecodeMulti(img)
    if not ok:
        return None
    for d in decoded:
        if d:
            return d
    return None


def _decode_qr(img: np.ndarray) -> tuple[str | None, str]:
    """Return (decoded_text, status). Status is decoded|preprocessed|unknown."""
    text = _try_detect(img)
    if text:
        return text, "decoded"

    # Otsu threshold to handle low-contrast / faded prints
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = _try_detect(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
    if text:
        return text, "preprocessed"

    # Try a few resamples — small QRs sometimes detect better when upscaled,
    # large ones when downscaled.
    h, w = img.shape[:2]
    for scale in (1.5, 0.75, 0.5, 2.0):
        scaled = cv2.resize(img, (int(w * scale), int(h * scale)))
        text = _try_detect(scaled)
        if text:
            return text, "preprocessed"

    return None, "unknown"


def _parse_qr(text: str) -> tuple[str, str] | None:
    if "/" not in text:
        return None
    cls, _, name = text.partition("/")
    cls, name = cls.strip(), name.strip()
    if not cls or not name:
        return None
    return cls, name


def index_pdf(pdf_path: Path, dpi: int = 250) -> list[PageRecord]:
    """Decode the QR on every page of the PDF."""
    doc = pymupdf.open(pdf_path)
    out: list[PageRecord] = []
    for page_num, page in enumerate(doc, start=1):
        img = _render_page(page, dpi)
        text, status = _decode_qr(img)
        cls, name = (None, None)
        if text:
            parsed = _parse_qr(text)
            if parsed:
                cls, name = parsed
            else:
                status = "unknown"  # decoded something but format didn't match
        out.append(PageRecord(page_num, cls, name, text, status))
    doc.close()
    return out


def _group_consecutive(pages: list[PageRecord]) -> list[StudentGroup]:
    groups: list[StudentGroup] = []
    current: StudentGroup | None = None
    for p in pages:
        if p.student_name is not None:
            new_student = (
                current is None
                or p.student_name != current.student_name
                or p.student_class != current.student_class
            )
            if new_student:
                current = StudentGroup(p.student_class, p.student_name)
                groups.append(current)
            current.pages.append(p)
        else:
            # Attribute missing-QR pages to the running student; rebalancing
            # may move them later.
            if current is None:
                current = StudentGroup("UNKNOWN", f"orphan_p{p.pdf_page_number}")
                groups.append(current)
            inferred = PageRecord(
                pdf_page_number=p.pdf_page_number,
                student_class=current.student_class,
                student_name=current.student_name,
                qr_raw=p.qr_raw,
                qr_status="inferred",
            )
            current.pages.append(inferred)
    return groups


def _rebalance(groups: list[StudentGroup]) -> list[StudentGroup]:
    """Move trailing inferred pages from oversized groups to undersized
    next-neighbours, using the modal group size as the expected packet length."""
    if len(groups) < 2:
        return groups

    sizes = [len(g.pages) for g in groups]
    # Mode of sizes — if everything is the same, no rebalancing needed.
    expected = max(set(sizes), key=sizes.count)

    for i in range(len(groups) - 1):
        cur, nxt = groups[i], groups[i + 1]
        while len(cur.pages) > expected and len(nxt.pages) < expected and cur.pages[-1].qr_status == "inferred":
            moved = cur.pages.pop()
            moved.student_class = nxt.student_class
            moved.student_name = nxt.student_name
            # Status stays "inferred"
            nxt.pages.insert(0, moved)
    return groups


def group_into_students(pages: list[PageRecord]) -> list[StudentGroup]:
    return _rebalance(_group_consecutive(pages))


def _print_summary(groups: list[StudentGroup]) -> None:
    print(f"{'group':<28} {'n':>3}  {'pdf pages':<25}  status")
    print("-" * 78)
    for g in groups:
        pdf_pgs = ",".join(str(p.pdf_page_number) for p in g.pages)
        statuses = ",".join(p.qr_status[:4] for p in g.pages)
        print(f"{g.folder_name:<28} {len(g.pages):>3}  {pdf_pgs:<25}  {statuses}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scan_index.py <path-to-pdf>")
        sys.exit(2)
    pdf_path = Path(sys.argv[1])
    pages = index_pdf(pdf_path)
    groups = group_into_students(pages)
    print(f"Indexed {len(pages)} pages -> {len(groups)} student groups\n")
    _print_summary(groups)
