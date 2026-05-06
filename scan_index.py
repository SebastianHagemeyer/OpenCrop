"""Parse a scanned exam PDF: decode per-page QR codes and group pages into students.

QR formats accepted:
  * 4-segment (preferred):  ``<class>/<firstname>/<page>/<total>``  (e.g. ``10MATD/Dj/1/2``)
  * 2-segment (legacy):     ``<class>/<firstname>``                (e.g. ``10MATD/William``)

The 4-segment QR explicitly identifies which student a page belongs to and the
page's position in that student's packet. The 2-segment legacy QR carries only
class + first name, so packet position is reconstructed from the order in which
the pages appear within their group. Mixing both formats in a single PDF is
supported (useful when different classes were printed at different times).

When QR decoding fails entirely, the page is assigned by inference from its
nearest decoded neighbour: e.g. a missing page immediately followed by ``X/2/2``
is inferred to be ``X/1/2``. Pages with no usable neighbour stay ``unknown``
and surface in the manifest.
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
    page_in_packet: int | None    # 1-based, from QR
    pages_total: int | None       # total pages in the student's packet
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


def _parse_qr(text: str) -> tuple[str, str, int | None, int | None] | None:
    """Parse a QR payload into (class, name, page, total).

    Accepts both the 4-segment and the legacy 2-segment formats. For the legacy
    form, page/total are returned as None and reconstructed later from the
    group's page order.
    """
    parts = [p.strip() for p in text.split("/")]
    if len(parts) == 2:
        cls, name = parts
        if not cls or not name:
            return None
        return cls, name, None, None
    if len(parts) == 4:
        cls, name, page_str, total_str = parts
        if not cls or not name:
            return None
        try:
            page = int(page_str)
            total = int(total_str)
        except ValueError:
            return None
        if page < 1 or total < 1 or page > total:
            return None
        return cls, name, page, total
    return None


def index_pdf(pdf_path: Path, dpi: int = 250) -> list[PageRecord]:
    """Decode the QR on every page of the PDF."""
    doc = pymupdf.open(pdf_path)
    out: list[PageRecord] = []
    for page_num, page in enumerate(doc, start=1):
        img = _render_page(page, dpi)
        text, status = _decode_qr(img)
        cls, name, pip, tot = (None, None, None, None)
        if text:
            parsed = _parse_qr(text)
            if parsed:
                cls, name, pip, tot = parsed
            else:
                status = "unknown"  # decoded something but format didn't match
        out.append(PageRecord(page_num, cls, name, pip, tot, text, status))
    doc.close()
    return out


def _infer_missing(pages: list[PageRecord]) -> None:
    """Fill in student + page-in-packet for undecoded pages, using the nearest
    decoded neighbour. Mutates pages in place. Pages with no usable neighbour
    stay unknown."""
    n = len(pages)
    for i, p in enumerate(pages):
        if p.student_name is not None:
            continue

        candidates: list[tuple[int, PageRecord, int]] = []  # (offset, neighbour, inferred_page)

        for j in range(i + 1, n):
            nb = pages[j]
            if nb.student_name is None or nb.page_in_packet is None:
                continue
            offset = j - i
            cand = nb.page_in_packet - offset
            if 1 <= cand <= nb.pages_total:
                candidates.append((offset, nb, cand))
            break

        for j in range(i - 1, -1, -1):
            nb = pages[j]
            if nb.student_name is None or nb.page_in_packet is None:
                continue
            offset = i - j
            cand = nb.page_in_packet + offset
            if 1 <= cand <= nb.pages_total:
                candidates.append((offset, nb, cand))
            break

        if not candidates:
            continue

        candidates.sort(key=lambda c: c[0])
        _, nb, cand = candidates[0]
        p.student_class = nb.student_class
        p.student_name = nb.student_name
        p.page_in_packet = cand
        p.pages_total = nb.pages_total
        p.qr_status = "inferred"


def group_into_students(pages: list[PageRecord]) -> list[StudentGroup]:
    _infer_missing(pages)

    groups: list[StudentGroup] = []
    seen: dict[tuple[str, str], StudentGroup] = {}
    for p in pages:
        if p.student_name is None:
            g = StudentGroup("UNKNOWN", f"orphan_p{p.pdf_page_number}")
            groups.append(g)
            g.pages.append(p)
            continue

        key = (p.student_class, p.student_name)
        g = seen.get(key)
        if g is None:
            g = StudentGroup(p.student_class, p.student_name)
            seen[key] = g
            groups.append(g)
        g.pages.append(p)

    for g in groups:
        g.pages.sort(key=lambda r: (r.page_in_packet if r.page_in_packet is not None else 999, r.pdf_page_number))

    # Reconstruct page-in-packet/pages-total for legacy 2-segment QR pages,
    # which carry no explicit position. Order is the PDF order within the group
    # (preserved by the sort above, since None sorts last as 999 and ties break
    # on pdf_page_number).
    for g in groups:
        legacy_pages = [p for p in g.pages if p.page_in_packet is None and p.qr_status != "unknown"]
        if not legacy_pages:
            continue
        if any(p.page_in_packet is not None for p in g.pages):
            # Mixed: some pages in this group already have explicit positions.
            # Refuse to guess — leave the legacy ones as-is so the manifest
            # surfaces the inconsistency.
            continue
        total = len(legacy_pages)
        for idx, p in enumerate(legacy_pages, start=1):
            p.page_in_packet = idx
            p.pages_total = total

    return groups


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
