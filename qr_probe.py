"""Probe script: render each page of a PDF and report any QR codes found."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf


def render_page(page: pymupdf.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def decode_qrs(img: np.ndarray) -> list[str]:
    detector = cv2.QRCodeDetector()
    ok, decoded, _points, _ = detector.detectAndDecodeMulti(img)
    if not ok:
        return []
    return [d for d in decoded if d]


def probe(pdf_path: Path, dpi: int = 250) -> None:
    doc = pymupdf.open(pdf_path)
    print(f"Opened {pdf_path.name}  pages={len(doc)}  render_dpi={dpi}")
    print("-" * 70)

    total_found = 0
    for page_num, page in enumerate(doc, start=1):
        img = render_page(page, dpi)
        qrs = decode_qrs(img)
        if qrs:
            total_found += len(qrs)
            for q in qrs:
                print(f"  page {page_num:>3}  QR: {q!r}")
        else:
            print(f"  page {page_num:>3}  (no QR detected)")

    doc.close()
    print("-" * 70)
    print(f"Total QR codes decoded: {total_found}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python qr_probe.py <path-to-pdf>")
        sys.exit(2)
    probe(Path(sys.argv[1]))
