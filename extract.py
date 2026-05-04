"""Apply a template YAML to a scan PDF, output per-question crops per student.

Usage:
    python extract.py <pdf> <template.yaml> <output_dir> [--dpi 300]

Output layout:
    <output_dir>/<exam>/manifest.csv
    <output_dir>/<exam>/<class>_<firstname>/Q01.png ... QNN.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import yaml

from scan_index import StudentGroup, group_into_students, index_pdf

EXTRACT_DPI = 300


def render_page(doc: pymupdf.Document, pdf_page_number: int, dpi: int) -> np.ndarray:
    page = doc[pdf_page_number - 1]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def crop_region(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    px0 = max(0, int(round(x0 * w)))
    py0 = max(0, int(round(y0 * h)))
    px1 = min(w, int(round(x1 * w)))
    py1 = min(h, int(round(y1 * h)))
    return img[py0:py1, px0:px1]


def extract(pdf_path: Path, template_path: Path, output_dir: Path, dpi: int) -> None:
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    pages_per_student: int = template["pages_per_student"]
    questions: list[dict] = template["questions"]
    exam_name: str = template.get("exam", pdf_path.stem)

    template_pages = sorted({q["page"] for q in questions})
    if template_pages and max(template_pages) > pages_per_student:
        sys.exit(
            f"Template references packet page {max(template_pages)} but "
            f"pages_per_student is {pages_per_student}"
        )

    qs_by_page: dict[int, list[dict]] = {}
    for q in questions:
        qs_by_page.setdefault(q["page"], []).append(q)

    print(f"Indexing {pdf_path.name}...")
    pages = index_pdf(pdf_path)
    groups = group_into_students(pages)
    print(f"  {len(groups)} student groups, {sum(len(g.pages) for g in groups)} pages")

    exam_out = output_dir / exam_name
    exam_out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)

    manifest_rows: list[dict] = []
    print(f"\nExtracting {len(questions)} questions per student at {dpi} DPI...")
    for group in groups:
        row = _extract_one_student(doc, group, qs_by_page, exam_out, pages_per_student, dpi)
        manifest_rows.append(row)
        marker = "OK  " if row["notes"] == "" else "WARN"
        print(f"  [{marker}] {group.folder_name:<28} {row['n_questions_extracted']:>3} crops  {row['notes']}")

    doc.close()

    manifest_path = exam_out / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "student_class", "student_name",
            "packet_pdf_pages", "qr_status_per_page",
            "n_questions_extracted", "notes",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. Output: {exam_out}")
    print(f"Manifest:    {manifest_path}")


def _extract_one_student(
    doc: pymupdf.Document,
    group: StudentGroup,
    qs_by_page: dict[int, list[dict]],
    exam_out: Path,
    pages_per_student: int,
    dpi: int,
) -> dict:
    base_row = {
        "student_class": group.student_class,
        "student_name": group.student_name,
        "packet_pdf_pages": ",".join(str(p.pdf_page_number) for p in group.pages),
        "qr_status_per_page": ",".join(p.qr_status for p in group.pages),
        "n_questions_extracted": 0,
        "notes": "",
    }

    if len(group.pages) != pages_per_student:
        base_row["notes"] = f"page count mismatch (got {len(group.pages)}, expected {pages_per_student})"
        return base_row

    student_dir = exam_out / group.folder_name
    student_dir.mkdir(parents=True, exist_ok=True)

    n_extracted = 0
    for packet_page_idx in range(1, pages_per_student + 1):
        qs = qs_by_page.get(packet_page_idx, [])
        if not qs:
            continue
        pdf_page_num = group.pages[packet_page_idx - 1].pdf_page_number
        page_img = render_page(doc, pdf_page_num, dpi)
        for q in qs:
            crop = crop_region(page_img, q["bbox"])
            out_path = student_dir / f"{q['q']}.png"
            cv2.imwrite(str(out_path), crop)
            n_extracted += 1

    base_row["n_questions_extracted"] = n_extracted
    return base_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-question crops from a scan PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the scan PDF")
    parser.add_argument("template", type=Path, help="Path to the template YAML")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--dpi", type=int, default=EXTRACT_DPI, help=f"Render DPI (default {EXTRACT_DPI})")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")
    if not args.template.exists():
        sys.exit(f"Template not found: {args.template}")

    extract(args.pdf, args.template, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
