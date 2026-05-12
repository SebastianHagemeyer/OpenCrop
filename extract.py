"""Apply a template YAML to a scan PDF, output per-question crops per student.

Usage:
    python extract.py <pdf> <template.yaml> <output_dir> [--dpi 300]
        [--sheet-pdf <blank_sheet.pdf>]

Output layout:
    <output_dir>/<exam>/manifest.csv
    <output_dir>/<exam>/<class>_<firstname>/Q01.png ... QNN.png

When a blank sheet PDF is supplied (--sheet-pdf, or QMARK_SHEET_PATH env var),
the same template is also applied to the unstamped worksheet to produce
reference crops under <output>/<exam>/_blank/<Q>.png. Each student crop is
then compared against its blank reference by ink-density delta, and the
verdict per (student, question) is written to <output>/<exam>/attempts.csv.
The marker UI can use attempts.csv to grey out / colour questions that
were left empty.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import yaml

from scan_index import StudentGroup, group_into_students, index_pdf

EXTRACT_DPI = 300

# Tunable thresholds for attempt detection. The "ink delta" is the fraction
# of dark pixels in the student crop minus the same in the blank reference.
# Below DELTA_UNATTEMPTED -> unattempted; below DELTA_BORDERLINE -> borderline
# (worth a human re-check); else attempted. INK_THRESHOLD is the grayscale
# value below which a pixel counts as ink — pen on paper sits well below 180
# after scanning, scan noise stays above it.
DELTA_UNATTEMPTED = 0.005   # 0.5%
DELTA_BORDERLINE = 0.020    # 2.0%
INK_THRESHOLD = 180

BLANK_DIR_NAME = "_blank"
ATTEMPTS_CSV_NAME = "attempts.csv"


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


def _ink_ratio(crop: np.ndarray) -> float:
    """Fraction of pixels darker than INK_THRESHOLD. Size-independent so
    student/blank crops at the same DPI are directly comparable."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    dark = int(np.count_nonzero(gray < INK_THRESHOLD))
    return dark / float(gray.size)


def _classify_attempt(student_ratio: float, blank_ratio: float) -> str:
    delta = student_ratio - blank_ratio
    if delta < DELTA_UNATTEMPTED:
        return "unattempted"
    if delta < DELTA_BORDERLINE:
        return "borderline"
    return "attempted"


def _render_blank_references(
    sheet_pdf: Path,
    questions: list[dict],
    pages_per_student: int,
    blank_out: Path,
    dpi: int,
) -> dict[str, float]:
    """Render the unstamped sheet PDF through the template, write each Q's
    blank crop to <blank_out>/<Q>.png, return ink ratio keyed by question
    code. Returns empty dict (and logs a warning) when the sheet PDF is
    missing or shorter than the packet — caller treats absent Qs as unknown.
    """
    if not sheet_pdf.exists():
        print(f"  blank reference: sheet PDF not found at {sheet_pdf}; skipping")
        return {}

    qs_by_page: dict[int, list[dict]] = {}
    for q in questions:
        qs_by_page.setdefault(q["page"], []).append(q)

    blank_out.mkdir(parents=True, exist_ok=True)
    ratios: dict[str, float] = {}

    doc = pymupdf.open(sheet_pdf)
    try:
        sheet_pages = doc.page_count
        if sheet_pages < pages_per_student:
            print(
                f"  blank reference: sheet PDF has {sheet_pages} page(s) but "
                f"template expects {pages_per_student} per packet; comparing what overlaps"
            )

        max_page = min(pages_per_student, sheet_pages)
        for page_idx in range(1, max_page + 1):
            qs = qs_by_page.get(page_idx, [])
            if not qs:
                continue
            page_img = render_page(doc, page_idx, dpi)
            for q in qs:
                crop = crop_region(page_img, q["bbox"])
                cv2.imwrite(str(blank_out / f"{q['q']}.png"), crop)
                ratios[q["q"]] = _ink_ratio(crop)
    finally:
        doc.close()

    print(f"  blank reference: wrote {len(ratios)} crop(s) to {blank_out}")
    return ratios


def extract(
    pdf_path: Path,
    template_path: Path,
    output_dir: Path,
    dpi: int,
    exam_name_override: str | None = None,
    sheet_pdf: Path | None = None,
) -> None:
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    pages_per_student: int = template["pages_per_student"]
    questions: list[dict] = template["questions"]
    exam_name: str = exam_name_override or template.get("exam", pdf_path.stem)

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

    blank_ratios: dict[str, float] = {}
    if sheet_pdf is not None:
        blank_ratios = _render_blank_references(
            sheet_pdf, questions, pages_per_student, exam_out / BLANK_DIR_NAME, dpi,
        )

    doc = pymupdf.open(pdf_path)

    manifest_rows: list[dict] = []
    attempts_rows: list[dict] = []
    print(f"\nExtracting {len(questions)} questions per student at {dpi} DPI...")
    for group in groups:
        row, student_attempts = _extract_one_student(
            doc, group, qs_by_page, exam_out, pages_per_student, dpi,
            blank_ratios=blank_ratios,
        )
        manifest_rows.append(row)
        attempts_rows.extend(student_attempts)
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

    if blank_ratios:
        attempts_path = exam_out / ATTEMPTS_CSV_NAME
        with attempts_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "student_folder", "q", "status",
                "blank_ink_ratio", "student_ink_ratio", "delta",
            ])
            writer.writeheader()
            writer.writerows(attempts_rows)
        n_unatt = sum(1 for r in attempts_rows if r["status"] == "unattempted")
        n_bord = sum(1 for r in attempts_rows if r["status"] == "borderline")
        print(f"\nAttempt detection: {len(attempts_rows)} crops scored — "
              f"{n_unatt} unattempted, {n_bord} borderline. Written to {attempts_path}")

    print(f"\nDone. Output: {exam_out}")
    print(f"Manifest:    {manifest_path}")


def _extract_one_student(
    doc: pymupdf.Document,
    group: StudentGroup,
    qs_by_page: dict[int, list[dict]],
    exam_out: Path,
    pages_per_student: int,
    dpi: int,
    blank_ratios: dict[str, float] | None = None,
) -> tuple[dict, list[dict]]:
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
        return base_row, []

    student_dir = exam_out / group.folder_name
    student_dir.mkdir(parents=True, exist_ok=True)

    blank_ratios = blank_ratios or {}
    n_extracted = 0
    attempts: list[dict] = []
    for packet_page_idx in range(1, pages_per_student + 1):
        qs = qs_by_page.get(packet_page_idx, [])
        if not qs:
            continue
        pdf_page_num = group.pages[packet_page_idx - 1].pdf_page_number
        page_img = render_page(doc, pdf_page_num, dpi)
        for q in qs:
            crop = crop_region(page_img, q["bbox"])
            cv2.imwrite(str(student_dir / f"{q['q']}.png"), crop)
            n_extracted += 1
            if not blank_ratios:
                continue
            q_code = q["q"]
            if q_code not in blank_ratios:
                attempts.append({
                    "student_folder": group.folder_name, "q": q_code,
                    "status": "unknown",
                    "blank_ink_ratio": "", "student_ink_ratio": "", "delta": "",
                })
                continue
            student_r = _ink_ratio(crop)
            blank_r = blank_ratios[q_code]
            attempts.append({
                "student_folder": group.folder_name, "q": q_code,
                "status": _classify_attempt(student_r, blank_r),
                "blank_ink_ratio": f"{blank_r:.4f}",
                "student_ink_ratio": f"{student_r:.4f}",
                "delta": f"{student_r - blank_r:+.4f}",
            })

    base_row["n_questions_extracted"] = n_extracted
    return base_row, attempts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-question crops from a scan PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the scan PDF")
    parser.add_argument("template", type=Path, help="Path to the template YAML")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--dpi", type=int, default=EXTRACT_DPI, help=f"Render DPI (default {EXTRACT_DPI})")
    parser.add_argument(
        "--exam-name",
        default=None,
        help="Override the output subfolder name (defaults to template['exam'] or PDF stem)",
    )
    parser.add_argument(
        "--sheet-pdf",
        type=Path,
        default=None,
        help=(
            "Blank worksheet PDF — when supplied, the same template is applied "
            "to it to produce reference crops under <output>/<exam>/_blank/ and "
            "an attempts.csv classifying each student crop as attempted, "
            "unattempted, or borderline. Defaults to env QMARK_SHEET_PATH."
        ),
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")
    if not args.template.exists():
        sys.exit(f"Template not found: {args.template}")

    sheet_pdf = args.sheet_pdf
    if sheet_pdf is None:
        env_sheet = os.environ.get("QMARK_SHEET_PATH", "").strip()
        if env_sheet:
            sheet_pdf = Path(env_sheet)
    if sheet_pdf is not None and not sheet_pdf.exists():
        print(f"WARNING: sheet PDF {sheet_pdf} not found; skipping attempt detection")
        sheet_pdf = None

    extract(
        args.pdf, args.template, args.output, dpi=args.dpi,
        exam_name_override=args.exam_name, sheet_pdf=sheet_pdf,
    )


if __name__ == "__main__":
    main()
