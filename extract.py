"""Apply a template YAML to a scan PDF, output per-question crops per student.

Usage:
    python extract.py <pdf> <template.yaml> <output_dir> [--dpi 300]
        [--sheet-pdf <blank_sheet.pdf>] [--skip-existing]

Output layout:
    <output_dir>/<exam>/manifest.csv
    <output_dir>/<exam>/<class>_<firstname>/Q01.png ... QNN.png

manifest.csv has one row per (student, q): `student_folder, q, status`.
Status is one of: attempted, unattempted, borderline, unknown. Without a
sheet PDF every status is "unknown". With a sheet PDF the unstamped
worksheet is rendered through the same template into
<output>/<exam>/_blank/<Q>.png and each student crop is compared against
its blank reference (see _residual_ink_metrics + _classify_attempt) to
fill in real attempted/unattempted/borderline verdicts.

--skip-existing reads the prior manifest and skips any student already
in it, then appends rows for newly-scanned students. Use this to merge
later scan PDFs into the same assignment without re-extracting (or
re-overwriting) work that has already been marked.
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

# Pipeline parameters — tune here if scans look systematically different.
# The detection pipeline (binary subtraction, which cancels printed text):
#   1. Align the blank crop to the student crop via phase correlation on
#      grayscale (rich Fourier signal), capped at small shifts.
#   2. Otsu-binarize both crops: paper -> 0, ink -> 255. This normalises
#      out the intensity gap between PDF render and scanned print — the
#      printed worksheet becomes "ink" in BOTH crops and cancels out.
#   3. Dilate the aligned-blank binary by DILATE_PX so sub-pixel
#      registration error doesn't leave print-edge halos.
#   4. `student_ink AND NOT dilated_blank_ink` = ink only the student
#      added (= handwriting).
#   5. Morphological open to remove isolated speckle, then two metrics:
#      - residual_ratio: fraction of crop pixels of student-only ink.
#      - largest_blob_px: size of the biggest connected component
#        (separates a real handwriting stroke from scattered noise).
ALIGNMENT_MAX_SHIFT_PX = 50          # cap big enough to absorb whole-page
                                     # scan offsets (~15-25 px on this corpus)
                                     # without taking obvious garbage shifts
CLAHE_CLIP_LIMIT = 3.0               # contrast-limited adaptive histogram eq
CLAHE_TILE = (16, 16)                # — boosts pencil contrast for detection
ADAPTIVE_BLOCK_SIZE = 51             # adaptive-threshold neighbourhood (px)
ADAPTIVE_C = 12                      # subtract from local mean for ink cutoff
DILATE_KSIZE = 3                     # kernel for dilating aligned blank
DILATE_ITERATIONS = 2                # ~3-4 px halo around printed text
EDGE_MARGIN_PX = 6                   # ignore this many pixels from each crop
                                     # edge — scan edges + page-binding shadows
                                     # otherwise show up as student-only ink
LINE_ASPECT_THRESHOLD = 15           # drop connected components whose longer
                                     # dim is >= 15x the shorter dim — these
                                     # are box borders / underlines that weren't
                                     # perfectly cancelled by the dilated blank
LINE_MIN_LONG_DIM = 60               # only filter when the long dim is at least
                                     # this many px (so a stray 1x16 spec, which
                                     # is also high-aspect, doesn't get spared)
CLUSTER_DILATE_PX = 5                # dilate the surviving ink by this many px
                                     # before grouping into clusters — merges
                                     # adjacent digits/strokes of one answer
                                     # into a single connected component
# Colored-ink detection. PEN ink (blue, red, green, etc.) is saturated;
# printed text is essentially black (very low saturation). Pixels with
# noticeable saturation and dark-ish value are almost certainly pen ink
# and provide a clean override signal when present.
COLOR_SAT_MIN = 40                   # HSV saturation (0-255) above this = colored
COLOR_VAL_MIN = 30                   # avoid near-black noise / bleed-through
COLOR_VAL_MAX = 220                  # avoid faint paper texture
_MORPH_KERNEL = np.ones((2, 2), np.uint8)

# Classification — conservative on unattempted (both metrics must agree),
# generous on attempted (either metric is enough). Borderline is the
# overlap zone the marker UI surfaces in amber.
#
# Binary subtraction means a truly blank crop should have near-zero
# residual (only stray scan noise survives speckle removal). Real
# handwriting forms one or more connected blobs of >=500 px even for a
# single-digit answer at 300 DPI.
UNATT_MAX_LARGEST_BLOB = 600         # px (raw ink within the densest cluster
                                     # — adjacent digits of one answer get
                                     # grouped via CLUSTER_DILATE_PX merging)
UNATT_MAX_RESIDUAL_RATIO = 0.0035    # 0.35% of crop pixels (CLAHE + adaptive
                                     # thresholding bumps the noise floor a
                                     # bit; pencil detection makes up for it)
UNATT_MAX_COLOR_INK_PX = 150         # negligible coloured pixels (blank scans
                                     # cap at ~150 from JPEG colour ringing
                                     # near printed black text)
ATT_MIN_LARGEST_BLOB = 800
ATT_MIN_RESIDUAL_RATIO = 0.0040      # 0.4%
ATT_MIN_COLOR_INK_PX = 200           # any cluster of coloured pen ink above
                                     # this is almost certainly real writing

BLANK_DIR_NAME = "_blank"
ENHANCED_DIR_NAME = "_enhanced"
MANIFEST_CSV_NAME = "manifest.csv"


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


def _align_blank(
    blank_gray: np.ndarray, student_gray: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Translate the blank crop to best fit the student crop. Returns the
    aligned blank plus the (dx, dy) that was applied. Falls back to the
    unaligned input on any phase-correlation failure or on suspiciously-
    large shifts (which usually mean it locked on to noise, not content)."""
    h, w = student_gray.shape[:2]
    if blank_gray.shape[:2] != (h, w):
        blank_gray = cv2.resize(blank_gray, (w, h))
    if w < 4 or h < 4:
        return blank_gray, 0.0, 0.0
    try:
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(
            blank_gray.astype(np.float32),
            student_gray.astype(np.float32),
            win,
        )
    except cv2.error:
        return blank_gray, 0.0, 0.0
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return blank_gray, 0.0, 0.0
    if abs(dx) > ALIGNMENT_MAX_SHIFT_PX or abs(dy) > ALIGNMENT_MAX_SHIFT_PX:
        return blank_gray, 0.0, 0.0
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        blank_gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return aligned, float(dx), float(dy)


def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    """CLAHE-boost a grayscale image — stretches local contrast so pencil
    marks (~150-180 grayscale) become darker relative to paper (~230)
    without amplifying noise the way a global histogram stretch would."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE)
    return clahe.apply(gray)


def _enhance_for_display(bgr: np.ndarray) -> np.ndarray:
    """Colour-preserving CLAHE for the marker UI. Boosts the L channel in
    LAB space so faint pencil writing pops on screen, leaves hue/saturation
    alone so coloured pen ink still looks correct."""
    if bgr.ndim != 3:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=CLAHE_TILE)
    L_enh = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L_enh, A, B]), cv2.COLOR_LAB2BGR)


def _color_ink_pixels(student_crop_bgr: np.ndarray) -> tuple[int, np.ndarray]:
    """Detect coloured pen ink — saturated pixels with dark-ish value.

    Printed text is essentially black (very low saturation), so this
    signal cleanly separates blue/red/green pen writing from any
    print-cancellation residue. A clean override when the student wrote
    in colour (most teenagers do). Returns (count, binary_mask).
    """
    if student_crop_bgr.ndim != 3:
        return 0, np.zeros(student_crop_bgr.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(student_crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > COLOR_SAT_MIN)
            & (val < COLOR_VAL_MAX)
            & (val > COLOR_VAL_MIN)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    return int(mask.sum()), mask


def _residual_ink_metrics(
    blank_crop: np.ndarray, student_crop: np.ndarray
) -> tuple[float, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Binary-subtraction residual-ink + colour-ink detection.

    Returns (residual_ratio, largest_blob_px, color_ink_px,
    largest_blob_mask, student_only_mask, color_mask,
    aligned_blank_gray, dx, dy).

    The key trick for residual_ratio is binarisation: PDF-render and
    scanned printed text have very different intensities, so a grayscale
    diff inflates every printed pixel into "added ink". After Otsu-
    binarising both crops the printed worksheet becomes ink in both —
    and cancels cleanly against the dilated, aligned blank. Only ink
    the student added survives.

    `color_ink_px` is a complementary high-confidence signal — coloured
    pen ink that simply cannot be the printed worksheet, regardless of
    alignment quality.
    """
    b_gray = cv2.cvtColor(blank_crop, cv2.COLOR_BGR2GRAY) if blank_crop.ndim == 3 else blank_crop
    s_gray = cv2.cvtColor(student_crop, cv2.COLOR_BGR2GRAY) if student_crop.ndim == 3 else student_crop

    # Boost local contrast — faint pencil marks (~150-180 grayscale) need
    # this to clear the binarisation threshold; printed text and pen ink
    # are unaffected because they're already deep in the dark range.
    b_gray = _clahe_gray(b_gray)
    s_gray = _clahe_gray(s_gray)

    # Align on grayscale — phase correlation locks onto rich content
    # better than on binary, where sharp edges produce noisier peaks.
    aligned_gray, dx, dy = _align_blank(b_gray, s_gray)

    # Adaptive thresholding: pixel is ink if it's darker than its local
    # neighbourhood mean by `ADAPTIVE_C` — handles varying paper darkness
    # and faint marks better than Otsu's single global threshold.
    s_bin = cv2.adaptiveThreshold(
        s_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    )
    ab_bin = cv2.adaptiveThreshold(
        aligned_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    )

    # Dilate the aligned blank to forgive 1-2 px sub-pixel misalignment
    # of printed text edges, so they don't leave halo strokes after XOR.
    dilate_k = np.ones((DILATE_KSIZE, DILATE_KSIZE), np.uint8)
    ab_dilated = cv2.dilate(ab_bin, dilate_k, iterations=DILATE_ITERATIONS)

    # Ink in student NOT covered by the dilated aligned blank.
    diff = cv2.bitwise_and(s_bin, cv2.bitwise_not(ab_dilated))

    # Speckle removal.
    diff_clean = cv2.morphologyEx(diff, cv2.MORPH_OPEN, _MORPH_KERNEL)

    # Zero out the regions where comparison is meaningless: the strip the
    # alignment translation filled with border value (the aligned blank
    # has no real content there) plus a small edge margin (scan edges /
    # page-binding shadows that exist only in the student crop).
    h, w = s_gray.shape[:2]
    valid_mask = np.full((h, w), 255, dtype=np.uint8)
    if EDGE_MARGIN_PX > 0:
        valid_mask[:EDGE_MARGIN_PX, :] = 0
        valid_mask[-EDGE_MARGIN_PX:, :] = 0
        valid_mask[:, :EDGE_MARGIN_PX] = 0
        valid_mask[:, -EDGE_MARGIN_PX:] = 0
    # Alignment-border regions: a positive dx shifts content right, so the
    # left strip of width dx is fill. Negative dx fills the right strip.
    pad_l = int(np.ceil(dx)) if dx > 0 else 0
    pad_r = int(-np.floor(dx)) if dx < 0 else 0
    pad_t = int(np.ceil(dy)) if dy > 0 else 0
    pad_b = int(-np.floor(dy)) if dy < 0 else 0
    if pad_l > 0: valid_mask[:, :pad_l] = 0
    if pad_r > 0: valid_mask[:, -pad_r:] = 0
    if pad_t > 0: valid_mask[:pad_t, :] = 0
    if pad_b > 0: valid_mask[-pad_b:, :] = 0
    diff_clean = cv2.bitwise_and(diff_clean, valid_mask)

    raw_mask = (diff_clean > 0).astype(np.uint8)

    # Drop connected components shaped like long thin lines — those are
    # printed box borders / underlines that survive the dilated-blank XOR
    # because alignment is never perfect at the page edges. Real
    # handwriting has moderate aspect ratios and gets through.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_mask, connectivity=8,
    )
    student_only_mask = np.zeros_like(raw_mask)
    for i in range(1, n_labels):
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        long_dim = max(cw, ch)
        short_dim = max(1, min(cw, ch))
        if long_dim >= LINE_MIN_LONG_DIM and (long_dim / short_dim) >= LINE_ASPECT_THRESHOLD:
            continue
        student_only_mask[labels == i] = 1

    residual = float(student_only_mask.sum()) / max(1, student_only_mask.size)
    # Cluster nearby ink: dilate so adjacent digits/strokes of one answer
    # merge into a single connected component (each digit on its own is
    # too small to clear the threshold). Then for each cluster count how
    # many *raw* ink pixels fall inside it. The largest "ink-in-cluster"
    # is the signal — distinguishes a dense answer like "250" from
    # scattered SC-mark noise where each spec sits in its own cluster.
    cluster_kernel = np.ones(
        (CLUSTER_DILATE_PX * 2 + 1, CLUSTER_DILATE_PX * 2 + 1), np.uint8,
    )
    clustered = cv2.dilate(student_only_mask, cluster_kernel)
    n_labels2, labels2, _, _ = cv2.connectedComponentsWithStats(
        clustered, connectivity=8,
    )
    largest_blob_px = 0
    largest_blob_mask = np.zeros_like(student_only_mask)
    for i in range(1, n_labels2):
        cluster_pixels = (labels2 == i).astype(np.uint8)
        ink_in_cluster = int((student_only_mask & cluster_pixels).sum())
        if ink_in_cluster > largest_blob_px:
            largest_blob_px = ink_in_cluster
            largest_blob_mask = (student_only_mask & cluster_pixels)

    color_ink_px, color_mask = _color_ink_pixels(student_crop)
    return (residual, largest_blob_px, color_ink_px,
            largest_blob_mask, student_only_mask, color_mask,
            aligned_gray, dx, dy)


def _classify_attempt(residual_ratio: float, largest_blob_px: int, color_ink_px: int) -> str:
    """Three-signal verdict. Coloured pen ink is the strongest signal — if
    present in any meaningful amount, the student wrote. Otherwise fall
    back to the binary-subtraction signals: either substantial residual
    OR a big ink cluster is enough for attempted; unattempted needs all
    three signals quiet."""
    if color_ink_px >= ATT_MIN_COLOR_INK_PX:
        return "attempted"
    if (residual_ratio >= ATT_MIN_RESIDUAL_RATIO
            or largest_blob_px >= ATT_MIN_LARGEST_BLOB):
        return "attempted"
    if (residual_ratio < UNATT_MAX_RESIDUAL_RATIO
            and largest_blob_px < UNATT_MAX_LARGEST_BLOB
            and color_ink_px < UNATT_MAX_COLOR_INK_PX):
        return "unattempted"
    return "borderline"


def _render_blank_references(
    sheet_pdf: Path,
    questions: list[dict],
    pages_per_student: int,
    blank_out: Path,
    dpi: int,
) -> dict[str, np.ndarray]:
    """Render the unstamped sheet PDF through the template, write each Q's
    blank crop to <blank_out>/<Q>.png, return the BGR crops keyed by Q
    code so attempt detection can compare them pixel-for-pixel against
    student crops. Empty dict (with a warning) when the sheet PDF is
    missing or shorter than the packet — caller treats absent Qs as
    unknown.
    """
    if not sheet_pdf.exists():
        print(f"  blank reference: sheet PDF not found at {sheet_pdf}; skipping")
        return {}

    qs_by_page: dict[int, list[dict]] = {}
    for q in questions:
        qs_by_page.setdefault(q["page"], []).append(q)

    blank_out.mkdir(parents=True, exist_ok=True)
    crops: dict[str, np.ndarray] = {}

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
                crops[q["q"]] = crop
    finally:
        doc.close()

    print(f"  blank reference: wrote {len(crops)} crop(s) to {blank_out}")
    return crops


def extract(
    pdf_path: Path,
    template_path: Path,
    output_dir: Path,
    dpi: int,
    exam_name_override: str | None = None,
    sheet_pdf: Path | None = None,
    skip_existing: bool = False,
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
    manifest_path = exam_out / MANIFEST_CSV_NAME

    # Carry forward prior manifest rows for students we'll skip. Without
    # this, re-running with --skip-existing on a partial scan would erase
    # the previous batch's statuses from manifest.csv.
    prior_rows: list[dict] = []
    skipped_folders: set[str] = set()
    if skip_existing and manifest_path.is_file():
        with manifest_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                folder = (row.get("student_folder") or "").strip()
                q_code = (row.get("q") or "").strip()
                status = (row.get("status") or "").strip()
                if folder and q_code and status:
                    prior_rows.append({"student_folder": folder, "q": q_code, "status": status})
                    skipped_folders.add(folder)
        if skipped_folders:
            print(f"  skip-existing: {len(skipped_folders)} student(s) already in manifest")

    blank_crops: dict[str, np.ndarray] = {}
    if sheet_pdf is not None:
        blank_crops = _render_blank_references(
            sheet_pdf, questions, pages_per_student, exam_out / BLANK_DIR_NAME, dpi,
        )

    # Always emit enhanced copies of the student crops — they're useful in
    # the marker UI even when no sheet PDF is supplied for attempt
    # detection (faint pencil is hard to read at any time).
    enhanced_root = exam_out / ENHANCED_DIR_NAME
    enhanced_root.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)

    manifest_rows: list[dict] = list(prior_rows)
    print(f"\nExtracting {len(questions)} questions per student at {dpi} DPI...")
    for group in groups:
        if group.folder_name in skipped_folders:
            print(f"  [SKIP] {group.folder_name:<28} (already in manifest)")
            continue
        rows, n_extracted, note = _extract_one_student(
            doc, group, qs_by_page, exam_out, pages_per_student, dpi,
            blank_crops=blank_crops,
            enhanced_root=enhanced_root,
        )
        marker = "OK  " if not note else "WARN"
        print(f"  [{marker}] {group.folder_name:<28} {n_extracted:>3} crops  {note}")
        manifest_rows.extend(rows)

    doc.close()

    _write_manifest_csv(manifest_path, manifest_rows)
    print(f"\nDone. Output: {exam_out}")
    print(f"Manifest:    {manifest_path}")


def _extract_one_student(
    doc: pymupdf.Document,
    group: StudentGroup,
    qs_by_page: dict[int, list[dict]],
    exam_out: Path,
    pages_per_student: int,
    dpi: int,
    blank_crops: dict[str, np.ndarray] | None = None,
    enhanced_root: Path | None = None,
) -> tuple[list[dict], int, str]:
    """Returns (manifest_rows, n_extracted, note).

    manifest_rows are (student_folder, q, status) dicts. note is empty on
    success, or a short failure reason (e.g. page count mismatch) — in
    that case manifest_rows is empty and the student gets no entries in
    manifest.csv. The caller surfaces the note in its console log.
    """
    if len(group.pages) != pages_per_student:
        note = f"page count mismatch (got {len(group.pages)}, expected {pages_per_student})"
        return [], 0, note

    student_dir = exam_out / group.folder_name
    student_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = (enhanced_root / group.folder_name) if enhanced_root is not None else None
    if enhanced_dir is not None:
        enhanced_dir.mkdir(parents=True, exist_ok=True)

    blank_crops = blank_crops or {}
    rows: list[dict] = []
    n_extracted = 0
    for packet_page_idx in range(1, pages_per_student + 1):
        qs = qs_by_page.get(packet_page_idx, [])
        if not qs:
            continue
        pdf_page_num = group.pages[packet_page_idx - 1].pdf_page_number
        page_img = render_page(doc, pdf_page_num, dpi)
        for q in qs:
            q_code = q["q"]
            crop = crop_region(page_img, q["bbox"])
            cv2.imwrite(str(student_dir / f"{q_code}.png"), crop)
            if enhanced_dir is not None:
                cv2.imwrite(
                    str(enhanced_dir / f"{q_code}.png"),
                    _enhance_for_display(crop),
                )
            n_extracted += 1
            blank_crop = blank_crops.get(q_code) if blank_crops else None
            if blank_crop is None:
                status = "unknown"
            else:
                metrics = _residual_ink_metrics(blank_crop, crop)
                residual, largest_blob_px, color_ink_px = metrics[0], metrics[1], metrics[2]
                status = _classify_attempt(residual, largest_blob_px, color_ink_px)
            rows.append({
                "student_folder": group.folder_name,
                "q": q_code,
                "status": status,
            })

    return rows, n_extracted, ""


def _write_manifest_csv(manifest_path: Path, rows: list[dict]) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["student_folder", "q", "status"])
        writer.writeheader()
        writer.writerows(rows)
    n_att = sum(1 for r in rows if r["status"] == "attempted")
    n_bord = sum(1 for r in rows if r["status"] == "borderline")
    n_unatt = sum(1 for r in rows if r["status"] == "unattempted")
    n_unk = sum(1 for r in rows if r["status"] == "unknown")
    print(
        f"\nManifest: {len(rows)} crops — "
        f"{n_att} attempted, {n_bord} borderline, "
        f"{n_unatt} unattempted, {n_unk} unknown."
    )


def rescore(exam_dir: Path) -> None:
    """Re-run attempt detection against existing crops on disk.

    Reuses <exam>/_blank/ and the per-student folders that the previous
    extract run left behind, and rewrites manifest.csv with the current
    classifier parameters. Lets you iterate on thresholds in seconds
    instead of re-rendering the scan PDF.
    """
    if not exam_dir.is_dir():
        sys.exit(f"Exam directory not found: {exam_dir}")
    blank_dir = exam_dir / BLANK_DIR_NAME
    if not blank_dir.is_dir():
        sys.exit(f"No {BLANK_DIR_NAME}/ folder at {blank_dir} — nothing to rescore.")

    blank_crops: dict[str, np.ndarray] = {}
    for fname in sorted(os.listdir(blank_dir)):
        if not fname.lower().endswith(".png"):
            continue
        q_code = os.path.splitext(fname)[0]
        img = cv2.imread(str(blank_dir / fname))
        if img is not None:
            blank_crops[q_code] = img
    if not blank_crops:
        sys.exit(f"No blank crops found in {blank_dir}.")
    print(f"Loaded {len(blank_crops)} blank reference crops from {blank_dir}.")

    enhanced_root = exam_dir / ENHANCED_DIR_NAME
    enhanced_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    student_count = 0
    for entry in sorted(os.listdir(exam_dir)):
        student_dir = exam_dir / entry
        if not student_dir.is_dir() or entry.startswith("_"):
            continue
        enhanced_dir = enhanced_root / entry
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        scored_this_student = 0
        for fname in sorted(os.listdir(student_dir)):
            if not fname.lower().endswith(".png"):
                continue
            q_code = os.path.splitext(fname)[0]
            blank_crop = blank_crops.get(q_code)
            if blank_crop is None:
                continue
            student_crop = cv2.imread(str(student_dir / fname))
            if student_crop is None:
                continue
            cv2.imwrite(
                str(enhanced_dir / fname),
                _enhance_for_display(student_crop),
            )
            metrics = _residual_ink_metrics(blank_crop, student_crop)
            residual, largest_blob_px, color_ink_px = metrics[0], metrics[1], metrics[2]
            status = _classify_attempt(residual, largest_blob_px, color_ink_px)
            manifest_rows.append({
                "student_folder": entry, "q": q_code, "status": status,
            })
            scored_this_student += 1
        if scored_this_student:
            student_count += 1
            print(f"  scored {scored_this_student:>3} crops for {entry}")

    print(f"\n{student_count} students processed.")
    _write_manifest_csv(exam_dir / MANIFEST_CSV_NAME, manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-question crops from a scan PDF.")
    parser.add_argument(
        "--rescore",
        type=Path,
        default=None,
        metavar="EXAM_DIR",
        help=(
            "Skip extraction and re-run attempt detection against the crops "
            "already on disk under EXAM_DIR (which must contain _blank/ and "
            "per-student folders). Rewrites manifest.csv without touching the "
            "scan PDF — fast loop for tuning thresholds."
        ),
    )
    parser.add_argument("pdf", type=Path, nargs="?", help="Path to the scan PDF")
    parser.add_argument("template", type=Path, nargs="?", help="Path to the template YAML")
    parser.add_argument("output", type=Path, nargs="?", help="Output directory")
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
            "to it to produce reference crops under <output>/<exam>/_blank/, "
            "and manifest.csv classifies each student crop as attempted, "
            "unattempted, or borderline. Defaults to env QMARK_SHEET_PATH."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip any student already present in the prior manifest.csv. "
            "Their rows are preserved verbatim and newly-scanned students "
            "are appended — use to merge later scans into an in-progress "
            "assignment without overwriting work already marked."
        ),
    )
    args = parser.parse_args()

    if args.rescore is not None:
        rescore(args.rescore)
        return

    if args.pdf is None or args.template is None or args.output is None:
        parser.error("pdf, template, and output are required (omit only with --rescore)")
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
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
