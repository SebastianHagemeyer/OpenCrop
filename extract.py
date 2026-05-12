"""Apply a template YAML to a scan PDF, output per-question crops per student.

Usage:
    python extract.py <pdf> <template.yaml> <output_dir> [--dpi 300]
        [--sheet-pdf <blank_sheet.pdf>]

Output layout:
    <output_dir>/<exam>/manifest.csv
    <output_dir>/<exam>/<class>_<firstname>/Q01.png ... QNN.png

When a blank sheet PDF is supplied (--sheet-pdf, or QMARK_SHEET_PATH env
var), the same template is applied to the unstamped worksheet to produce
reference crops under <output>/<exam>/_blank/<Q>.png. Each student crop
is then compared against its blank reference using an aligned residual-
ink detector (see _residual_ink_metrics), and the verdict per (student,
question) is written to <output>/<exam>/attempts.csv along with the
applied alignment shift. A side-by-side debug PNG for every (student, q)
lands at <output>/<exam>/_debug/<student>/<q>.png — useful when tuning
the thresholds or sanity-checking false positives.
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
DEBUG_DIR_NAME = "_debug"
ENHANCED_DIR_NAME = "_enhanced"
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


def _build_debug_image(
    aligned_blank_gray: np.ndarray,
    student_crop: np.ndarray,
    ink_mask_clean: np.ndarray,
    largest_blob_mask: np.ndarray,
    color_mask: np.ndarray,
    status: str,
    residual: float,
    largest_blob_px: int,
    color_ink_px: int,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Three-panel side-by-side: aligned blank | student | residual overlay.

    Overlay panel: student crop with all detected residual-ink pixels in
    red, and the LARGEST connected component re-painted in orange on top
    — so a quick look tells you (a) what the detector counted as ink, and
    (b) whether the biggest contiguous chunk looks like real handwriting
    or like a smear of scattered noise.

    Header bar shows verdict, both metrics, and the alignment shift.
    """
    if student_crop.ndim == 2:
        student_show = cv2.cvtColor(student_crop, cv2.COLOR_GRAY2BGR)
    else:
        student_show = student_crop
    blank_show = cv2.cvtColor(aligned_blank_gray, cv2.COLOR_GRAY2BGR)
    h, w = student_show.shape[:2]
    if blank_show.shape[:2] != (h, w):
        blank_show = cv2.resize(blank_show, (w, h))

    overlay = student_show.copy()
    color_b = color_mask.astype(bool)
    all_ink = ink_mask_clean.astype(bool)
    big = largest_blob_mask.astype(bool)
    overlay[all_ink] = (0, 0, 255)        # red: all binary-residual ink
    overlay[big]     = (0, 165, 255)      # orange: largest cluster
    overlay[color_b] = (255, 0, 255)      # magenta: coloured-pen pixels

    sep_w = 6
    sep = np.full((h, sep_w, 3), 80, dtype=np.uint8)
    panel = np.hstack([blank_show, sep, student_show, sep, overlay])

    label = (
        f"{status:>11}  "
        f"residual={residual*100:6.3f}%  "
        f"cluster_ink={largest_blob_px:>5}px  "
        f"color_ink={color_ink_px:>5}px  "
        f"shift=({dx:+.1f},{dy:+.1f})"
    )
    label_h = 28
    label_bar = np.full((label_h, panel.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(
        label_bar, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([label_bar, panel])


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

    blank_crops: dict[str, np.ndarray] = {}
    if sheet_pdf is not None:
        blank_crops = _render_blank_references(
            sheet_pdf, questions, pages_per_student, exam_out / BLANK_DIR_NAME, dpi,
        )

    debug_root: Path | None = None
    enhanced_root: Path | None = None
    if blank_crops:
        debug_root = exam_out / DEBUG_DIR_NAME
        debug_root.mkdir(parents=True, exist_ok=True)
    # Always emit enhanced copies of the student crops — they're useful in
    # the marker UI even when no sheet PDF is supplied for attempt
    # detection (faint pencil is hard to read at any time).
    enhanced_root = exam_out / ENHANCED_DIR_NAME
    enhanced_root.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)

    manifest_rows: list[dict] = []
    attempts_rows: list[dict] = []
    print(f"\nExtracting {len(questions)} questions per student at {dpi} DPI...")
    for group in groups:
        row, student_attempts = _extract_one_student(
            doc, group, qs_by_page, exam_out, pages_per_student, dpi,
            blank_crops=blank_crops,
            debug_root=debug_root,
            enhanced_root=enhanced_root,
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

    if blank_crops:
        attempts_path = exam_out / ATTEMPTS_CSV_NAME
        _write_attempts_csv(attempts_path, attempts_rows)
        if debug_root is not None:
            print(f"Debug images:    {debug_root}")

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
    debug_root: Path | None = None,
    enhanced_root: Path | None = None,
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
    debug_dir = (debug_root / group.folder_name) if debug_root is not None else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = (enhanced_root / group.folder_name) if enhanced_root is not None else None
    if enhanced_dir is not None:
        enhanced_dir.mkdir(parents=True, exist_ok=True)

    blank_crops = blank_crops or {}
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
            if enhanced_dir is not None:
                cv2.imwrite(
                    str(enhanced_dir / f"{q['q']}.png"),
                    _enhance_for_display(crop),
                )
            n_extracted += 1
            if not blank_crops:
                continue
            q_code = q["q"]
            blank_crop = blank_crops.get(q_code)
            if blank_crop is None:
                attempts.append({
                    "student_folder": group.folder_name, "q": q_code,
                    "status": "unknown",
                    "residual_ratio": "", "alignment_dx": "", "alignment_dy": "",
                })
                continue
            (residual, largest_blob_px, color_ink_px,
             largest_mask, ink_mask, color_mask,
             aligned_blank, dx, dy) = _residual_ink_metrics(blank_crop, crop)
            status = _classify_attempt(residual, largest_blob_px, color_ink_px)
            attempts.append({
                "student_folder": group.folder_name, "q": q_code,
                "status": status,
                "residual_ratio": f"{residual:.5f}",
                "largest_blob_px": str(largest_blob_px),
                "color_ink_px": str(color_ink_px),
                "alignment_dx": f"{dx:+.2f}",
                "alignment_dy": f"{dy:+.2f}",
            })
            if debug_dir is not None:
                debug_img = _build_debug_image(
                    aligned_blank, crop, ink_mask, largest_mask, color_mask,
                    status, residual, largest_blob_px, color_ink_px, dx, dy,
                )
                cv2.imwrite(str(debug_dir / f"{q_code}.png"), debug_img)

    base_row["n_questions_extracted"] = n_extracted
    return base_row, attempts


def _write_attempts_csv(attempts_path: Path, rows: list[dict]) -> None:
    with attempts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "student_folder", "q", "status",
            "residual_ratio", "largest_blob_px", "color_ink_px",
            "alignment_dx", "alignment_dy",
        ])
        writer.writeheader()
        writer.writerows(rows)
    n_att = sum(1 for r in rows if r["status"] == "attempted")
    n_bord = sum(1 for r in rows if r["status"] == "borderline")
    n_unatt = sum(1 for r in rows if r["status"] == "unattempted")
    print(
        f"\nAttempt detection: {len(rows)} crops scored — "
        f"{n_att} attempted, {n_bord} borderline, {n_unatt} unattempted. "
        f"Written to {attempts_path}"
    )


def rescore(exam_dir: Path) -> None:
    """Re-run attempt detection against existing crops on disk.

    Reuses <exam>/_blank/ and the per-student folders that the previous
    extract run left behind, recomputes attempts.csv with the current
    classifier parameters, and refreshes <exam>/_debug/ images. Lets you
    iterate on thresholds in seconds instead of re-rendering the scan PDF.
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

    debug_root = exam_dir / DEBUG_DIR_NAME
    debug_root.mkdir(parents=True, exist_ok=True)
    enhanced_root = exam_dir / ENHANCED_DIR_NAME
    enhanced_root.mkdir(parents=True, exist_ok=True)

    attempts_rows: list[dict] = []
    student_count = 0
    for entry in sorted(os.listdir(exam_dir)):
        student_dir = exam_dir / entry
        if not student_dir.is_dir() or entry.startswith("_"):
            continue
        debug_dir = debug_root / entry
        debug_dir.mkdir(parents=True, exist_ok=True)
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
            (residual, largest_blob_px, color_ink_px,
             largest_mask, ink_mask, color_mask,
             aligned_blank, dx, dy) = _residual_ink_metrics(blank_crop, student_crop)
            status = _classify_attempt(residual, largest_blob_px, color_ink_px)
            attempts_rows.append({
                "student_folder": entry, "q": q_code,
                "status": status,
                "residual_ratio": f"{residual:.5f}",
                "largest_blob_px": str(largest_blob_px),
                "color_ink_px": str(color_ink_px),
                "alignment_dx": f"{dx:+.2f}",
                "alignment_dy": f"{dy:+.2f}",
            })
            debug_img = _build_debug_image(
                aligned_blank, student_crop, ink_mask, largest_mask, color_mask,
                status, residual, largest_blob_px, color_ink_px, dx, dy,
            )
            cv2.imwrite(str(debug_dir / f"{q_code}.png"), debug_img)
            scored_this_student += 1
        if scored_this_student:
            student_count += 1
            print(f"  scored {scored_this_student:>3} crops for {entry}")

    print(f"\n{student_count} students processed.")
    _write_attempts_csv(exam_dir / ATTEMPTS_CSV_NAME, attempts_rows)
    print(f"Debug images:    {debug_root}")


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
            "per-student folders). Refreshes attempts.csv and _debug/ images "
            "without touching the scan PDF — fast loop for tuning thresholds."
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
            "to it to produce reference crops under <output>/<exam>/_blank/, an "
            "attempts.csv classifying each student crop as attempted, "
            "unattempted, or borderline (aligned residual-ink detector), and a "
            "side-by-side debug PNG per (student, q) under "
            "<output>/<exam>/_debug/. Defaults to env QMARK_SHEET_PATH."
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
    )


if __name__ == "__main__":
    main()
