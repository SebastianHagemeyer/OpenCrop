# exam-region-extractor — context for a Claude agent

Python tool that takes a scanned exam PDF (each page has a QR code identifying the student) and produces per-question image crops organised by student. The intended downstream consumer is a separate **marker** tool (likely on its own git branch) that iterates over the crops and lets the teacher grade them.

This file is the handoff doc. It assumes you have not seen the conversation that built this.

## Repo layout

- `qr_probe.py` — one-shot CLI: prints the decoded QR string per page of a PDF.
- `scan_index.py` — library + CLI: decodes QRs and groups pages into per-student packets using the page/total embedded in each QR. Has retry preprocessing (Otsu, rescaling) and neighbour-based inference for any pages whose QR still won't decode.
- `make_template.py` — Tkinter GUI: pick a reference student, click+drag rectangles for each question on each page, save to YAML.
- `extract.py` — CLI: applies a template to a PDF, writes per-student crops + a manifest. Useful for archiving, sharing answers, or feeding a marker that doesn't want pymupdf/opencv as deps. NOT required for on-the-fly access (see below).
- `requirements.txt` — pymupdf, opencv-python, numpy, Pillow, PyYAML.
- `workScans/` — gitignored. Real student PDFs live here locally.
- `output/` — gitignored. Will hold extracted crops once `extract.py` exists.
- `templates/` — gitignored. Holds saved template YAMLs.

## QR code format on the scans

Each page has one QR encoding the literal string:

```
<class>/<firstname>/<page>/<total>
```

Examples from `workScans/10MATD_combinedTEST.pdf`: `10MATD/Dj/1/2`, `10MATD/Ruby/2/2`, `10MATD/Amarni-Faith/1/2`.

- `class` is **not** required to be constant across one PDF — a single scan can mix classes (e.g. `matpremix.pdf` contains both `10MATD` and `10MATG` packets).
- `firstname` is the only student identifier (no surname, no ID). First-name collisions are scoped per `(class, firstname)`, so the same first name in two different classes is fine.
- `page` / `total` give the page's position in this student's packet (1-based). This is authoritative — `scan_index.py` does not need PDF order to figure out packet structure.
- A legacy 2-segment form (`<class>/<firstname>`, no page/total) is also accepted. Mixed in one PDF is fine. For legacy pages, packet position is reconstructed from the order the pages appear within their group.

QR detection rate on `10MATD_combinedTEST.pdf` at 250 DPI: 47/52 plain, 52/52 with the Otsu + rescaling fallback in `_decode_qr`. If a page still can't be decoded, `_infer_missing` in `scan_index.py` assigns it from the nearest decoded neighbour (e.g. a missing page immediately followed by `X/2/2` is inferred to be `X/1/2`). Pages with no usable neighbour stay `unknown` and are visible in the manifest.

## Output folder schema (what the marker iterates over)

```
output/
└── <exam_name>/                          # e.g. workscan10Dpretest/
    ├── manifest.csv                      # one row per student
    ├── attempts.csv                      # only if --sheet-pdf was supplied
    ├── _blank/                           # only if --sheet-pdf was supplied
    │   ├── Q01.png                       # the unstamped sheet, cropped through the same template
    │   ├── Q02.png
    │   └── ...
    ├── 10MATD_Ruby/
    │   ├── Q01.png
    │   ├── Q02.png
    │   ├── ...
    │   └── Q12.png
    ├── 10MATD_Ali/
    │   └── ...
    └── ...
```

Conventions:

- **Student folder name:** `<class>_<firstname>` (the only thing the QR gives us). The marker should skip any folder starting with `_` (currently just `_blank`).
- **Question file name:** `Q01.png` ... `QNN.png`. Zero-padded so a file-manager sort matches numeric order. Always PNG (lossless — pen strokes stay sharp).
- **`manifest.csv`** columns: `student_class, student_name, packet_pdf_pages, qr_status_per_page, n_questions_extracted, notes`. `qr_status_per_page` is comma-joined per-page values from {`decoded`, `preprocessed`, `inferred`, `unknown`} — useful for flagging crops that came from a recovered/inferred page vs a confident decode.

## Attempt detection (when --sheet-pdf is supplied)

When the dashboard launches OpenCrop it sets `QMARK_SHEET_PATH` to the unstamped worksheet PDF; the same value can be passed on the CLI as `--sheet-pdf`. When present, `extract.py` renders that blank sheet through the *same template* and writes:

- **`_blank/Q01.png … QNN.png`** — reference "empty" crops, useful for diffs in the marker UI and as a visual baseline.
- **`attempts.csv`** with columns `student_folder, q, status, residual_ratio, largest_blob_px, alignment_dx, alignment_dy`. `status` is one of:
  - `attempted` — either metric clearly above the floor (residual ≥ 3% **or** largest blob ≥ 5500 px). One signal is enough.
  - `borderline` — small but non-zero residual; worth a human re-check.
  - `unattempted` — both metrics quiet (residual < 2% **and** largest blob < 3000 px); the marker UI greys out and the teacher can score 0 with Ctrl+0.
  - `unknown` — no blank reference for this Q (sheet PDF was shorter than the packet, etc).
- **`_debug/<student>/<q>.png`** — a three-panel side-by-side: aligned blank | student | residual overlay. Detected ink is painted red; the *largest connected component* (the one that drives the blob metric) is painted orange on top. A header strip shows the verdict, both metrics, and the alignment shift. Open these in any image viewer to spot-check false positives/negatives.

### The detector pipeline

1. **Phase-correlation alignment** translates the blank crop to best fit the student crop (capped at ±20 px to avoid locking onto noise on a feature-poor crop).
2. **Gaussian blur** (3×3, σ=0.8) of both crops absorbs sub-pixel registration error and PDF-vs-scan anti-aliasing differences.
3. **Intensity diff** `max(blank − student, 0)` — how much darker each pixel got. Scan paper-darkening (~10–20 grayscale units) stays under the cutoff; pen strokes (100+ units darker) survive.
4. **Threshold** at `INK_DIFF_THRESHOLD = 60` grayscale units, then **morphological open** with a 2×2 kernel kills isolated speckle noise.
5. Two metrics fall out of the cleaned mask:
   - `residual_ratio` = fraction of crop pixels still marked as added ink. Picks up scattered-mark answers (ticks, asterisks) that don't form a big blob.
   - `largest_blob_px` = size of the biggest connected component. Picks up contiguous handwriting strokes; print-edge noise stays in many tiny blobs.

Tunable parameters live at the top of `extract.py`: pipeline ones (`ALIGNMENT_MAX_SHIFT_PX`, `BLUR_KSIZE`, `BLUR_SIGMA`, `INK_DIFF_THRESHOLD`) and band cutoffs (`UNATT_MAX_*`, `ATT_MIN_*`). The cutoffs assume a 300 DPI scan of a single-column maths worksheet — denser sheets (graph paper, dense formulas) raise the noise floor and need higher cutoffs.

### Iterating on thresholds without re-extracting

Re-rendering the scan PDF takes minutes per exam. Once you have student crops and `_blank/` on disk, use:

```
python extract.py --rescore <exam_dir>
```

It loads existing crops, re-runs only the comparison step, and rewrites `attempts.csv` + `_debug/` images in seconds. Drop in new threshold values, rescore, eyeball the debug images, repeat.

### Marker UI fallback

When `attempts.csv` is missing, the marker should fall back to "all unknown" — i.e. treat every question as attempted by default and skip the greying behaviour.

## Template YAML schema

Produced by `make_template.py`, consumed by `extract.py`. Shape:

```yaml
exam: workscan10Dpretest                  # str — derived from PDF stem
reference_student: 10MATD_Ruby            # str — which student was used to define the regions
pages_per_student: 2                      # int — number of pages in one student's packet
questions:
  - q: Q01
    page: 1                               # 1 = first page of packet, NOT absolute PDF page
    bbox: [0.05, 0.10, 0.95, 0.25]        # [x0, y0, x1, y1] normalized to [0, 1] of page width/height
  - q: Q02
    page: 1
    bbox: [0.05, 0.27, 0.95, 0.42]
  - q: Q07
    page: 2
    bbox: [0.05, 0.10, 0.95, 0.30]
  # ...
```

- **`bbox` is normalized.** Multiply by rendered page width/height to get pixel coords. This makes the template DPI-independent — extract.py can render at any DPI and the same template still works.
- **`page` is student-relative**, not absolute PDF page. The same template applies to every student because pages-per-student is fixed and pages are printed identically.
- `questions` is sorted by Q-number on save.

## How to run

```
python -m pip install -r requirements.txt

# Inspect raw QR contents of a scan (one line per page)
python qr_probe.py workScans\10MATD_combinedTEST.pdf

# Show grouped student structure (one line per student)
python scan_index.py workScans\10MATD_combinedTEST.pdf

# Define question regions (opens Tkinter GUI; takes ~10–15s to index first)
python make_template.py workScans\10MATD_combinedTEST.pdf

# Extract per-question crops to disk (optional — see "on-the-fly" below)
python extract.py workScans\10MATD_combinedTEST.pdf 10MATD_combinedTEST.yaml output\
# Custom DPI: --dpi 400
# With attempt detection (writes _blank/, attempts.csv, _debug/):
python extract.py workScans\10MATD_combinedTEST.pdf 10MATD_combinedTEST.yaml output\ --sheet-pdf Sheets\10MATD_combinedTEST.pdf

# Re-run JUST the attempt detection on existing crops (fast threshold-tuning loop):
python extract.py --rescore output\10MATD_combinedTEST
```

## On-the-fly crop access (preferred for the marker)

You don't need to pre-extract crops to disk. The marker can read crops directly from the PDF + template YAML. Sketch:

```python
from pathlib import Path
import cv2, numpy as np, pymupdf, yaml
from scan_index import group_into_students, index_pdf

template = yaml.safe_load(Path("10MATD_combinedTEST.yaml").read_text())
groups   = group_into_students(index_pdf(Path("workScans/10MATD_combinedTEST.pdf")))
doc      = pymupdf.open("workScans/10MATD_combinedTEST.pdf")

DPI = 300
zoom = pymupdf.Matrix(DPI / 72.0, DPI / 72.0)

def crop_for(group, q):
    pdf_pg = group.pages[q["page"] - 1].pdf_page_number
    pix    = doc[pdf_pg - 1].get_pixmap(matrix=zoom, alpha=False)
    img    = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    img    = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w   = img.shape[:2]
    x0, y0, x1, y1 = q["bbox"]
    return img[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]

# Example: get Ruby's Q03
ruby = next(g for g in groups if g.folder_name == "10MATD_Ruby")
q03  = next(q for q in template["questions"] if q["q"] == "Q03")
img  = crop_for(ruby, q03)
```

A render is ~200–500ms per page at 300 DPI; subsequent crops on the same page are free. Cache the rendered page if you're displaying multiple questions from the same page sequentially.

When pre-extraction wins: archiving graded exams, handing off without the original PDF, marker tool that wants zero PDF/CV deps.

## Things to be careful about

- **Never commit `workScans/`, `output/`, `templates/`, or `*.pdf` / `roster*.csv`.** They contain student data. The `.gitignore` already excludes them; do not override with `git add -f`.
- First-name-only identifier means two students sharing a first name in one class will collide. Current test class has no collisions; add disambiguation when needed.
- Legacy 2-segment QRs (`<class>/<firstname>`) are accepted as a fallback. Pages are grouped by `(class, name)` and packet position is reconstructed from PDF order within the group. This means legacy packets must be printed/scanned contiguously — interleaving two legacy packets with the same first name across the PDF will collapse them into one (mis-paginated) group. The 4-segment form is still preferred when you have control over the QR.
- Image coords in templates are normalized — never store pixel coords, they break across DPIs.
