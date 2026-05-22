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

QR detection rate on `10MATD_combinedTEST.pdf` at 250 DPI: 47/52 plain, 52/52 with the Otsu + rescaling fallback in `_decode_qr`. If a page still can't be decoded, `_infer_missing` in `scan_index.py` assigns it from the nearest decoded neighbour (e.g. a missing page immediately followed by `X/2/2` is inferred to be `X/1/2`). Pages with no usable neighbour stay `unknown` and never produce a student folder.

## Launcher view (Check scan output)

The launcher's main surface is `ScanResultView` (`scan_view.py`) — a class-roster table, **not** a debug log. Each indexed student is one row: name, PDF pages, and a coloured status pill (green `decoded`, blue `inferred`, purple `manual` for sidecar overrides). The header above the table reports the dominant class and the matched-student/page totals. A small log panel sits below for one-line messages from each pipeline stage.

When orphans exist, an orange banner appears above the roster — `"⚠ N page(s) couldn't be matched to a student (pages …). Click Resolve to assign them by hand."` — with a Resolve button that opens the recovery dialog. The banner is the only place orphan pages surface; they never get rendered as roster rows. After Apply, the banner disappears and the recovered students show up tagged `manual`. Make changes to the layout/colors there, not in `app.py`.

## Recovering orphan pages (the dialog)

Inference can't bridge across a run of orphans — if Shylah tore her QR and Arvin drew over his, all four of their pages sit between Trey and Yusra with no neighbour to lean on, and they fall out as `UNKNOWN_orphan_p<N>` groups (one fake "student" per page).

After **Check scan**, if any page still has `qr_status == "unknown"` the **orphan recovery dialog** (`orphan_dialog.py`) auto-opens. It shows a thumbnail of each orphan page (rendered at ~60 DPI — good enough to read the printed Name field, the teacher's title block, and most of the student's handwriting) alongside an editable combobox prefilled with rostered students not yet decoded in this scan. The teacher types or picks a first name per page; same name on multiple pages combines them into one packet in PDF order. Confirming writes a `<pdf>.qrfix.json` sidecar next to the scan:

```json
{
  "overrides": {
    "23": {"class": "10MATD", "name": "Arvin",  "page_in_packet": 1, "pages_total": 2},
    "24": {"class": "10MATD", "name": "Arvin",  "page_in_packet": 2, "pages_total": 2},
    "25": {"class": "10MATD", "name": "Shylah", "page_in_packet": 1, "pages_total": 2},
    "26": {"class": "10MATD", "name": "Shylah", "page_in_packet": 2, "pages_total": 2}
  }
}
```

`scan_index.index_pdf()` reads the sidecar on every call (so extract, rescore, and any later check picks the fix up automatically). The button **Fix orphans...** re-opens the dialog without re-rendering the PDF; the launcher caches the last indexed page list for that.

The roster used for the dropdown comes from `QMARK_CLASS_PATH` (an .xlsx with `Name:` in the first column — what the dashboard hands off). Missing/unreadable roster → the combobox stays freely typable, no choices listed. The sidecar is the user-editable source of truth — delete or hand-edit the file to undo or tweak recoveries.

Each row also has a **Packet page** spinbox. Default value is the row's PDF-order index (row 1 → 1, row 2 → 2, …), so the historical behaviour is the default. The spinner lets the teacher override when pages were handed in out of order (e.g. a 2-page packet where the student stapled them in reverse). On Apply, pages for the same student are grouped by name and sorted by the spinbox values; duplicate packet pages for one name trigger a warning instead of silently overwriting an override. `pages_total` is set to `max(packet_page)` per name, so a sparse claim (1 and 3 with no 2) is allowed — the teacher knows their data.

## Skip / view from the roster (right-click)

Right-clicking a student row in `ScanResultView` opens a small menu:

- **View pages...** (also bound to double-click) opens `PageViewerDialog` — a horizontal strip of low-DPI thumbnails for every PDF page in that student's packet. Useful for confirming a recovered student really is who you think they are, or spotting a mis-grouping before extract.
- **Skip from extraction** / **Include in extraction** toggles whether the student is excluded. Skipped rows render strikethrough + grey + a `skipped — <status>` label, and the header counts them separately (e.g. `10MATD — 18 students, 2 skipped`).

Skip state lives in the same `<pdf>.qrfix.json` sidecar as the recovery overrides, under a `skipped_students` array of folder_names:

```json
{
  "overrides": { ... },
  "skipped_students": ["10MATG_Jamie", "10MATG_Jordan"]
}
```

`extract.extract()` calls `load_skipped_students(pdf_path)` directly (not via the launcher), so the skip is honoured by the CLI too. The sidecar is deleted automatically when both `overrides` and `skipped_students` are empty, keeping the data directory clean when the teacher undoes everything.

Primary use case: scanning two classes' worth of papers in one PDF, then skipping the wrong-class students for *this* extract run. Toggle them back in later for the other class's extract.

## Extract reuses the scan cache (and auto-Checks if missing)

`extract.extract()` accepts a `cached_pages: list[PageRecord] | None` kwarg. When the launcher passes its `_last_pages` through, extract skips `index_pdf` entirely — same ~70s → instant win as the region editor cache. The CLI path (`python extract.py …`) is unaffected: no kwarg → fall back to indexing.

The launcher's `_extract` is a request-then-resume flow now:

1. Gather params (template, exam name, sheet PDF, checkboxes) into `self._pending_extract`.
2. If `_last_pdf == pdf` and `_last_pages is not None`: skip straight to `_resume_pending_extract`.
3. Otherwise: log "No scan cached — running Check scan first" and trigger `_check_scan`. The standard `_on_pages_indexed` handler notices the pending request and hands off to `_resume_pending_extract` instead of auto-popping the orphan dialog (so the user isn't prompted twice for the same orphans).
4. `_resume_pending_extract`: if any orphans remain, show a 3-button dialog (`Resolve now…` / `Extract anyway` / `Cancel`). `Resolve now…` opens the recovery dialog; whatever's left after that just flows through as `orphan_pXX/` folders (single-prompt rule — no nagging the user with a second confirmation).
5. Spawn the extract worker thread with `cached_pages` set.

## Region editor reuses the scan cache

The launcher caches the post-`index_pdf` page list on `self._last_pages` after Check scan finishes. When the user clicks **Define regions** for the same PDF, the launcher passes that list through as `TemplateEditor(pdf, cached_pages=...)`. The editor short-circuits its own streaming decoder (which re-renders every page at `INDEX_DPI`) and bootstraps from the cached groups in ~300ms instead of ~30–70s. Since `_last_pages` already carries any sidecar overrides, recovered students appear in the reference-student dropdown too. Opening a different PDF later via the editor's "Open PDF..." button still falls back to the streaming path — the cache only fires on the first `_load_pdf` call after construction.

## Output folder schema (what the marker iterates over)

```
output/
└── <exam_name>/                          # e.g. workscan10Dpretest/
    ├── manifest.csv                      # one row per (student, q)
    ├── _blank/                           # only if --sheet-pdf was supplied
    │   ├── Q01.png                       # the unstamped sheet, cropped through the same template
    │   ├── Q02.png
    │   └── ...
    ├── _enhanced/                        # CLAHE-boosted copies of student crops (marker uses these)
    │   └── 10MATD_Ruby/Q01.png ...
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

- **Student folder name:** `<class>_<firstname>` (the only thing the QR gives us). The marker should skip any folder starting with `_` (currently `_blank` and `_enhanced`).
- **Question file name:** `Q01.png` ... `QNN.png`. Zero-padded so a file-manager sort matches numeric order. Always PNG (lossless — pen strokes stay sharp).
- **`manifest.csv`** columns: `student_folder, q, status`. One row per (student, q). Status is one of {`attempted`, `unattempted`, `borderline`, `unknown`}. Without a sheet PDF every row's status is `unknown`. The marker reads this file directly to grey out unattempted Qs.

## Attempt detection (when --sheet-pdf is supplied)

When the dashboard launches OpenCrop it sets `QMARK_SHEET_PATH` to the unstamped worksheet PDF; the same value can be passed on the CLI as `--sheet-pdf`. When present, `extract.py` renders that blank sheet through the *same template* and writes:

- **`_blank/Q01.png … QNN.png`** — reference "empty" crops, useful for diffs in the marker UI and as a visual baseline.
- **`manifest.csv`** gets real verdicts in its `status` column instead of `unknown`. Statuses:
  - `attempted` — either metric clearly above the floor (residual ≥ 3% **or** largest blob ≥ 5500 px). One signal is enough.
  - `borderline` — small but non-zero residual; worth a human re-check.
  - `unattempted` — both metrics quiet (residual < 2% **and** largest blob < 3000 px); the marker UI greys out and the teacher can score 0 with Ctrl+0.
  - `unknown` — no blank reference for this Q (sheet PDF was shorter than the packet, etc).

The raw detector metrics (residual_ratio, largest_blob_px, alignment dx/dy) are no longer persisted — only the classified status. If you need to tune thresholds and inspect numerics, edit the constants at the top of `extract.py` and re-run with `--rescore`; the verdicts that come out are what the marker sees.

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

It loads existing crops, re-runs only the comparison step, and rewrites `manifest.csv` in seconds. Drop in new threshold values, rescore, look at the marker UI, repeat.

### Skipping students already extracted

Pass `--skip-existing` (or tick the **Skip students already in manifest** checkbox in the launcher GUI) when extracting a later scan PDF on the same `<exam_name>`. The pre-existing `manifest.csv` is read, any student already listed there is skipped, and the newly-scanned students are appended. Crops/statuses for the prior batch survive untouched — use this to merge incremental scans into one in-progress assignment without overwriting work the teacher has already marked.

### Marker UI fallback

When `manifest.csv` is missing, the marker should fall back to "all unknown" — i.e. treat every question as attempted by default and skip the greying behaviour.

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
mc_pages: [3]                             # optional — packet-relative page indices that are MC
                                          # answer sheets. extract.py renders each as a whole-page
                                          # image MC_p<N>.png per student (no bbox, no manifest row).
                                          # qmark's MC grader picks them up by filename and shows
                                          # them next to the answer cells. Toggle in the region
                                          # editor with the "Mark this page as MC" button.
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
# With attempt detection (writes _blank/ and real status values in manifest.csv):
python extract.py workScans\10MATD_combinedTEST.pdf 10MATD_combinedTEST.yaml output\ --sheet-pdf Sheets\10MATD_combinedTEST.pdf

# Merge a later scan into an in-progress exam (existing students preserved):
python extract.py workScans\10MATD_later.pdf 10MATD_combinedTEST.yaml output\ --sheet-pdf Sheets\10MATD_combinedTEST.pdf --skip-existing

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
