# exam-region-extractor — context for a Claude agent

Python tool that takes a scanned exam PDF (each page has a QR code identifying the student) and produces per-question image crops organised by student. The intended downstream consumer is a separate **marker** tool (likely on its own git branch) that iterates over the crops and lets the teacher grade them.

This file is the handoff doc. It assumes you have not seen the conversation that built this.

## Repo layout

- `qr_probe.py` — one-shot CLI: prints the decoded QR string per page of a PDF.
- `scan_index.py` — library + CLI: decodes QRs and groups consecutive same-name pages into per-student packets. Has retry preprocessing (Otsu, rescaling) and a packet-size rebalancer for QR-decode failures.
- `make_template.py` — Tkinter GUI: pick a reference student, click+drag rectangles for each question on each page, save to YAML.
- `extract.py` — CLI: applies a template to a PDF, writes per-student crops + a manifest. Useful for archiving, sharing answers, or feeding a marker that doesn't want pymupdf/opencv as deps. NOT required for on-the-fly access (see below).
- `requirements.txt` — pymupdf, opencv-python, numpy, Pillow, PyYAML.
- `workScans/` — gitignored. Real student PDFs live here locally.
- `output/` — gitignored. Will hold extracted crops once `extract.py` exists.
- `templates/` — gitignored. Holds saved template YAMLs.

## QR code format on the scans

Each page has one QR encoding the literal string:

```
<class>/<firstname>
```

Examples from `workScans/workscan10Dpretest.pdf`: `10MATD/Ruby`, `10MATD/Ali`, `10MATD/Kierra`.

- `class` is constant across one PDF.
- `firstname` is the only student identifier (no surname, no ID).
- **The QR does not encode a page number.** Page-within-packet is inferred from PDF page order: consecutive pages with the same `firstname` belong to one student, in order. This is a known limitation — page numbers may be added to the QR in a future scan format.

QR detection rate observed at 250 DPI: 36/40 raw, 40/40 with Otsu+rescaling fallback. If `scan_index.py` still can't decode a page, it attributes the page to the running student and may move it to the next student via packet-size rebalancing (see `_rebalance` in `scan_index.py`).

## Output folder schema (what the marker iterates over)

```
output/
└── <exam_name>/                          # e.g. workscan10Dpretest/
    ├── manifest.csv                      # one row per student
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

- **Student folder name:** `<class>_<firstname>` (the only thing the QR gives us).
- **Question file name:** `Q01.png` ... `QNN.png`. Zero-padded so a file-manager sort matches numeric order. Always PNG (lossless — pen strokes stay sharp).
- **`manifest.csv`** columns: `student_class, student_name, packet_pdf_pages, qr_status_per_page, n_questions_extracted, notes`. `qr_status_per_page` is comma-joined per-page values from {`decoded`, `preprocessed`, `inferred`, `unknown`} — useful for flagging crops that came from a recovered/inferred page vs a confident decode.

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
python qr_probe.py workScans\workscan10Dpretest.pdf

# Show grouped student structure (one line per student)
python scan_index.py workScans\workscan10Dpretest.pdf

# Define question regions (opens Tkinter GUI; takes ~10–15s to index first)
python make_template.py workScans\workscan10Dpretest.pdf

# Extract per-question crops to disk (optional — see "on-the-fly" below)
python extract.py workScans\workscan10Dpretest.pdf workscan10Dpretest.yaml output\
# Custom DPI: --dpi 400
```

## On-the-fly crop access (preferred for the marker)

You don't need to pre-extract crops to disk. The marker can read crops directly from the PDF + template YAML. Sketch:

```python
from pathlib import Path
import cv2, numpy as np, pymupdf, yaml
from scan_index import group_into_students, index_pdf

template = yaml.safe_load(Path("workscan10Dpretest.yaml").read_text())
groups   = group_into_students(index_pdf(Path("workScans/workscan10Dpretest.pdf")))
doc      = pymupdf.open("workScans/workscan10Dpretest.pdf")

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
- If a future scan format encodes the page number in the QR (e.g. `10MATD/Ruby/1`), update `_parse_qr()` in `scan_index.py` to use the explicit page number instead of PDF-order inference, and consider dropping the rebalancing logic.
- Image coords in templates are normalized — never store pixel coords, they break across DPIs.
