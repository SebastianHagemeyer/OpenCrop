# exam-region-extractor

Crops per-question answer regions from scanned student exams into browsable image folders — one folder per student, `Q01.png` … `QNN.png` inside. Built for teachers who scan a stack of paper exams as one big PDF and want each student's work split out and ready for marking.

The catch: each printed page must carry a QR code that says who the page belongs to and where it sits in the student's packet. Once the QRs are there, this tool handles the rest — decoding, grouping pages into per-student packets (even when scans are out of order or some QRs are damaged), defining question regions once on a reference student, and slicing the same regions out of every other student's pages.

## What you get

- **`qr_probe.py`** — quick CLI: dump the decoded QR string per page of a PDF. Good first sanity check on a new scan.
- **`scan_index.py`** — library + CLI: decodes QRs, groups pages into per-student packets. Has a fallback path for QRs that don't decode on the first try (Otsu threshold + rescaling) and an inference pass that recovers pages from neighbours when decoding fails entirely.
- **`make_template.py`** — PySide6 GUI: pick a reference student, click+drag a rectangle for each question on each page, save a YAML template.
- **`extract.py`** — CLI: applies a template to a PDF and writes per-student crops + a manifest CSV.
- **`app.py`** — PySide6 launcher that wraps all of the above (Check scan → Define regions → Extract crops) with a shared log panel. This is the easiest way to drive the whole pipeline.

## Requirements

- Python 3.10+
- Windows, macOS, or Linux (developed on Windows; the GUIs are PySide6 so they run anywhere Qt does)
- A scanner that can produce a multi-page PDF
- A way to put a QR code on each printed page (see "QR format" below)

## Install

```
python -m pip install -r requirements.txt
```

Dependencies: `pymupdf`, `opencv-python`, `numpy`, `PyYAML`, `PySide6`.

## QR format on the printed page

Each page must have one QR encoding this string:

```
<class>/<firstname>/<page>/<total>
```

Examples: `10MATD/Alex/1/2`, `10MATD/Jordan/2/2`, `10DEMO/Sam/1/3`.

Field meanings:

| Field | What it is |
|---|---|
| `class` | Class identifier. Can vary across pages in the same PDF — one scan may mix multiple classes. |
| `firstname` | Student's first name. The only per-student identifier. Two students with the same first name in the same class will collide; add a disambiguation suffix when needed. |
| `page` | 1-based position of this page in the student's packet. |
| `total` | Total pages in the student's packet. |

A legacy 2-segment form (`<class>/<firstname>`, no page/total) is also accepted as a fallback. Mixing 4-segment and 2-segment QRs in one PDF is fine. With 2-segment QRs, packet order is reconstructed from the order pages appear in the PDF, so a legacy packet must be printed/scanned contiguously.

How to put the QR on the page is up to you. Common approaches: bake it into the worksheet template before printing, or print a strip of pre-generated QR stickers (one per student per page) and apply them by hand. Any standard QR generator works — the encoded payload is just the string above.

## Typical workflow

```
python -m pip install -r requirements.txt

# 1. Quick QR sanity check on a fresh scan
python qr_probe.py path/to/scan.pdf

# 2. See how the tool grouped pages into students
python scan_index.py path/to/scan.pdf

# 3. Open the region editor on the scan, click+drag a rectangle per question,
#    then Save Template (writes a YAML next to the PDF or in templates/).
python make_template.py path/to/scan.pdf

# 4. Extract per-question crops to disk
python extract.py path/to/scan.pdf path/to/template.yaml output/
#    Optional: --dpi 400  --exam-name custom_subfolder_name
```

Or run all four steps from one window:

```
python app.py
```

(On Windows, double-click `run.bat` to launch the GUI without a terminal window.)

## Output layout

```
output/
└── <exam_name>/
    ├── manifest.csv
    ├── 10MATD_Alex/
    │   ├── Q01.png
    │   ├── Q02.png
    │   └── …
    ├── 10MATD_Jordan/
    │   └── …
    └── …
```

- **Student folder name:** `<class>_<firstname>` (the only thing the QR gives us).
- **Question filename:** `Q01.png` … `QNN.png`. Zero-padded so a file-manager sort matches numeric order. Always PNG (lossless — pen strokes stay sharp).
- **`manifest.csv`** columns: `student_class`, `student_name`, `packet_pdf_pages`, `qr_status_per_page`, `n_questions_extracted`, `notes`. `qr_status_per_page` is comma-joined per-page values from `decoded` / `preprocessed` / `inferred` / `unknown` — useful for flagging crops that came from a recovered or inferred page vs a confident decode.

## Template YAML

Produced by `make_template.py`, consumed by `extract.py`:

```yaml
exam: my_exam
reference_student: 10MATD_Alex     # which student was used to define the regions
pages_per_student: 2
questions:
  - q: Q01
    page: 1                         # 1 = first page of packet, NOT absolute PDF page
    bbox: [0.05, 0.10, 0.95, 0.25]  # [x0, y0, x1, y1] normalized to [0, 1]
  - q: Q02
    page: 1
    bbox: [0.05, 0.27, 0.95, 0.42]
  …
```

- `bbox` is **normalized** to page width/height, so the same template works at any render DPI.
- `page` is **student-relative**, not absolute PDF page, so the same template applies to every student in the same exam.

## Using the crops without pre-extracting

If you're building a downstream tool (e.g. a marking UI) you don't have to write crops to disk first. You can read them on the fly straight from the PDF + template YAML:

```python
from pathlib import Path
import cv2, numpy as np, pymupdf, yaml
from scan_index import group_into_students, index_pdf

template = yaml.safe_load(Path("template.yaml").read_text())
groups   = group_into_students(index_pdf(Path("scan.pdf")))
doc      = pymupdf.open("scan.pdf")
zoom     = pymupdf.Matrix(300 / 72.0, 300 / 72.0)

def crop_for(group, q):
    pdf_pg = group.pages[q["page"] - 1].pdf_page_number
    pix    = doc[pdf_pg - 1].get_pixmap(matrix=zoom, alpha=False)
    img    = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    img    = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w   = img.shape[:2]
    x0, y0, x1, y1 = q["bbox"]
    return img[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
```

A page render is ~200–500ms at 300 DPI; subsequent crops on the same page are free. Cache the rendered page if you're showing several questions from one page in a row.

Pre-extraction (`extract.py`) is the right call when you want to archive graded work, hand off without the original PDF, or feed a tool that doesn't want pymupdf/opencv as dependencies.

## Privacy

Student PDFs, extracted crops, templates, and rosters are real student data. The repo's `.gitignore` excludes them by default:

```
workScans/
output/
templates/
roster*.csv
*.pdf
```

Don't override this with `git add -f`. If you fork this repo for your own school, keep the same exclusions.

## Build a standalone Windows executable

You can ship OpenCrop without asking the recipient to install Python. The build is driven by [Nuitka](https://nuitka.net), which compiles the Python sources to native code and bundles every dependency (PySide6, OpenCV, pymupdf, numpy, etc.) into a folder.

```
python -m pip install nuitka
python build.py
```

First-time builds take 5–15 minutes (Nuitka may auto-download a C compiler if none is on PATH). Subsequent builds are faster because Nuitka caches compiled C objects.

Output: `dist/app.dist/` — a self-contained folder with `OpenCrop.exe` and all required DLLs. Zip the folder and share it; the recipient runs `OpenCrop.exe` directly. No Python install required on their side.

The compiled `.exe` uses `paper.ico` for both the file icon and the window icon.

## License

MIT — see [LICENSE](LICENSE).
