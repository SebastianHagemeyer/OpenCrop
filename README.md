# exam-region-extractor

Crops per-question answer regions from scanned student exams (with QR-coded pages) into browsable image folders — one folder per student, `Q01.png ... QNN.png` inside.

## Setup

```
python -m pip install -r requirements.txt
```

## QR probe

Inspect what's encoded in a scan's QR codes:

```
python qr_probe.py path/to/scan.pdf
```

Prints decoded QR contents per page.

## Notes

Student data (PDF scans, extracted images, roster files) is gitignored and must never be committed.
