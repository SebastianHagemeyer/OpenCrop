"""Build a Windows distributable of OpenCrop with Nuitka.

Usage:
    python -m pip install nuitka
    python build.py

Output: dist/app.dist/ — folder containing OpenCrop.exe plus all bundled
DLLs (PySide6, OpenCV, pymupdf, numpy, etc.). Zip the folder and ship it.
The end user does not need Python installed.

A first-time build downloads a C compiler if one isn't on PATH and takes
5-15 minutes. Subsequent builds are faster thanks to Nuitka's cache.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"


def main() -> None:
    DIST.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--enable-plugin=pyside6",
        "--include-data-files=paper.ico=paper.ico",
        "--windows-icon-from-ico=paper.ico",
        "--windows-console-mode=disable",
        "--output-dir=" + str(DIST),
        "--output-filename=OpenCrop.exe",
        "--company-name=OpenCrop",
        "--product-name=OpenCrop",
        "--product-version=0.1.0",
        "--file-version=0.1.0",
        "--file-description=Per-question crop extractor for QR-coded exam scans",
        "--assume-yes-for-downloads",
        "--remove-output",
        "app.py",
    ]
    print("Running:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=HERE)
    print(f"\nBuild complete. Distributable: {DIST / 'app.dist'}")
    print("Zip that folder and share it — end users run OpenCrop.exe inside.")


if __name__ == "__main__":
    main()
