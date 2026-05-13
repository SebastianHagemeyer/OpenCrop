"""Launcher GUI for the exam-region-extractor pipeline.

Wraps the three pipeline stages (check scan -> define regions -> extract crops)
in one PySide6 window with a shared log panel. Each stage either calls the
underlying module directly (Check scan) or shells out to the existing CLI script
(Define regions, Extract) so the original tools stay untouched.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from qmark_theme import apply_qmark_theme
from scan_index import group_into_students, index_pdf

HERE = Path(__file__).resolve().parent
DEFAULT_PDF_REL = Path("workScans/10MATD_combinedTEST.pdf")
# When launched from the qmark dashboard, write crops into qmark's
# Student Work directory and use qmark's assignment name as the
# per-exam folder, so the marker picks them up directly.
_QMARK_WORK = os.environ.get("QMARK_WORK_DIR", "")
QMARK_ASSIGNMENT_NAME = os.environ.get("QMARK_ASSIGNMENT_NAME", "").strip()
QMARK_CLASS_NAME = os.environ.get("QMARK_CLASS_NAME", "").strip()
# Blank worksheet PDF supplied by the dashboard. When present, extract.py
# also crops a reference "empty" version of each question and classifies
# every student crop as attempted/unattempted/borderline — see attempts.csv
# under each exam's output folder.
QMARK_SHEET_PATH = os.environ.get("QMARK_SHEET_PATH", "").strip()


def _qmark_output_name() -> str:
    """Per-extraction subfolder name handed off by the dashboard.

    <Class>_<Assignment> when both are present (so the marker can find
    crops for the right cohort), or just <Assignment> if no class was
    given. Empty string when neither is set — caller falls back to the
    PDF stem.
    """
    if QMARK_CLASS_NAME and QMARK_ASSIGNMENT_NAME:
        return f"{QMARK_CLASS_NAME}_{QMARK_ASSIGNMENT_NAME}"
    return QMARK_ASSIGNMENT_NAME


def _writable_output_root() -> Path:
    """Per-user writable fallback for OpenCrop's output dir.

    HERE is read-only inside an MSIX install, so writing crops to
    HERE/output silently fails or gets virtualized into the per-package
    container. Use %LOCALAPPDATA%\\OpenCrop\\output instead when
    QMARK_WORK_DIR isn't supplied by the parent dashboard.
    """
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    d = Path(base) / "OpenCrop" / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


OUTPUT_DIR = Path(_QMARK_WORK) if _QMARK_WORK else _writable_output_root()
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
ICON_PATH = HERE / "paper.ico"


class Launcher(QMainWindow):
    log_signal = Signal(str)
    busy_signal = Signal(bool, str)
    tpl_path_signal = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Exam region extractor")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(820, 540)
        self.setMinimumSize(640, 360)

        self._build()
        self.log_signal.connect(self._append_log)
        self.busy_signal.connect(self._set_busy)
        self.tpl_path_signal.connect(self.tpl_edit.setText)

        default_pdf = HERE / DEFAULT_PDF_REL
        if default_pdf.exists():
            self.pdf_edit.setText(str(default_pdf))
            self.exam_name_edit.setText(default_pdf.stem)
            self._autofill_template()
        qmark_output = _qmark_output_name()
        if qmark_output:
            self.exam_name_edit.setText(qmark_output)
        if QMARK_SHEET_PATH:
            self.sheet_edit.setText(QMARK_SHEET_PATH)

    # ---------- layout ----------

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 4)
        outer.setSpacing(4)

        pdf_row = QHBoxLayout()
        pdf_row.addWidget(QLabel("Scan PDF:"))
        self.pdf_edit = QLineEdit()
        pdf_row.addWidget(self.pdf_edit, 1)
        pdf_btn = QPushButton("Browse...")
        pdf_btn.clicked.connect(self._browse_pdf)
        pdf_row.addWidget(pdf_btn)
        outer.addLayout(pdf_row)

        tpl_row = QHBoxLayout()
        tpl_row.addWidget(QLabel("Template YAML:"))
        self.tpl_edit = QLineEdit()
        tpl_row.addWidget(self.tpl_edit, 1)
        tpl_btn = QPushButton("Browse...")
        tpl_btn.clicked.connect(self._browse_template)
        tpl_row.addWidget(tpl_btn)
        outer.addLayout(tpl_row)

        sheet_row = QHBoxLayout()
        sheet_row.addWidget(QLabel("Sheet PDF:"))
        self.sheet_edit = QLineEdit()
        self.sheet_edit.setPlaceholderText(
            "Optional — blank worksheet for empty-question reference"
        )
        sheet_row.addWidget(self.sheet_edit, 1)
        sheet_btn = QPushButton("Browse...")
        sheet_btn.clicked.connect(self._browse_sheet)
        sheet_row.addWidget(sheet_btn)
        outer.addLayout(sheet_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output name:"))
        self.exam_name_edit = QLineEdit()
        out_row.addWidget(self.exam_name_edit, 1)
        out_row.addWidget(QLabel(" (subfolder under output/)"))
        outer.addLayout(out_row)

        actions = QHBoxLayout()
        self.btn_check = QPushButton("1. Check scan")
        self.btn_check.clicked.connect(self._check_scan)
        self.btn_define = QPushButton("2. Define regions")
        self.btn_define.clicked.connect(self._define_regions)
        self.btn_extract = QPushButton("3. Extract crops")
        self.btn_extract.clicked.connect(self._extract)
        self.btn_open = QPushButton("Open output folder")
        self.btn_open.clicked.connect(self._open_output)
        for b in (self.btn_check, self.btn_define, self.btn_extract, self.btn_open):
            actions.addWidget(b)
        self.skip_existing_cb = QCheckBox("Skip students already in manifest")
        self.skip_existing_cb.setToolTip(
            "When on, Extract reads manifest.csv first and skips any "
            "student already listed there — newly-scanned students get "
            "appended without overwriting prior work."
        )
        actions.addWidget(self.skip_existing_cb)
        actions.addStretch(1)
        outer.addLayout(actions)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log.setFont(QFont("Consolas", 10))
        outer.addWidget(self.log, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready.")

    # ---------- helpers ----------

    def _append_log(self, msg: str) -> None:
        if msg.endswith("\n"):
            msg = msg[:-1]
        self.log.appendPlainText(msg)

    def _set_busy(self, busy: bool, label: str = "") -> None:
        for b in (self.btn_check, self.btn_define, self.btn_extract):
            b.setEnabled(not busy)
        self.status.showMessage(label if busy else "Ready.")

    def _pdf_path(self) -> Path | None:
        s = self.pdf_edit.text().strip()
        if not s:
            QMessageBox.critical(self, "No PDF", "Pick a scan PDF first.")
            return None
        p = Path(s)
        if not p.is_absolute():
            p = (HERE / p).resolve()
        if not p.exists():
            QMessageBox.critical(self, "Missing file", f"PDF not found:\n{p}")
            return None
        return p

    def _template_search_paths(self, pdf_stem: str) -> list[Path]:
        """Candidate template locations in preference order.

        qmark's Sheets folder is checked first so a template saved there
        from Define regions wins over older copies in HERE / HERE/templates.
        """
        paths: list[Path] = []
        qmark_sheets = os.environ.get("QMARK_SHEETS_DIR", "").strip()
        if qmark_sheets:
            sheets_root = Path(qmark_sheets)
            paths.append(sheets_root / f"{pdf_stem}.yaml")
            paths.append(sheets_root / "templates" / f"{pdf_stem}.yaml")
        paths.append(HERE / f"{pdf_stem}.yaml")
        paths.append(HERE / "templates" / f"{pdf_stem}.yaml")
        return paths

    def _template_path(self, pdf: Path) -> Path:
        """Best-guess template path when the user hasn't picked one — the
        first candidate that exists, or the preferred save location."""
        for c in self._template_search_paths(pdf.stem):
            if c.exists():
                return c
        return self._template_search_paths(pdf.stem)[0]

    def _autofill_template(self) -> Path | None:
        pdf_str = self.pdf_edit.text().strip()
        if not pdf_str:
            return None
        pdf = Path(pdf_str)
        if not pdf.is_absolute():
            pdf = (HERE / pdf).resolve()
        for c in self._template_search_paths(pdf.stem):
            if c.exists():
                self.tpl_edit.setText(str(c))
                return c
        return None

    def _browse_pdf(self) -> None:
        # Prefer qmark's Data/Scans/ when launched from the dashboard, else
        # the local workScans/ scratch folder, else the OpenCrop folder.
        scans_dir = os.environ.get("QMARK_SCANS_DIR", "")
        if scans_dir and Path(scans_dir).is_dir():
            initial = Path(scans_dir)
        elif (HERE / "workScans").is_dir():
            initial = HERE / "workScans"
        else:
            initial = HERE
        picked, _ = QFileDialog.getOpenFileName(
            self, "Pick scan PDF", str(initial),
            "PDF files (*.pdf);;All files (*.*)",
        )
        if picked:
            self.pdf_edit.setText(picked)
            # When running under qmark, the dashboard's <Class>_<Assignment>
            # is the canonical output-folder name — don't clobber it with
            # the PDF stem.
            if not _qmark_output_name():
                self.exam_name_edit.setText(Path(picked).stem)
            self._autofill_template()

    def _browse_template(self) -> None:
        qmark_sheets = os.environ.get("QMARK_SHEETS_DIR", "").strip()
        if qmark_sheets and Path(qmark_sheets).is_dir():
            initial = Path(qmark_sheets)
        elif (HERE / "templates").is_dir():
            initial = HERE / "templates"
        else:
            initial = HERE
        picked, _ = QFileDialog.getOpenFileName(
            self, "Pick template YAML", str(initial),
            "YAML files (*.yaml *.yml);;All files (*.*)",
        )
        if picked:
            self.tpl_edit.setText(picked)

    def _browse_sheet(self) -> None:
        qmark_sheets = os.environ.get("QMARK_SHEETS_DIR", "").strip()
        current = self.sheet_edit.text().strip()
        if current and Path(current).parent.is_dir():
            initial = Path(current).parent
        elif qmark_sheets and Path(qmark_sheets).is_dir():
            initial = Path(qmark_sheets)
        else:
            initial = HERE
        picked, _ = QFileDialog.getOpenFileName(
            self, "Pick blank worksheet PDF", str(initial),
            "PDF files (*.pdf);;All files (*.*)",
        )
        if picked:
            self.sheet_edit.setText(picked)

    def _run_in_thread(self, work) -> None:
        threading.Thread(target=work, daemon=True).start()

    # ---------- stage 1: check scan ----------

    def _check_scan(self) -> None:
        pdf = self._pdf_path()
        if not pdf:
            return
        self._set_busy(True, "Indexing pages and decoding QRs...")
        self._append_log(f"\n=== Checking {pdf.name} ===")

        def work() -> None:
            try:
                pages = index_pdf(pdf)
                groups = group_into_students(pages)
                lines = [
                    f"Indexed {len(pages)} pages -> {len(groups)} student groups",
                    "",
                    f"{'group':<28} {'n':>3}  {'pdf pages':<25}  status",
                    "-" * 78,
                ]
                for g in groups:
                    pdf_pgs = ",".join(str(p.pdf_page_number) for p in g.pages)
                    statuses = ",".join(p.qr_status[:4] for p in g.pages)
                    lines.append(f"{g.folder_name:<28} {len(g.pages):>3}  {pdf_pgs:<25}  {statuses}")
                self.log_signal.emit("\n".join(lines) + "\n")
            except Exception as e:
                self.log_signal.emit(f"ERROR: {e}\n")
            finally:
                self.busy_signal.emit(False, "")

        self._run_in_thread(work)

    # ---------- stage 2: define regions ----------

    def _define_regions(self) -> None:
        pdf = self._pdf_path()
        if not pdf:
            return
        self._set_busy(True, "Region editor open — finish and close it to continue.")
        self._append_log(f"\n=== Opening region editor on {pdf.name} ===")

        try:
            from make_template import TemplateEditor
        except Exception as e:
            self._append_log(f"ERROR importing region editor: {e}\n")
            self._set_busy(False, "")
            return

        try:
            self._editor = TemplateEditor(pdf)
        except Exception as e:
            self._append_log(f"ERROR launching editor: {e}\n")
            self._set_busy(False, "")
            return
        self._editor.setAttribute(Qt.WA_DeleteOnClose, True)
        self._editor_pdf = pdf
        self._editor.destroyed.connect(self._on_editor_closed)
        self._editor.resize(1280, 900)
        self._editor.show()

    def _on_editor_closed(self, _obj: object | None = None) -> None:
        pdf = getattr(self, "_editor_pdf", None)
        self._editor = None
        self.log_signal.emit("Region editor closed.\n")
        if pdf is not None:
            # Pick whichever copy was written most recently — the editor
            # may have left an older template in HERE while the user
            # saved the new one into qmark's Sheets folder.
            existing = [c for c in self._template_search_paths(pdf.stem) if c.exists()]
            if existing:
                latest = max(existing, key=lambda p: p.stat().st_mtime)
                self.tpl_path_signal.emit(str(latest))
                self.log_signal.emit(f"Template found: {latest}\n")
        self.busy_signal.emit(False, "")

    # ---------- stage 3: extract ----------

    def _extract(self) -> None:
        pdf = self._pdf_path()
        if not pdf:
            return
        tpl_str = self.tpl_edit.text().strip()
        if tpl_str:
            tpl = Path(tpl_str)
            if not tpl.is_absolute():
                tpl = (HERE / tpl).resolve()
        else:
            tpl = self._template_path(pdf)
        if not tpl.exists():
            QMessageBox.critical(
                self,
                "Missing template",
                f"Template YAML not found:\n{tpl}\n\nPick one with the Browse button next to "
                "Template YAML, or run Define regions to create one.",
            )
            return
        self.tpl_edit.setText(str(tpl))

        exam_name = self.exam_name_edit.text().strip()
        if not exam_name:
            exam_name = pdf.stem
            self.exam_name_edit.setText(exam_name)
        if any(c in exam_name for c in '\\/:*?"<>|'):
            QMessageBox.critical(
                self,
                "Bad output name",
                'Output name cannot contain any of: \\ / : * ? " < > |',
            )
            return

        OUTPUT_DIR.mkdir(exist_ok=True)
        self._set_busy(True, "Extracting crops...")
        self._append_log(f"\n=== Extracting {pdf.name} with {tpl.name} -> output/{exam_name} ===")

        log_signal = self.log_signal
        busy_signal = self.busy_signal

        class _LogStream(io.TextIOBase):
            def write(self, s: str) -> int:
                if s:
                    log_signal.emit(s)
                return len(s)

            def flush(self) -> None:
                pass

        sheet_pdf: Path | None = None
        sheet_str = self.sheet_edit.text().strip()
        if sheet_str:
            candidate = Path(sheet_str)
            if not candidate.is_absolute():
                candidate = (HERE / candidate).resolve()
            if candidate.exists():
                sheet_pdf = candidate
                self._append_log(f"Blank reference: {candidate}")
            else:
                self._append_log(
                    f"WARNING: Sheet PDF {candidate} not found; "
                    "attempt detection disabled."
                )
        else:
            self._append_log(
                "No Sheet PDF — attempt detection disabled "
                "(no _blank/ or attempts.csv will be written)."
            )

        skip_existing = self.skip_existing_cb.isChecked()
        if skip_existing:
            self._append_log("Skip-existing: on (students already in manifest.csv will be left alone).")

        def work() -> None:
            try:
                from extract import extract as run_extract

                with contextlib.redirect_stdout(_LogStream()):
                    run_extract(
                        pdf, tpl, OUTPUT_DIR, dpi=300,
                        exam_name_override=exam_name,
                        sheet_pdf=sheet_pdf,
                        skip_existing=skip_existing,
                    )
                log_signal.emit("Extract finished.\n")
            except SystemExit as e:
                log_signal.emit(f"Extract aborted: {e}\n")
            except Exception as e:
                log_signal.emit(f"ERROR: {e}\n")
            finally:
                busy_signal.emit(False, "")

        self._run_in_thread(work)

    # ---------- open output ----------

    def _open_output(self) -> None:
        target = OUTPUT_DIR
        exam_name = self.exam_name_edit.text().strip()
        if not exam_name:
            pdf_str = self.pdf_edit.text().strip()
            if pdf_str:
                exam_name = Path(pdf_str).stem
        if exam_name:
            sub = OUTPUT_DIR / exam_name
            if sub.is_dir():
                target = sub
        if not target.exists():
            QMessageBox.information(self, "Not yet", f"{target} doesn't exist yet — run Extract first.")
            return
        try:
            os.startfile(str(target))
        except OSError as e:
            QMessageBox.critical(self, "Could not open", str(e))


def main() -> None:
    app = QApplication(sys.argv)
    apply_qmark_theme(app)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    win = Launcher()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
