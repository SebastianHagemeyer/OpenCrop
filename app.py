"""Launcher GUI for the exam-region-extractor pipeline.

Wraps the three pipeline stages (check scan -> define regions -> extract crops)
in one PySide6 window with a shared log panel. Each stage either calls the
underlying module directly (Check scan) or shells out to the existing CLI script
(Define regions, Extract) so the original tools stay untouched.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
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

from scan_index import group_into_students, index_pdf

HERE = Path(__file__).resolve().parent
DEFAULT_PDF_REL = Path("workScans/10MATD_combinedTEST.pdf")
OUTPUT_DIR = HERE / "output"


class Launcher(QMainWindow):
    log_signal = Signal(str)
    busy_signal = Signal(bool, str)
    tpl_path_signal = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Exam region extractor")
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

    def _template_path(self, pdf: Path) -> Path:
        return HERE / f"{pdf.stem}.yaml"

    def _autofill_template(self) -> Path | None:
        pdf_str = self.pdf_edit.text().strip()
        if not pdf_str:
            return None
        pdf = Path(pdf_str)
        if not pdf.is_absolute():
            pdf = (HERE / pdf).resolve()
        for c in (HERE / f"{pdf.stem}.yaml", HERE / "templates" / f"{pdf.stem}.yaml"):
            if c.exists():
                self.tpl_edit.setText(str(c))
                return c
        return None

    def _browse_pdf(self) -> None:
        initial = HERE / "workScans" if (HERE / "workScans").is_dir() else HERE
        picked, _ = QFileDialog.getOpenFileName(
            self, "Pick scan PDF", str(initial),
            "PDF files (*.pdf);;All files (*.*)",
        )
        if picked:
            self.pdf_edit.setText(picked)
            self.exam_name_edit.setText(Path(picked).stem)
            self._autofill_template()

    def _browse_template(self) -> None:
        initial = HERE / "templates" if (HERE / "templates").is_dir() else HERE
        picked, _ = QFileDialog.getOpenFileName(
            self, "Pick template YAML", str(initial),
            "YAML files (*.yaml *.yml);;All files (*.*)",
        )
        if picked:
            self.tpl_edit.setText(picked)

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
        script = HERE / "make_template.py"
        if not script.exists():
            QMessageBox.critical(self, "Missing script", f"Cannot find {script}")
            return
        self._set_busy(True, "Region editor open — finish and close it to continue.")
        self._append_log(f"\n=== Opening region editor on {pdf.name} ===")

        def work() -> None:
            try:
                proc = subprocess.run([sys.executable, str(script), str(pdf)], cwd=str(HERE))
                self.log_signal.emit(f"Region editor closed (exit code {proc.returncode}).\n")
                for c in (HERE / f"{pdf.stem}.yaml", HERE / "templates" / f"{pdf.stem}.yaml"):
                    if c.exists():
                        self.tpl_path_signal.emit(str(c))
                        self.log_signal.emit(f"Template found: {c.name}\n")
                        break
            except Exception as e:
                self.log_signal.emit(f"ERROR launching editor: {e}\n")
            finally:
                self.busy_signal.emit(False, "")

        self._run_in_thread(work)

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

        script = HERE / "extract.py"
        OUTPUT_DIR.mkdir(exist_ok=True)
        self._set_busy(True, "Extracting crops...")
        self._append_log(f"\n=== Extracting {pdf.name} with {tpl.name} -> output/{exam_name} ===")

        def work() -> None:
            try:
                proc = subprocess.Popen(
                    [
                        sys.executable, "-u", str(script),
                        str(pdf), str(tpl), str(OUTPUT_DIR),
                        "--exam-name", exam_name,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(HERE),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_signal.emit(line)
                proc.wait()
                self.log_signal.emit(f"Extract finished (exit code {proc.returncode}).\n")
            except Exception as e:
                self.log_signal.emit(f"ERROR: {e}\n")
            finally:
                self.busy_signal.emit(False, "")

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
    win = Launcher()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
