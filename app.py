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
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from orphan_dialog import OrphanRecoveryDialog
from page_viewer import PageViewerDialog
from qmark_theme import apply_qmark_theme
from scan_index import (
    _apply_overrides,
    group_into_students,
    index_pdf,
    load_skipped_students,
    save_sidecar_overrides,
    save_skipped_students,
)
from scan_view import ScanResultView

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
# Class roster xlsx supplied by the dashboard. Used by the orphan
# recovery dialog to populate the student-name dropdown with the names
# that the QR pipeline didn't pick up for this scan.
QMARK_CLASS_PATH = os.environ.get("QMARK_CLASS_PATH", "").strip()


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
    pages_indexed_signal = Signal(object)  # carries list[PageRecord]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Exam region extractor")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(820, 540)
        self.setMinimumSize(640, 360)

        # Cached page list from the last Check scan run, used to
        # populate the orphan recovery dialog without re-rendering
        # the PDF (~30s for 32 pages).
        self._last_pages = None
        self._last_pdf: Path | None = None
        # When Extract is clicked but no scan cache exists for the
        # current PDF, we stash the extract parameters here, kick off
        # Check scan, and resume after _on_pages_indexed fires. None
        # means no pending extract.
        self._pending_extract: dict | None = None

        self._build()
        self.log_signal.connect(self._append_log)
        self.busy_signal.connect(self._set_busy)
        self.tpl_path_signal.connect(self.tpl_edit.setText)
        self.pages_indexed_signal.connect(self._on_pages_indexed)

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
        self.btn_fix = QPushButton("Fix orphans...")
        self.btn_fix.clicked.connect(self._open_orphan_dialog)
        self.btn_fix.setEnabled(False)
        self.btn_fix.setToolTip(
            "Manually assign students to pages whose QR couldn't be "
            "read (torn paper, drawn-over QR, etc). Enabled after "
            "Check scan finds at least one orphan page."
        )
        self.btn_define = QPushButton("2. Define regions")
        self.btn_define.clicked.connect(self._define_regions)
        self.btn_extract = QPushButton("3. Extract crops")
        self.btn_extract.clicked.connect(self._extract)
        self.btn_open = QPushButton("Open output folder")
        self.btn_open.clicked.connect(self._open_output)
        for b in (self.btn_check, self.btn_fix, self.btn_define, self.btn_extract, self.btn_open):
            actions.addWidget(b)
        self.skip_existing_cb = QCheckBox("Skip students already in manifest")
        self.skip_existing_cb.setToolTip(
            "When on, Extract reads manifest.csv first and skips any "
            "student already listed there — newly-scanned students get "
            "appended without overwriting prior work."
        )
        actions.addWidget(self.skip_existing_cb)
        self.include_mc_cb = QCheckBox("Include MC pages")
        self.include_mc_cb.setChecked(True)
        self.include_mc_cb.setToolTip(
            "When on, the template's mc_pages list is honoured: each "
            "marked packet page is written as a whole-page image "
            "(MC_p<N>.png) per student so the MC grader can show what "
            "the student filled in. Untick to skip MC captures even if "
            "the template marks any."
        )
        actions.addWidget(self.include_mc_cb)
        actions.addStretch(1)
        outer.addLayout(actions)

        # Roster (top) + log (bottom) split: the roster is the primary
        # surface so a teacher can see the class at a glance instead of
        # parsing a debug dump; the log stays around for messages and
        # extract output but takes the smaller share by default.
        split = QSplitter(Qt.Vertical)
        self.scan_view = ScanResultView()
        self.scan_view.resolve_clicked.connect(self._open_orphan_dialog)
        self.scan_view.view_pages_requested.connect(self._view_pages_for)
        self.scan_view.skip_toggled.connect(self._toggle_skip)
        split.addWidget(self.scan_view)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log.setFont(QFont("Consolas", 10))
        self.log.setPlaceholderText("Messages from Check / Define / Extract will appear here.")
        split.addWidget(self.log)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 140])
        outer.addWidget(split, 1)

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
        # Fix-orphans button is only meaningful after a Check scan
        # surfaced at least one orphan; honour that state instead of
        # blindly re-enabling here.
        self.btn_fix.setEnabled((not busy) and self._has_orphans())
        self.status.showMessage(label if busy else "Ready.")

    def _has_orphans(self) -> bool:
        if not self._last_pages:
            return False
        return any(p.qr_status == "unknown" for p in self._last_pages)

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
        self._append_log(f"=== Checking {pdf.name} ===")

        def work() -> None:
            try:
                pages = index_pdf(pdf)
                # Hand the page list back to the UI thread; the view and
                # any auto-popping dialog are owned there.
                self.pages_indexed_signal.emit((pdf, pages))
            except Exception as e:
                self.log_signal.emit(f"ERROR: {e}\n")
            finally:
                self.busy_signal.emit(False, "")

        self._run_in_thread(work)

    def _refresh_view(self) -> None:
        """Re-render the class roster from the cached page list."""
        if self._last_pages is None:
            self.scan_view.clear()
            return
        skipped = load_skipped_students(self._last_pdf) if self._last_pdf else set()
        self.scan_view.set_pages(self._last_pages, skipped=skipped)

    def _view_pages_for(self, folder_name: str) -> None:
        """Open the page-viewer dialog for one roster row."""
        if not self._last_pdf or not self._last_pages:
            return
        groups = group_into_students(self._last_pages)
        match = next((g for g in groups if g.folder_name == folder_name), None)
        if match is None:
            return
        page_nums = [p.pdf_page_number for p in match.pages]
        dlg = PageViewerDialog(
            self._last_pdf, page_nums, title=match.student_name, parent=self,
        )
        dlg.exec()

    def _toggle_skip(self, folder_name: str, skip: bool) -> None:
        """Right-click → Skip from extraction (or undo)."""
        if not self._last_pdf:
            return
        current = load_skipped_students(self._last_pdf)
        if skip:
            current.add(folder_name)
            verb = "Skipping"
        else:
            current.discard(folder_name)
            verb = "Including"
        save_skipped_students(self._last_pdf, current)
        self._append_log(f"{verb} {folder_name} (extraction).")
        self._refresh_view()

    def _summarize_pages(self, pages) -> tuple[int, int]:
        """Return (n_students, n_orphan_pages) without mutating pages
        any more than group_into_students would have already."""
        groups = group_into_students(pages)
        n_students = sum(1 for g in groups if g.student_class != "UNKNOWN")
        n_orphan = sum(len(g.pages) for g in groups if g.student_class == "UNKNOWN")
        return n_students, n_orphan

    def _on_pages_indexed(self, payload) -> None:
        """UI-thread handler: cache page list, paint roster, pop dialog if needed."""
        pdf, pages = payload
        self._last_pdf = pdf
        self._last_pages = pages
        self._refresh_view()
        n_students, n_orphan = self._summarize_pages(pages)
        msg = f"Indexed {len(pages)} pages -> {n_students} student"
        msg += "s" if n_students != 1 else ""
        if n_orphan:
            msg += f"  ({n_orphan} orphan page{'s' if n_orphan != 1 else ''} need a name)"
        self._append_log(msg)
        self.btn_fix.setEnabled(n_orphan > 0)
        # If Extract is waiting on this scan, hand off — it owns the
        # orphan prompt for that path so we don't double-prompt the
        # user. Otherwise standalone Check auto-pops the dialog.
        if self._pending_extract is not None and self._pending_extract.get("pdf") == pdf:
            self._resume_pending_extract()
        elif n_orphan > 0:
            self._open_orphan_dialog()

    def _open_orphan_dialog(self) -> None:
        if not self._last_pages or not self._last_pdf:
            QMessageBox.information(
                self,
                "Nothing to recover",
                "Run Check scan first — the dialog needs the indexed page list.",
            )
            return
        if not self._has_orphans():
            QMessageBox.information(
                self,
                "No orphans",
                "Every page in this scan was matched to a student. Nothing to recover.",
            )
            return

        roster = Path(QMARK_CLASS_PATH) if QMARK_CLASS_PATH else None
        dlg = OrphanRecoveryDialog(
            self._last_pdf,
            self._last_pages,
            class_hint=QMARK_CLASS_NAME,
            roster_path=roster,
            parent=self,
        )
        result = dlg.exec()
        if result != QDialog.Accepted:
            self._append_log("Recovery dialog cancelled — orphans left as-is.")
            return
        if not dlg.selections:
            self._append_log("Recovery dialog closed without naming any pages — orphans left as-is.")
            return

        sidecar = save_sidecar_overrides(self._last_pdf, dlg.selections)
        _apply_overrides(self._last_pages, dlg.selections)

        # Summarise what just happened — name the recovered students so
        # the user sees the names that just landed in the roster.
        names = sorted({s["name"] for s in dlg.selections.values() if s.get("name")})
        n_pages = len(dlg.selections)
        n_students = len(names)
        names_blob = ", ".join(names) if names else "(none)"
        self._append_log(
            f"Recovered {n_students} student"
            f"{'s' if n_students != 1 else ''} "
            f"({names_blob}) from {n_pages} orphan page"
            f"{'s' if n_pages != 1 else ''} -> saved {sidecar.name}"
        )
        self.status.showMessage(
            f"Recovered {names_blob} — sidecar saved.", 6000
        )

        # Repaint the roster: the orphan banner disappears and the new
        # students appear as rows tagged 'manual'.
        self._refresh_view()
        self.btn_fix.setEnabled(self._has_orphans())

    # ---------- stage 2: define regions ----------

    def _define_regions(self) -> None:
        pdf = self._pdf_path()
        if not pdf:
            return
        self._set_busy(True, "Region editor open — finish and close it to continue.")
        self._append_log(f"=== Opening region editor on {pdf.name} ===")

        try:
            from make_template import TemplateEditor
        except Exception as e:
            self._append_log(f"ERROR importing region editor: {e}\n")
            self._set_busy(False, "")
            return

        # Hand the editor the page list we already indexed for Check
        # scan (if it's for the same PDF) so it skips its own ~30s
        # streaming decode. Cached pages already carry any sidecar
        # recovery, so the editor sees Shylah/Arvin too.
        cached = None
        if self._last_pdf == pdf and self._last_pages:
            cached = self._last_pages
            self._append_log("Reusing cached scan from Check scan — no re-indexing needed.")

        try:
            self._editor = TemplateEditor(pdf, cached_pages=cached)
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

        sheet_pdf: Path | None = None
        sheet_str = self.sheet_edit.text().strip()
        if sheet_str:
            candidate = Path(sheet_str)
            if not candidate.is_absolute():
                candidate = (HERE / candidate).resolve()
            if candidate.exists():
                sheet_pdf = candidate
            else:
                self._append_log(
                    f"WARNING: Sheet PDF {candidate} not found; "
                    "attempt detection disabled."
                )

        OUTPUT_DIR.mkdir(exist_ok=True)

        # Stash everything we need to run extract; either run now (if
        # we have a fresh scan cache) or queue it behind Check scan.
        self._pending_extract = {
            "pdf": pdf,
            "tpl": tpl,
            "exam_name": exam_name,
            "sheet_pdf": sheet_pdf,
            "skip_existing": self.skip_existing_cb.isChecked(),
            "include_mc_pages": self.include_mc_cb.isChecked(),
        }

        if self._last_pdf == pdf and self._last_pages is not None:
            # Cache for this PDF is fresh — go straight to the orphan
            # check and worker thread.
            self._resume_pending_extract()
            return

        # No cache (or cache is for a different PDF). Index this PDF
        # first; _on_pages_indexed will call _resume_pending_extract.
        self._append_log(
            "No scan cached for this PDF — running Check scan first."
        )
        self._check_scan()

    def _resume_pending_extract(self) -> None:
        """After Check scan has run, finish the queued extract request.

        Pops a prompt if orphans remain, then spawns the worker thread
        with cached_pages so extract.py skips re-indexing.
        """
        params = self._pending_extract
        if params is None:
            return
        pdf = params["pdf"]
        if self._last_pdf != pdf or self._last_pages is None:
            # Shouldn't happen, but bail rather than extract a stale PDF.
            self._pending_extract = None
            self._set_busy(False, "")
            return

        # Orphan gate: if any pages still aren't matched to a student,
        # surface the choice rather than silently writing orphan_pXX
        # folders.
        if self._has_orphans():
            box = QMessageBox(self)
            box.setWindowTitle("Unmatched pages")
            box.setIcon(QMessageBox.Warning)
            n_orphan = sum(1 for p in self._last_pages if p.qr_status == "unknown")
            box.setText(
                f"<b>{n_orphan} page(s) in this scan still aren't matched "
                f"to a student.</b>"
            )
            box.setInformativeText(
                "Extract them anyway (one orphan_p&lt;N&gt; folder per "
                "stray page), or resolve them now via the recovery "
                "dialog?"
            )
            resolve_btn = box.addButton("Resolve now…", QMessageBox.AcceptRole)
            anyway_btn = box.addButton("Extract anyway", QMessageBox.DestructiveRole)
            cancel_btn = box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(resolve_btn)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_btn:
                self._pending_extract = None
                self._append_log("Extract cancelled.")
                self._set_busy(False, "")
                return
            if clicked is resolve_btn:
                self._open_orphan_dialog()
                # User may have skipped some; we don't re-prompt — if
                # any orphans remain, we just proceed and they fall
                # through as orphan_pXX folders. The single-prompt
                # rule keeps the flow predictable.

        # Spawn the actual extraction. The work runs off the UI thread
        # because extract.py is CPU-heavy (PDF render + cv2 per crop).
        self._pending_extract = None
        self._set_busy(True, "Extracting crops...")
        self._append_log(
            f"=== Extracting {pdf.name} with {params['tpl'].name} "
            f"-> output/{params['exam_name']} ==="
        )
        if params["sheet_pdf"]:
            self._append_log(f"Blank reference: {params['sheet_pdf']}")
        else:
            self._append_log(
                "No Sheet PDF — attempt detection disabled "
                "(no _blank/ or attempts.csv will be written)."
            )
        if params["skip_existing"]:
            self._append_log("Skip-existing: on.")
        if not params["include_mc_pages"]:
            self._append_log("Include MC pages: off.")

        log_signal = self.log_signal
        busy_signal = self.busy_signal
        cached_pages = self._last_pages

        class _LogStream(io.TextIOBase):
            def write(self, s: str) -> int:
                if s:
                    log_signal.emit(s)
                return len(s)

            def flush(self) -> None:
                pass

        def work() -> None:
            try:
                from extract import extract as run_extract

                with contextlib.redirect_stdout(_LogStream()):
                    run_extract(
                        pdf, params["tpl"], OUTPUT_DIR, dpi=300,
                        exam_name_override=params["exam_name"],
                        sheet_pdf=params["sheet_pdf"],
                        skip_existing=params["skip_existing"],
                        include_mc_pages=params["include_mc_pages"],
                        cached_pages=cached_pages,
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
