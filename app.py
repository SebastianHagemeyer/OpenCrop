"""Launcher GUI for the exam-region-extractor pipeline.

Wraps the three pipeline stages (check scan -> define regions -> extract crops)
in one Tkinter window with a shared log panel. Each stage either calls the
underlying module directly (Check scan) or shells out to the existing CLI script
(Define regions, Extract) so the original tools stay untouched.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from scan_index import group_into_students, index_pdf

HERE = Path(__file__).resolve().parent
DEFAULT_PDF_REL = Path("workScans/10MATD_combinedTEST.pdf")
OUTPUT_DIR = HERE / "output"


class Launcher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Exam region extractor")
        root.geometry("820x540")
        root.minsize(640, 360)

        default_pdf = HERE / DEFAULT_PDF_REL
        self.pdf_var = tk.StringVar(value=str(default_pdf) if default_pdf.exists() else "")
        self.tpl_var = tk.StringVar(value="")
        self.exam_name_var = tk.StringVar(value=default_pdf.stem if default_pdf.exists() else "")
        self.status_var = tk.StringVar(value="Ready.")
        self._log_queue: queue.Queue[str] = queue.Queue()

        self._build()
        if self.pdf_var.get():
            self._autofill_template()
        self._poll_log_queue()

    # ---------- layout ----------

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 10, 10, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Scan PDF:    ").pack(side="left")
        ttk.Entry(top, textvariable=self.pdf_var).pack(side="left", fill="x", expand=True, padx=(8, 4))
        ttk.Button(top, text="Browse...", command=self._browse_pdf).pack(side="left")

        tpl_row = ttk.Frame(self.root, padding=(10, 0, 10, 4))
        tpl_row.pack(fill="x")
        ttk.Label(tpl_row, text="Template YAML:").pack(side="left")
        ttk.Entry(tpl_row, textvariable=self.tpl_var).pack(side="left", fill="x", expand=True, padx=(8, 4))
        ttk.Button(tpl_row, text="Browse...", command=self._browse_template).pack(side="left")

        out_row = ttk.Frame(self.root, padding=(10, 0, 10, 4))
        out_row.pack(fill="x")
        ttk.Label(out_row, text="Output name:  ").pack(side="left")
        ttk.Entry(out_row, textvariable=self.exam_name_var).pack(side="left", fill="x", expand=True, padx=(8, 4))
        ttk.Label(out_row, text=" (subfolder under output/)").pack(side="left")

        actions = ttk.Frame(self.root, padding=(10, 4))
        actions.pack(fill="x")
        self.btn_check = ttk.Button(actions, text="1. Check scan", command=self._check_scan)
        self.btn_define = ttk.Button(actions, text="2. Define regions", command=self._define_regions)
        self.btn_extract = ttk.Button(actions, text="3. Extract crops", command=self._extract)
        self.btn_open = ttk.Button(actions, text="Open output folder", command=self._open_output)
        for b in (self.btn_check, self.btn_define, self.btn_extract, self.btn_open):
            b.pack(side="left", padx=(0, 6))

        log_frame = ttk.Frame(self.root, padding=(10, 4, 10, 4))
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, wrap="none", height=20, font=("Consolas", 10))
        self.log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_frame, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.config(yscrollcommand=sb.set, state="disabled")

        ttk.Label(self.root, textvariable=self.status_var, padding=(10, 4), relief="sunken").pack(fill="x")

    # ---------- helpers ----------

    def _log(self, msg: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", msg if msg.endswith("\n") else msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _set_busy(self, busy: bool, label: str = "") -> None:
        state = "disabled" if busy else "normal"
        for b in (self.btn_check, self.btn_define, self.btn_extract):
            b.config(state=state)
        self.status_var.set(label if busy else "Ready.")

    def _pdf_path(self) -> Path | None:
        s = self.pdf_var.get().strip()
        if not s:
            messagebox.showerror("No PDF", "Pick a scan PDF first.")
            return None
        p = Path(s)
        if not p.is_absolute():
            p = (HERE / p).resolve()
        if not p.exists():
            messagebox.showerror("Missing file", f"PDF not found:\n{p}")
            return None
        return p

    def _template_path(self, pdf: Path) -> Path:
        return HERE / f"{pdf.stem}.yaml"

    def _autofill_template(self) -> Path | None:
        """Look for a YAML matching the current PDF and set tpl_var if found."""
        pdf_str = self.pdf_var.get().strip()
        if not pdf_str:
            return None
        pdf = Path(pdf_str)
        if not pdf.is_absolute():
            pdf = (HERE / pdf).resolve()
        for c in (HERE / f"{pdf.stem}.yaml", HERE / "templates" / f"{pdf.stem}.yaml"):
            if c.exists():
                self.tpl_var.set(str(c))
                return c
        return None

    def _browse_pdf(self) -> None:
        initial = HERE / "workScans" if (HERE / "workScans").is_dir() else HERE
        picked = filedialog.askopenfilename(
            title="Pick scan PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialdir=str(initial),
        )
        if picked:
            self.pdf_var.set(picked)
            self.exam_name_var.set(Path(picked).stem)
            self._autofill_template()

    def _browse_template(self) -> None:
        initial = HERE / "templates" if (HERE / "templates").is_dir() else HERE
        picked = filedialog.askopenfilename(
            title="Pick template YAML",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=str(initial),
        )
        if picked:
            self.tpl_var.set(picked)

    def _run_in_thread(self, work) -> None:
        threading.Thread(target=work, daemon=True).start()

    # ---------- stage 1: check scan ----------

    def _check_scan(self) -> None:
        pdf = self._pdf_path()
        if not pdf:
            return
        self._set_busy(True, "Indexing pages and decoding QRs...")
        self._log(f"\n=== Checking {pdf.name} ===")

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
                self._log_queue.put("\n".join(lines) + "\n")
            except Exception as e:
                self._log_queue.put(f"ERROR: {e}\n")
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        self._run_in_thread(work)

    # ---------- stage 2: define regions ----------

    def _define_regions(self) -> None:
        pdf = self._pdf_path()
        if not pdf:
            return
        script = HERE / "make_template.py"
        if not script.exists():
            messagebox.showerror("Missing script", f"Cannot find {script}")
            return
        self._set_busy(True, "Region editor open — finish and close it to continue.")
        self._log(f"\n=== Opening region editor on {pdf.name} ===")

        def work() -> None:
            try:
                proc = subprocess.run([sys.executable, str(script), str(pdf)], cwd=str(HERE))
                self._log_queue.put(f"Region editor closed (exit code {proc.returncode}).\n")
                found = self._autofill_template()
                if found is not None:
                    self._log_queue.put(f"Template found: {found.name}\n")
            except Exception as e:
                self._log_queue.put(f"ERROR launching editor: {e}\n")
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        self._run_in_thread(work)

    # ---------- stage 3: extract ----------

    def _extract(self) -> None:
        pdf = self._pdf_path()
        if not pdf:
            return
        tpl_str = self.tpl_var.get().strip()
        if tpl_str:
            tpl = Path(tpl_str)
            if not tpl.is_absolute():
                tpl = (HERE / tpl).resolve()
        else:
            tpl = self._template_path(pdf)
        if not tpl.exists():
            messagebox.showerror(
                "Missing template",
                f"Template YAML not found:\n{tpl}\n\nPick one with the Browse button next to "
                "Template YAML, or run Define regions to create one.",
            )
            return
        self.tpl_var.set(str(tpl))

        exam_name = self.exam_name_var.get().strip()
        if not exam_name:
            exam_name = pdf.stem
            self.exam_name_var.set(exam_name)
        if any(c in exam_name for c in '\\/:*?"<>|'):
            messagebox.showerror(
                "Bad output name",
                'Output name cannot contain any of: \\ / : * ? " < > |',
            )
            return

        script = HERE / "extract.py"
        OUTPUT_DIR.mkdir(exist_ok=True)
        self._set_busy(True, "Extracting crops...")
        self._log(f"\n=== Extracting {pdf.name} with {tpl.name} -> output/{exam_name} ===")

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
                    self._log_queue.put(line)
                proc.wait()
                self._log_queue.put(f"Extract finished (exit code {proc.returncode}).\n")
            except Exception as e:
                self._log_queue.put(f"ERROR: {e}\n")
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        self._run_in_thread(work)

    # ---------- open output ----------

    def _open_output(self) -> None:
        target = OUTPUT_DIR
        exam_name = self.exam_name_var.get().strip()
        if not exam_name:
            pdf_str = self.pdf_var.get().strip()
            if pdf_str:
                exam_name = Path(pdf_str).stem
        if exam_name:
            sub = OUTPUT_DIR / exam_name
            if sub.is_dir():
                target = sub
        if not target.exists():
            messagebox.showinfo("Not yet", f"{target} doesn't exist yet — run Extract first.")
            return
        try:
            os.startfile(str(target))
        except OSError as e:
            messagebox.showerror("Could not open", str(e))

    # ---------- log pump ----------

    def _poll_log_queue(self) -> None:
        try:
            while True:
                self._log(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)


def main() -> None:
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
