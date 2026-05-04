"""Tkinter GUI for defining per-question regions on a reference student's pages.

Usage:
    python make_template.py [path/to/scan.pdf]

If no PDF is given, you'll be asked to pick one.

Workflow:
    1. The scan is indexed (QR-decoded, grouped into students).
    2. The first student is shown by default — change via the dropdown.
    3. Click + drag a rectangle over each question, in order. Q numbers
       auto-advance (Q01 → Q02 → ...).
    4. Use Prev / Next (or Left / Right arrows) to flip pages within the packet.
    5. Save Template writes a YAML keyed by student-relative page index, so it
       applies to every other student in the same exam.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pymupdf
import yaml
from PIL import Image, ImageTk

from scan_index import StudentGroup, group_into_students, index_pdf

INDEX_DPI = 200       # QR scan dpi (lower = faster startup)
DISPLAY_DPI = 180     # rendering dpi for the on-screen image
MIN_BBOX = 0.01       # ignore drags smaller than 1% of page (likely accidental)


class TemplateEditor:
    def __init__(self, root: tk.Tk, pdf_path: Path) -> None:
        self.root = root
        self.pdf_path = pdf_path
        self.root.title(f"Template editor — {self.pdf_path.name}")

        self.doc: pymupdf.Document | None = None
        self.groups: list[StudentGroup] = []
        self.current_group_idx = 0
        self.current_page_idx = 0
        self.regions: list[dict] = []         # [{q, page, bbox}]
        self.next_q_num = 1

        self.tk_img: ImageTk.PhotoImage | None = None
        self.img_dims: tuple[int, int] = (1, 1)
        self.display_scale = 1.0

        self.draw_start: tuple[int, int] | None = None
        self.preview_rect: int | None = None

        self._build_ui()
        self._load_pdf(self.pdf_path)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=4)
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Open PDF…", command=self._open_pdf_dialog).pack(side="left")
        ttk.Label(top, text="  Reference student: ").pack(side="left")
        self.student_var = tk.StringVar()
        self.student_combo = ttk.Combobox(top, textvariable=self.student_var, state="readonly", width=30)
        self.student_combo.bind("<<ComboboxSelected>>", lambda e: self._on_student_change())
        self.student_combo.pack(side="left")
        self.page_label = ttk.Label(top, text="")
        self.page_label.pack(side="left", padx=12)

        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        self.canvas = tk.Canvas(body, bg="gray80", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Configure>", lambda e: self._render_current_page())

        side = ttk.Frame(body, padding=8)
        side.pack(side="right", fill="y")
        side.configure(width=260)
        side.pack_propagate(False)

        ttk.Label(side, text="Defining:").pack(anchor="w")
        self.defining_label = ttk.Label(side, text="Q01", font=("TkDefaultFont", 14, "bold"))
        self.defining_label.pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="Regions defined:").pack(anchor="w")
        self.region_list = tk.Listbox(side, height=18)
        self.region_list.pack(fill="both", expand=True, pady=(2, 4))
        ttk.Button(side, text="Delete selected", command=self._delete_selected).pack(fill="x")

        ttk.Separator(side).pack(fill="x", pady=8)
        nav = ttk.Frame(side)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀ Prev page", command=self._prev_page).pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="Next page ▶", command=self._next_page).pack(side="left", expand=True, fill="x")

        ttk.Button(side, text="Save template…", command=self._save).pack(fill="x", pady=(8, 0))

        self.root.bind("<Left>", lambda e: self._prev_page())
        self.root.bind("<Right>", lambda e: self._next_page())

    # ---------- PDF loading / indexing ----------

    def _open_pdf_dialog(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        self._load_pdf(Path(path))

    def _load_pdf(self, path: Path) -> None:
        self.root.title(f"Template editor — {path.name}  (indexing…)")
        self.root.update_idletasks()
        if self.doc is not None:
            self.doc.close()
        self.doc = pymupdf.open(path)
        pages = index_pdf(path, dpi=INDEX_DPI)
        self.groups = [g for g in group_into_students(pages) if g.student_class != "UNKNOWN"]
        if not self.groups:
            messagebox.showerror("No students found", "Could not decode any QR codes in this PDF.")
            return
        self.pdf_path = path
        self.current_group_idx = 0
        self.current_page_idx = 0
        self.regions.clear()
        self.next_q_num = 1
        self.student_combo["values"] = [g.folder_name for g in self.groups]
        self.student_combo.current(0)
        self._refresh_region_list()
        self._render_current_page()
        self.root.title(f"Template editor — {path.name}")

    # ---------- State accessors ----------

    def _current_group(self) -> StudentGroup:
        return self.groups[self.current_group_idx]

    def _current_pdf_page_number(self) -> int:
        return self._current_group().pages[self.current_page_idx].pdf_page_number

    def _current_page_in_packet(self) -> int:
        return self.current_page_idx + 1

    # ---------- Rendering ----------

    def _render_current_page(self) -> None:
        if not self.groups or self.doc is None:
            return
        pdf_pg = self._current_pdf_page_number()
        page = self.doc[pdf_pg - 1]
        zoom = DISPLAY_DPI / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        pil = Image.fromarray(arr)
        self.img_dims = pil.size

        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        scale = min(cw / pil.width, ch / pil.height)
        self.display_scale = scale
        new_size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
        pil = pil.resize(new_size, Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        page_in_packet = self._current_page_in_packet()
        w, h = self.img_dims
        for r in self.regions:
            if r["page"] != page_in_packet:
                continue
            x0, y0, x1, y1 = r["bbox"]
            cx0, cy0 = x0 * w * scale, y0 * h * scale
            cx1, cy1 = x1 * w * scale, y1 * h * scale
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="red", width=2)
            self.canvas.create_text(
                cx0 + 4, cy0 + 4, anchor="nw", text=r["q"],
                fill="red", font=("TkDefaultFont", 12, "bold"),
            )

        n_pages = len(self._current_group().pages)
        self.page_label.config(text=f"Packet page {self._current_page_in_packet()} of {n_pages}  (PDF page {pdf_pg})")
        self.defining_label.config(text=f"Q{self.next_q_num:02d}")

    # ---------- Mouse interaction ----------

    def _on_press(self, event: tk.Event) -> None:
        self.draw_start = (event.x, event.y)
        if self.preview_rect is not None:
            self.canvas.delete(self.preview_rect)
        self.preview_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="blue", width=2, dash=(4, 2),
        )

    def _on_drag(self, event: tk.Event) -> None:
        if self.draw_start is None or self.preview_rect is None:
            return
        x0, y0 = self.draw_start
        self.canvas.coords(self.preview_rect, x0, y0, event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self.draw_start is None:
            return
        x0, y0 = self.draw_start
        x1, y1 = event.x, event.y
        if self.preview_rect is not None:
            self.canvas.delete(self.preview_rect)
        self.preview_rect = None
        self.draw_start = None

        w, h = self.img_dims
        s = self.display_scale
        nx0 = max(0.0, min(x0, x1) / s / w)
        ny0 = max(0.0, min(y0, y1) / s / h)
        nx1 = min(1.0, max(x0, x1) / s / w)
        ny1 = min(1.0, max(y0, y1) / s / h)
        if (nx1 - nx0) < MIN_BBOX or (ny1 - ny0) < MIN_BBOX:
            return

        self.regions.append({
            "q": f"Q{self.next_q_num:02d}",
            "page": self._current_page_in_packet(),
            "bbox": [round(nx0, 4), round(ny0, 4), round(nx1, 4), round(ny1, 4)],
        })
        self.next_q_num += 1
        self._refresh_region_list()
        self._render_current_page()

    # ---------- Region list management ----------

    def _refresh_region_list(self) -> None:
        self.region_list.delete(0, "end")
        for r in self.regions:
            self.region_list.insert("end", f"{r['q']}  page {r['page']}")

    def _renumber(self) -> None:
        for i, r in enumerate(self.regions, start=1):
            r["q"] = f"Q{i:02d}"
        self.next_q_num = len(self.regions) + 1

    def _delete_selected(self) -> None:
        sel = self.region_list.curselection()
        if not sel:
            return
        del self.regions[sel[0]]
        self._renumber()
        self._refresh_region_list()
        self._render_current_page()

    # ---------- Navigation ----------

    def _on_student_change(self) -> None:
        self.current_group_idx = self.student_combo.current()
        self.current_page_idx = 0
        self._render_current_page()

    def _prev_page(self) -> None:
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self._render_current_page()

    def _next_page(self) -> None:
        if self.current_page_idx < len(self._current_group().pages) - 1:
            self.current_page_idx += 1
            self._render_current_page()

    # ---------- Save ----------

    def _save(self) -> None:
        if not self.regions:
            messagebox.showwarning("Nothing to save", "Define some regions first.")
            return
        pages_per_student = len(self._current_group().pages)
        templates_dir = self.pdf_path.parent / "templates"
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            initialfile=f"{self.pdf_path.stem}.yaml",
            initialdir=str(templates_dir if templates_dir.exists() else self.pdf_path.parent),
            filetypes=[("YAML", "*.yaml")],
        )
        if not path:
            return
        regions_sorted = sorted(self.regions, key=lambda r: int(r["q"][1:]))
        template = {
            "exam": self.pdf_path.stem,
            "reference_student": self._current_group().folder_name,
            "pages_per_student": pages_per_student,
            "questions": [
                {"q": r["q"], "page": r["page"], "bbox": r["bbox"]}
                for r in regions_sorted
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(template, f, sort_keys=False)
        messagebox.showinfo("Saved", f"Wrote {len(regions_sorted)} regions to:\n{path}")


def main() -> None:
    if len(sys.argv) >= 2:
        pdf_path = Path(sys.argv[1])
    else:
        # No CLI arg → ask user
        root_tmp = tk.Tk()
        root_tmp.withdraw()
        chosen = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        root_tmp.destroy()
        if not chosen:
            sys.exit(0)
        pdf_path = Path(chosen)

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    root = tk.Tk()
    root.geometry("1280x900")
    TemplateEditor(root, pdf_path)
    root.mainloop()


if __name__ == "__main__":
    main()
