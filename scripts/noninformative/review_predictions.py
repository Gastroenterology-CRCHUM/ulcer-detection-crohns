"""
scripts/review_predictions.py
==============================
Interactive GUI to review False Positives and False Negatives
from the non-informative classifier and optionally reassign
frames to the correct raw_inf/ folder.

Workflow
--------
    1. Load test_predictions.csv (produced by train_noninformative.py).
    2. Filter by outcome: FP, FN, or both.
    3. For each frame, see the image + prediction details.
    4. Press KEEP to accept the current label, or REASSIGN to move it
       to the correct class folder (and optionally a cause sub-folder).
    5. A reassignment log is saved to models_dir/reassignments.csv.

What "reassign" does
--------------------
    - Moves (or copies) the image file from its current location
      to the appropriate raw_inf/ subfolder.
    - Updates the reassignment log (does NOT rewrite the manifest —
      re-run preprocess_inf.py after a review session to rebuild it).

Usage
-----
    python scripts/review_predictions.py

    # Start directly on a predictions CSV
    python scripts/noninformative/review_predictions.py \\
        --predictions output/informative/models/test_predictions.csv \\
        --raw-inf     data/informative/raw \\
        --filter      FP FN         # default: both
        --move                      # move files (default: copy)
"""

from __future__ import annotations

import argparse
import re
import shutil
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import pandas as pd
from PIL import Image, ImageTk

from src.config.paths import get_default_paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NON_INFORMATIVE_CAUSES = [
    "Blur",
    "Low light",
    "Debris",
    "Bubbles",
    "Light reflection",
    "Tool",
    "Motion blur",
    "Tissue proximity",
    "Other / Combination",
]

COLOR = {
    "FP": "#e74c3c",  # wrongly flagged as non-informative
    "FN": "#f39c12",  # missed non-informative
    "TP": "#27ae60",
    "TN": "#2980b9",
}
BG = "#1a1a2e"
PANEL = "#16213e"
ACCENT = "#e94560"
TEXT = "#ecf0f1"
DIM = "#95a5a6"
INF_COL = "#2ecc71"
NONINF_COL = "#e74c3c"


def _safe_cause(cause: str) -> str:
    return re.sub(r"[^\w\-]", "_", cause.strip()).strip("_") or "Unknown"


def _hhmmss(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


# ---------------------------------------------------------------------------
# Reviewer application
# ---------------------------------------------------------------------------


class PredictionReviewer(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.title("Prediction Reviewer — FP / FN")
        self.configure(bg=BG)
        self.minsize(1100, 700)

        self.raw_inf_dir = Path(args.raw_inf)
        self.move_files = args.move
        self.filter_tags = set(args.filter)

        self._df: pd.DataFrame | None = None
        self._rows: list[dict] = []  # filtered rows
        self._pos: int = 0
        self._reassigned: list[dict] = []
        self._csv_mode: str = "predictions"
        self.filtered_dir = Path(args.filtered_dir) if hasattr(args, "filtered_dir") else None

        self._build_ui()
        self._bind_keys()

        if args.predictions:
            self.after(100, lambda: self._load_csv(Path(args.predictions)))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._build_menu()

        left = tk.Frame(self, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        bar = tk.Frame(left, bg="#0f0f1a", height=26)
        bar.pack(fill=tk.X)
        self.lbl_progress = tk.Label(
            bar, text="Load a predictions CSV", bg="#0f0f1a", fg=DIM, font=("Consolas", 10)
        )
        self.lbl_progress.pack(side=tk.LEFT, padx=8)
        self.lbl_timestamp = tk.Label(bar, text="", bg="#0f0f1a", fg=ACCENT, font=("Consolas", 10))
        self.lbl_timestamp.pack(side=tk.RIGHT, padx=8)

        right = tk.Frame(self, bg=PANEL, width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        self._build_panel(right)

    def _build_menu(self):
        m = tk.Menu(self)
        self.config(menu=m)
        fm = tk.Menu(m, tearoff=False)
        m.add_cascade(label="File", menu=fm)
        fm.add_command(label="Open predictions CSV…  Ctrl+O", command=lambda: self._load_csv())
        fm.add_command(label="Export reassignment log…", command=self._export_log)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.destroy)

    def _build_panel(self, parent):
        tk.Label(parent, text="REVIEW", bg=PANEL, fg=ACCENT, font=("Consolas", 13, "bold")).pack(
            pady=(14, 4)
        )

        self.lbl_file = tk.Label(
            parent, text="No file loaded", bg=PANEL, fg=DIM, font=("Consolas", 8), wraplength=310
        )
        self.lbl_file.pack(padx=8)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # Filter toggle
        tk.Label(
            parent, text="Show outcomes:", bg=PANEL, fg=TEXT, font=("Segoe UI", 9, "bold")
        ).pack(anchor=tk.W, padx=12)
        filt_row = tk.Frame(parent, bg=PANEL)
        filt_row.pack(fill=tk.X, padx=12, pady=4)
        self._filter_vars: dict[str, tk.BooleanVar] = {}
        for tag, color in [
            ("FP", NONINF_COL),
            ("FN", "#f39c12"),
            ("TP", INF_COL),
            ("TN", "#3498db"),
        ]:
            v = tk.BooleanVar(value=(tag in self.filter_tags))
            self._filter_vars[tag] = v
            tk.Checkbutton(
                filt_row,
                text=tag,
                variable=v,
                bg=PANEL,
                fg=color,
                selectcolor=PANEL,
                activebackground=PANEL,
                font=("Segoe UI", 9, "bold"),
                command=self._apply_filter,
            ).pack(side=tk.LEFT, padx=6)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        # Prediction info box
        self.info_frame = tk.Frame(parent, bg="#0f1520")
        self.info_frame.pack(fill=tk.X, padx=8, pady=4)
        self.lbl_info = tk.Label(
            self.info_frame,
            text="",
            bg="#0f1520",
            fg=TEXT,
            font=("Consolas", 9),
            justify=tk.LEFT,
            wraplength=300,
        )
        self.lbl_info.pack(padx=8, pady=6)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        # Reassign section
        tk.Label(parent, text="Reassign to:", bg=PANEL, fg=TEXT, font=("Segoe UI", 9, "bold")).pack(
            anchor=tk.W, padx=12
        )

        self.new_label_var = tk.StringVar(value="Informative")
        for val, color in [("Informative", INF_COL), ("Non-Informative", NONINF_COL)]:
            tk.Radiobutton(
                parent,
                text=val,
                variable=self.new_label_var,
                value=val,
                command=self._on_label_change,
                bg=PANEL,
                fg=color,
                selectcolor=PANEL,
                activebackground=PANEL,
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor=tk.W, padx=24)

        self.cause_frame = tk.Frame(parent, bg=PANEL)
        self.cause_frame.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(self.cause_frame, text="Cause:", bg=PANEL, fg=TEXT, font=("Segoe UI", 9)).pack(
            anchor=tk.W
        )
        self.cause_var = tk.StringVar(value=NON_INFORMATIVE_CAUSES[0])
        self.cause_combo = ttk.Combobox(
            self.cause_frame,
            textvariable=self.cause_var,
            values=NON_INFORMATIVE_CAUSES,
            state="readonly",
            width=32,
        )
        self.cause_combo.pack(fill=tk.X)
        self.cause_frame.pack_forget()

        btn_cfg = dict(relief=tk.FLAT, cursor="hand2", font=("Segoe UI", 9, "bold"), pady=6)

        tk.Button(
            parent,
            text="✔  KEEP current label  [K]",
            bg="#27ae60",
            fg="white",
            command=self._keep,
            **btn_cfg,
        ).pack(fill=tk.X, padx=12, pady=(8, 3))

        tk.Button(
            parent,
            text="↔  REASSIGN  [R]",
            bg=ACCENT,
            fg="white",
            command=self._reassign,
            **btn_cfg,
        ).pack(fill=tk.X, padx=12, pady=3)

        tk.Button(
            parent, text="⏭  SKIP  [S]", bg="#555577", fg="white", command=self._skip, **btn_cfg
        ).pack(fill=tk.X, padx=12, pady=3)

        tk.Button(
            parent, text="←  Back  [Left]", bg="#34495e", fg="white", command=self._back, **btn_cfg
        ).pack(fill=tk.X, padx=12, pady=(3, 10))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        self.lbl_stats = tk.Label(
            parent, text="", bg=PANEL, fg=DIM, font=("Consolas", 8), justify=tk.LEFT
        )
        self.lbl_stats.pack(padx=12, anchor=tk.W, pady=6)

        tk.Button(
            parent,
            text="📥  Export reassignment log",
            bg="#2c3e50",
            fg="white",
            command=self._export_log,
            relief=tk.FLAT,
            cursor="hand2",
            font=("Segoe UI", 9, "bold"),
            pady=5,
        ).pack(fill=tk.X, padx=12, pady=(0, 8), side=tk.BOTTOM)

        self.lbl_status = tk.Label(
            parent, text="", bg=PANEL, fg=DIM, font=("Consolas", 8), wraplength=310
        )
        self.lbl_status.pack(side=tk.BOTTOM, padx=8, pady=4)

    def _bind_keys(self):
        self.bind("k", lambda e: self._keep())
        self.bind("K", lambda e: self._keep())
        self.bind("r", lambda e: self._reassign())
        self.bind("R", lambda e: self._reassign())
        self.bind("s", lambda e: self._skip())
        self.bind("S", lambda e: self._skip())
        self.bind("<Left>", lambda e: self._back())
        self.bind("<Control-o>", lambda e: self._load_csv())

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_csv(self, path: Path | None = None):
        if path is None:
            p = filedialog.askopenfilename(
                title="Open predictions CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not p:
                return
            path = Path(p)

        self._df = pd.read_csv(path)

        # Detect CSV format
        if "outcome" in self._df.columns:
            # Format: test_predictions.csv (from train_noninformative.py)
            self._csv_mode = "predictions"
            required = {"image_path", "label", "pred_label", "outcome"}
        elif "category" in self._df.columns:
            # Format: review_queue.csv (filter_raw_ulcer.py) or predictions.csv (filter_frames.py)
            self._csv_mode = "review_queue"
            required = {"image_path", "pred_label", "pred_prob", "category"}
            # Normalize so the rest of the code works uniformly
            self._df["outcome"] = "UNCERTAIN"
            self._df["label"] = -1  # no ground truth available
            id_col = next((c for c in ("vid_id", "video_id") if c in self._df.columns), None)
            self._df["sample_id"] = self._df[id_col] if id_col else "?"
            self._df["cause"] = ""
        else:
            messagebox.showerror(
                "Unknown format", "CSV must have an 'outcome' or 'category' column."
            )
            return

        missing = required - set(self._df.columns)
        if missing:
            messagebox.showerror("Missing columns", f"CSV missing: {missing}")
            return

        self.lbl_file.config(text=f"[{self._csv_mode}]  {path.name}")
        self._pos = 0
        self._reassigned = []
        self._apply_filter()
        self._set_status(f"Loaded {len(self._df):,} rows — mode: {self._csv_mode}")

    def _apply_filter(self):
        if self._df is None:
            return
        if self._csv_mode == "review_queue":
            # All frames are uncertain -- no outcome filter applies
            self._rows = self._df.to_dict("records")
            # Hide FP/FN/TP/TN checkboxes (not relevant in this mode)
            for v in self._filter_vars.values():
                v.set(True)
        else:
            active = {tag for tag, v in self._filter_vars.items() if v.get()}
            self._rows = self._df[self._df["outcome"].isin(active)].to_dict("records")
        self._pos = 0
        self._show_current()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _show_current(self):
        if not self._rows:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 400,
                self.canvas.winfo_height() // 2 or 300,
                text="No frames match the current filter.",
                fill=DIM,
                font=("Consolas", 13),
            )
            self.lbl_progress.config(text="0 / 0")
            return

        if self._pos >= len(self._rows):
            self._session_done()
            return

        row = self._rows[self._pos]
        self._render_frame(row)
        self._update_info(row)
        self._update_progress()

    def _render_frame(self, row: dict):
        if self._csv_mode == "review_queue":
            if row.get("dest_path"):
                path = str(row["dest_path"])
            elif row.get("relative_path") and self.filtered_dir:
                path = str(self.filtered_dir / row["relative_path"])
            else:
                path = str(row["image_path"])
        else:
            path = str(row["image_path"])

        img = cv2.imread(path)
        if img is None:
            self._set_status(f"Cannot load: {path}", error=True)
            self._skip()
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        img = Image.fromarray(img).resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._tk_img)

        # Outcome badge
        tag = row.get("outcome", "?")
        color = COLOR.get(tag, DIM)
        self.canvas.create_rectangle(6, 6, 56, 26, fill=color, outline="")
        self.canvas.create_text(31, 16, text=tag, fill="white", font=("Segoe UI", 9, "bold"))

        # Prob bar (bottom)
        prob = float(row.get("pred_prob", 0))
        _bar_y = nh + (ch - nh) // 2 - 8 if nh < ch else ch - 16
        self.canvas.create_rectangle(0, ch - 12, cw, ch, fill="#0f0f1a", outline="")
        bar_color = NONINF_COL if row.get("pred_label") == 0 else INF_COL
        self.canvas.create_rectangle(0, ch - 12, int(cw * prob), ch, fill=bar_color, outline="")
        self.canvas.create_text(
            cw // 2, ch - 6, text=f"p = {prob:.3f}", fill="white", font=("Consolas", 8)
        )

        ts = row.get("timestamp_s") or row.get("frame_number", "")
        self.lbl_timestamp.config(text=f"frame {row.get('frame_number', '?')} · {ts}")

    def _update_info(self, row: dict):
        pred_name = "Informative" if row["pred_label"] == 1 else "Non-Informative"
        prob = float(row.get("pred_prob", 0))

        if self._csv_mode == "review_queue":
            sample = row.get("vid_id") or row.get("video_id", "?")
            seg = row.get("segment_id", "?")
            self.lbl_info.config(
                text=(
                    f"Video      : {sample}\n"
                    f"Segment    : {seg}\n"
                    f"Predicted  : {pred_name}  (p={prob:.3f})\n"
                    f"Uncertainty: {row.get('uncertainty', '?')}\n"
                    f"Status     : UNCERTAIN -> to classify"
                )
            )
            # No outcome to pre-select -- neutral default
            self.new_label_var.set("Informative")
            self.cause_frame.pack_forget()
        else:
            # Original predictions mode
            true_name = "Informative" if row["label"] == 1 else "Non-Informative"
            cause = row.get("cause", "") or "—"
            sample = row.get("sample_id", row.get("patient_id", "?"))
            outcome = row.get("outcome", "?")
            self.lbl_info.config(
                text=(
                    f"Sample   : {sample}\n"
                    f"GT label : {true_name}\n"
                    f"GT cause : {cause}\n"
                    f"Predicted: {pred_name}  (p={prob:.3f})\n"
                    f"Outcome  : {outcome}"
                )
            )
            if outcome == "FP":
                self.new_label_var.set("Informative")
                self.cause_frame.pack_forget()
            elif outcome == "FN":
                self.new_label_var.set("Non-Informative")
                self.cause_frame.pack(fill=tk.X, padx=12, pady=4)
            else:
                self.new_label_var.set(true_name)
                if true_name == "Non-Informative":
                    self.cause_frame.pack(fill=tk.X, padx=12, pady=4)
                else:
                    self.cause_frame.pack_forget()

    def _update_progress(self):
        done = self._pos
        total = len(self._rows)
        n_re = len(self._reassigned)
        self.lbl_progress.config(text=f"Frame {done + 1} / {total}  ·  ↔ {n_re} reassigned")
        n_kept = sum(1 for r in self._reassigned if r["action"] == "keep")
        n_rea = sum(1 for r in self._reassigned if r["action"] == "reassign")
        self.lbl_stats.config(text=f"Kept       : {n_kept}\nReassigned : {n_rea}")

    def _on_label_change(self):
        if self.new_label_var.get() == "Non-Informative":
            self.cause_frame.pack(fill=tk.X, padx=12, pady=4)
        else:
            self.cause_frame.pack_forget()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _keep(self):
        if not self._rows or self._pos >= len(self._rows):
            return
        row = self._rows[self._pos]
        self._log_action(row, action="keep", new_label=None, new_cause=None, new_path=None)
        self._pos += 1
        self._show_current()

    def _skip(self):
        if not self._rows or self._pos >= len(self._rows):
            return
        self._pos += 1
        self._show_current()

    def _back(self):
        if self._pos == 0:
            return
        self._pos -= 1
        # Remove last logged action if it matches
        if self._reassigned and self._reassigned[-1]["frame_idx"] == self._rows[self._pos].get(
            "frame_number"
        ):
            self._reassigned.pop()
        self._show_current()

    def _reassign(self):
        if not self._rows or self._pos >= len(self._rows):
            return
        row = self._rows[self._pos]
        new_label = self.new_label_var.get()
        new_cause = self.cause_var.get() if new_label == "Non-Informative" else ""

        if self._csv_mode == "review_queue":
            if self.filtered_dir is None:
                self._set_status("--filtered-dir is required in review_queue mode", error=True)
                return

            # Resolve source path
            if row.get("dest_path"):
                # Legacy filter_raw_ulcer.py format: dest_path inside uncertain/
                src_path = Path(str(row["dest_path"]))
                if not src_path.exists():
                    src_path = Path(str(row["image_path"]))
                try:
                    rel = src_path.relative_to(self.filtered_dir / "uncertain")
                except ValueError:
                    rel = Path(src_path.name)
            elif row.get("relative_path"):
                # filter_frames.py format: image at filtered_dir/relative_path
                rel = Path(str(row["relative_path"]))
                src_path = self.filtered_dir / rel
                if not src_path.exists():
                    src_path = Path(str(row["image_path"]))
            else:
                src_path = Path(str(row["image_path"]))
                rel = Path(src_path.name)

            if new_label == "Informative":
                dst_path = self.filtered_dir / "informative" / rel
            else:
                dst_path = self.filtered_dir / "non_informative" / rel

        else:
            # Predictions mode -- original behaviour, reassign to raw_inf/
            src_path = Path(str(row["image_path"]))
            if not src_path.exists():
                self._set_status(f"File not found: {src_path}", error=True)
                return
            if new_label == "Informative":
                dst_path = self.raw_inf_dir / "Informative" / src_path.name
            else:
                dst_path = (
                    self.raw_inf_dir / "Non-Informative" / _safe_cause(new_cause) / src_path.name
                )

        if not src_path.exists():
            self._set_status(f"File not found: {src_path}", error=True)
            return

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if self.move_files:
            src_path.replace(dst_path)
            op = "moved"
        else:
            shutil.copy2(src_path, dst_path)
            op = "copied"

        self._log_action(
            row, action="reassign", new_label=new_label, new_cause=new_cause, new_path=str(dst_path)
        )
        self._set_status(f"↔ {op} → {new_label}" + (f" [{new_cause}]" if new_cause else ""))
        self._pos += 1
        self._show_current()

    def _log_action(self, row: dict, action: str, new_label, new_cause, new_path):
        self._reassigned.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "action": action,
                "image_path": row.get("image_path"),
                "frame_idx": row.get("frame_number"),
                "sample_id": row.get("sample_id", row.get("patient_id")),
                "true_label": row.get("label"),
                "pred_label": row.get("pred_label"),
                "pred_prob": row.get("pred_prob"),
                "outcome": row.get("outcome"),
                "gt_cause": row.get("cause", ""),
                "new_label": new_label,
                "new_cause": new_cause,
                "new_path": new_path,
            }
        )

    # ------------------------------------------------------------------
    # Export & session done
    # ------------------------------------------------------------------

    def _export_log(self):
        if not self._reassigned:
            messagebox.showinfo("Nothing to export", "No actions logged yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save reassignment log",
            defaultextension=".csv",
            initialfile="reassignments.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        pd.DataFrame(self._reassigned).to_csv(path, index=False)
        self._set_status(f"Log saved → {Path(path).name}")
        messagebox.showinfo("Saved", f"Reassignment log saved to:\n{path}")

    def _session_done(self):
        self.canvas.delete("all")
        n_re = sum(1 for r in self._reassigned if r["action"] == "reassign")
        cw, ch = self.canvas.winfo_width() or 600, self.canvas.winfo_height() or 400
        self.canvas.create_text(
            cw // 2,
            ch // 2,
            text=f"✔  Review complete\n\n"
            f"{len(self._reassigned)} frames reviewed\n"
            f"{n_re} reassigned\n\n"
            f"Export the log, then re-run\n"
            f"preprocess_inf.py to rebuild the manifest.",
            fill=INF_COL,
            font=("Consolas", 13),
            justify=tk.CENTER,
        )
        self.lbl_progress.config(text=f"Done  ·  {n_re} reassigned")

    def _set_status(self, msg: str, error: bool = False):
        self.lbl_status.config(text=msg, fg=NONINF_COL if error else DIM)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    default_paths = get_default_paths()
    filtering_cfg = default_paths.get_task_output_config("ulcer_filtering")

    parser = argparse.ArgumentParser(
        description="Interactive FP/FN reviewer for non-informative classifier"
    )
    parser.add_argument("--predictions", default=None, help="Path to test_predictions.csv")
    parser.add_argument(
        "--raw-inf",
        default=str(default_paths.informative_raw_dir),
        help="Root raw_inf/ directory for reassignment",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default=["FP", "FN"],
        choices=["FP", "FN", "TP", "TN"],
        help="Outcomes to show (default: FP FN)",
    )
    parser.add_argument(
        "--move", action="store_true", help="Move files instead of copying on reassignment"
    )
    parser.add_argument(
        "--filtered-dir",
        default=str(filtering_cfg["output_dir"]),
        help="Output directory of the non-informative filter. "
        "In filter_raw_ulcer.py mode: contains uncertain/, informative/, non_informative/. "
        "In filter_frames.py mode: the filtered directory containing kept frames.",
    )
    args = parser.parse_args()

    app = PredictionReviewer(args)
    app.mainloop()
