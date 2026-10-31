#!/usr/bin/env python3
# Copyright (C) 2026 Lorenzo Orsini
# Author(s):
#   - Lorenzo Orsini, lorenzo.orsini3@gmail.com
#   - Alex Carnoli, alex.carnoli@gmail.com
# This file is part of CpG Island Predictor (CIP).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
gui.py — Graphical front end for CIP (Tkinter, no extra dependencies).

Layout::

    ┌ Open FASTA…  Outputs  Clear │ [ ] BED  [ ] GFF3 │ Help ┐
    ├ console (terminal look) ──────────────────────┬ side panel ───────────┤
    │ - Sequence 'seq_2' is a CpG island.           │ MODEL   version, …    │
    │         Probability of CpG island: 0.9777     │ LAST RUN  counts, bar │
    ├ input line ───────────────────────────────────┴───────────────────────┤

CIP.py is not duplicated: this window imports it and calls the same functions
(``predict_from_fasta`` and friends). Whatever they print — progress bar,
warnings, coloured results — is captured and shown in the console.

Two ways to run an analysis:
  - **Open FASTA…** picks a file; the BED / GFF3 switches decide the extra outputs.
  - type ``path [--bed] [--gff3]`` in the input line, as in the terminal version
    (quotes are accepted for paths with spaces).
Other lines: ``/help``, ``/clear``, ``/quit``.

Keys:  Enter run · Up/Down history · Ctrl+O open · Ctrl+L clear the screen ·
       Ctrl+= / Ctrl+- font size · F11 full screen on/off.

Start with:  python3 gui.py
"""
import contextlib, os, queue, re, shlex, signal, subprocess, sys, threading, time, traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

# ── Colours ───────────────────────────────────────────────────────────────────
BG        = "#1e1e1e"
BG_PANEL  = "#252526"
BG_INPUT  = "#2a2a2b"
BG_BTN    = "#333336"
BG_BTN_HI = "#44444a"
FG        = "#d4d4d4"
FG_DIM    = "#858585"
GREEN     = "#98c379"
CYAN      = "#56b6c2"
BLUE      = "#7aa2c8"
YELLOW    = "#e5c07b"
RED       = "#e06c75"

MONO_CANDIDATES = ("DejaVu Sans Mono", "Liberation Mono", "Ubuntu Mono",
                   "Noto Sans Mono", "Cousine", "Courier New")
FASTA_TYPES = [("FASTA", "*.fa *.fasta *.fna *.ffn *.faa *.frn *.txt"), ("All files", "*")]
PANEL_W = 250

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

HELP_TEXT = """\
Usage
  Open FASTA…\tchoose a file and analyse it (the BED / GFF3 switches add output files)
  <path> [--bed] [--gff3]\ttype a path in the input line; use quotes if it has spaces
  Outputs\topen the outs/ folder with the saved results
  /help\tthis text
  /clear\tclear the screen
  /quit\texit

Every run saves outs/<date>_<id>.csv, plus .bed / .gff3 when requested."""


class ConsoleStream:
    """
    File-like object that turns ``print()`` / tqdm output into console lines.

    Understands ``\\r`` (the text after it replaces the current line: progress
    bars) and the few ANSI colours colorama emits (green = CpG island, red = not
    an island, grey = dim).  Lines starting with WARNING: / ERROR: get their own colour.
    """

    encoding = "utf-8"          # lets tqdm draw its bar with block characters, as in a terminal

    def __init__(self, emit, shared=None):
        self.emit = emit
        self.line = ""
        self.shared = shared if shared is not None else {"owner": None}

    @classmethod
    def pair(cls, emit):
        """(stdout, stderr) streams that share the console: a warning printed while a
        progress bar is being drawn goes on its own line instead of being glued to the bar."""
        shared = {"owner": None}
        return cls(emit, shared), cls(emit, shared)

    def _commit(self) -> None:
        self._show("\n")
        self.line = ""

    def _tag(self, raw: str) -> str:
        if "\x1b[92m" in raw:
            return "yes"
        if "\x1b[91m" in raw:
            return "no"
        if "\x1b[90m" in raw:
            return "dim"
        text = _ANSI.sub("", raw)
        if text.startswith("WARNING:"):
            return "warn"
        if text.startswith("ERROR"):
            return "error"
        return "out"

    def _show(self, end: str) -> None:
        self.emit(_ANSI.sub("", self.line), self._tag(self.line), end)

    def write(self, s: str) -> int:
        owner = self.shared["owner"]
        if owner is not None and owner is not self and owner.line:
            owner._commit()
        self.shared["owner"] = self
        for part in re.split(r"(\r\n|\r|\n)", s):
            if part in ("\n", "\r\n"):
                self._show("\n")
                self.line = ""
            elif part == "\r":
                self.line = ""               # what follows replaces the live line
            elif part:
                self.line += part
                self._show("\r")             # live line: replaced by the next write
        return len(s)

    def finish(self) -> None:
        if self.line:
            self._show("\n")
            self.line = ""

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class CipApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CIP")
        self.root.geometry("1100x680")
        self.root.minsize(760, 420)
        self.root.configure(bg=BG)
        try:
            self.root.attributes("-zoomed", True)        # start maximized (X11 / XWayland)
        except tk.TclError:
            pass
        self.fullscreen = False

        # Everything that must reach the screen from any thread goes through this queue.
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self._predicting = False
        self._closing = False
        self._dead = False
        self._live = False
        self.history: list[str] = []
        self.hist_idx = 0
        self.last_dir = os.getcwd()

        # Filled in by the boot worker once CIP.py has been imported.
        self.cip = None
        self.model = None
        self.threshold = 0.5

        self._build_fonts()
        self._build_style()
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", lambda: self._quit())
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Control-o>", lambda e: self._open_dialog())
        self.root.bind("<Control-l>", lambda e: self._clear_screen())
        self.root.bind("<Control-equal>", lambda e: self._zoom(+1))
        self.root.bind("<Control-plus>", lambda e: self._zoom(+1))
        self.root.bind("<Control-minus>", lambda e: self._zoom(-1))

        self._set_busy(True)                     # locked until boot has finished
        self.root.after(25, self._pump)
        self.root.after(80, self._boot)

    # ══ construction ══════════════════════════════════════════════════════════

    def _build_fonts(self):
        families = set(tkfont.families(self.root))
        name = next((f for f in MONO_CANDIDATES if f in families), None)
        base = tkfont.nametofont("TkFixedFont")
        fam = name or base.actual("family")
        self.font_size = 11
        self.font      = tkfont.Font(family=fam, size=self.font_size)
        self.font_bold = tkfont.Font(family=fam, size=self.font_size, weight="bold")
        self.font_ui   = tkfont.Font(family=fam, size=10)
        self.font_ui_b = tkfont.Font(family=fam, size=10, weight="bold")

    def _build_style(self):
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure("Tool.TButton", background=BG_BTN, foreground=FG, borderwidth=0,
                     focusthickness=0, padding=(10, 4), font=self.font_ui)
        st.map("Tool.TButton",
               background=[("disabled", BG_PANEL), ("active", BG_BTN_HI)],
               foreground=[("disabled", FG_DIM)])
        st.configure("Tool.TCheckbutton", background=BG_PANEL, foreground=FG, font=self.font_ui,
                     focusthickness=0, indicatorcolor=BG_BTN, indicatormargin=(0, 0, 4, 0))
        st.map("Tool.TCheckbutton",
               background=[("active", BG_PANEL)],
               foreground=[("disabled", FG_DIM)],
               indicatorcolor=[("selected", GREEN), ("!selected", BG_BTN)])
        st.configure("Vertical.TScrollbar", background=BG_BTN, troughcolor=BG, borderwidth=0,
                     arrowcolor=FG_DIM)

    def _build_ui(self):
        root = self.root

        # ── top bar ───────────────────────────────────────────────────────
        bar = tk.Frame(root, bg=BG_PANEL)
        bar.pack(side="top", fill="x")
        self.cmd_buttons: list = []

        def add_button(text, command):
            b = ttk.Button(bar, text=text, style="Tool.TButton", command=command,
                           takefocus=False, width=0)
            b.pack(side="left", padx=(6, 0), pady=6)
            self.cmd_buttons.append(b)
            return b

        def add_sep():
            tk.Frame(bar, bg=BG_BTN_HI, width=1).pack(side="left", fill="y", padx=(10, 4), pady=8)

        add_button("Open FASTA…", self._open_dialog)
        add_button("Outputs", self._open_outputs)
        add_button("Clear", self._clear_screen)
        add_sep()
        self.var_bed  = tk.BooleanVar(value=False)
        self.var_gff3 = tk.BooleanVar(value=False)
        for text, var in (("BED", self.var_bed), ("GFF3", self.var_gff3)):
            c = ttk.Checkbutton(bar, text=text, variable=var, style="Tool.TCheckbutton", takefocus=False)
            c.pack(side="left", padx=(8, 0), pady=6)
            self.cmd_buttons.append(c)
        add_sep()
        add_button("Help", lambda: self._run_line("/help"))

        # ── bottom input line ─────────────────────────────────────────────
        inp = tk.Frame(root, bg=BG_INPUT)
        inp.pack(side="bottom", fill="x")
        tk.Label(inp, text="FASTA ›", bg=BG_INPUT, fg=GREEN, font=self.font_bold).pack(
            side="left", padx=(10, 6), pady=8)
        self.entry = tk.Entry(inp, bg=BG_INPUT, fg=FG, insertbackground=FG, relief="flat",
                              highlightthickness=0, font=self.font,
                              disabledbackground=BG_INPUT, disabledforeground=FG_DIM)
        self.entry.pack(side="left", fill="x", expand=True, pady=8, padx=(0, 10))
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<KP_Enter>", self._submit)
        self.entry.bind("<Up>",   lambda e: self._history(-1))
        self.entry.bind("<Down>", lambda e: self._history(+1))

        # ── side panel ────────────────────────────────────────────────────
        side = tk.Frame(root, bg=BG_PANEL, width=PANEL_W)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self._build_panel(side)

        # ── console ───────────────────────────────────────────────────────
        mid = tk.Frame(root, bg=BG)
        mid.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical")
        sb.pack(side="right", fill="y")
        self.text = tk.Text(mid, bg=BG, fg=FG, font=self.font, wrap="word", relief="flat",
                            highlightthickness=0, padx=12, pady=8, state="disabled",
                            insertbackground=BG, yscrollcommand=sb.set, cursor="arrow",
                            selectbackground="#3a4a5c", spacing1=1, spacing3=1)
        self.text.pack(side="left", fill="both", expand=True)
        sb.config(command=self.text.yview)
        self._configure_tags()
        self.text.bind("<Key>", self._text_key)
        self.text.bind("<Button-1>", lambda e: self.text.focus_set())

    def _configure_tags(self):
        t = self.text
        t.tag_configure("out",   foreground=FG)
        t.tag_configure("info",  foreground=BLUE)
        t.tag_configure("dim",   foreground=FG_DIM)
        t.tag_configure("warn",  foreground=YELLOW)
        t.tag_configure("error", foreground=RED)
        t.tag_configure("yes",   foreground=GREEN)     # CpG island
        t.tag_configure("no",    foreground=RED)       # not a CpG island
        t.tag_configure("you",   foreground=GREEN)
        t.tag_configure("title", foreground=CYAN, font=self.font_bold)
        self._configure_help_tag()

    def _configure_help_tag(self):
        x = self.font.measure("0") * 26
        self.text.tag_configure("help", foreground=FG, tabs=(x,), lmargin1=0, lmargin2=x)

    def _build_panel(self, side):
        def header(text):
            tk.Label(side, text=text, bg=BG_PANEL, fg=FG_DIM, font=self.font_ui_b,
                     anchor="w").pack(fill="x", padx=14, pady=(14, 4))

        def grid_of(keys) -> dict:
            grid = tk.Frame(side, bg=BG_PANEL)
            grid.pack(fill="x", padx=14)
            grid.columnconfigure(1, weight=1)
            vars_ = {}
            for r, key in enumerate(keys):
                tk.Label(grid, text=key, bg=BG_PANEL, fg=FG_DIM, font=self.font_ui,
                         anchor="nw").grid(row=r, column=0, sticky="nw", pady=1)
                var = tk.StringVar(value="—")
                vars_[key] = var
                tk.Label(grid, textvariable=var, bg=BG_PANEL, fg=FG, font=self.font_ui, anchor="nw",
                         justify="left", wraplength=PANEL_W - 120).grid(
                             row=r, column=1, sticky="nw", padx=(10, 0), pady=1)
            return vars_

        header("MODEL")
        self.status_vars = grid_of(("Version", "State", "Architecture", "Trained on",
                                    "Tested on", "Threshold", "Features"))
        header("LAST RUN")
        self.run_vars = grid_of(("File", "Sequences", "CpG islands", "Not islands", "Skipped", "Time"))
        self.run_canvas = tk.Canvas(side, width=PANEL_W - 28, height=14, bg="#383838",
                                    highlightthickness=0)
        self.run_canvas.pack(padx=14, pady=(8, 0), anchor="w")
        self.bar_yes = self.run_canvas.create_rectangle(0, 0, 0, 14, width=0, fill=GREEN)
        self.bar_no  = self.run_canvas.create_rectangle(0, 0, 0, 14, width=0, fill=RED)
        tk.Label(side, text="green: island · red: not an island", bg=BG_PANEL, fg=FG_DIM,
                 font=self.font_ui, anchor="w").pack(fill="x", padx=14, pady=(2, 0))

    # ══ thread-safe output ════════════════════════════════════════════════════

    def emit(self, text: str, tag: str = "out", end: str = "\n") -> None:
        """Safe to call from any thread."""
        self.q.put(("out", text, tag, end))

    def _pump(self):
        try:
            while True:
                item = self.q.get_nowait()
                try:
                    if item[0] == "out":
                        self._write(item[1], item[2], item[3])
                    elif item[0] == "call":
                        item[1]()
                except Exception:
                    if self.cip:
                        self.cip.log.error("GUI error: %s", traceback.format_exc())
        except queue.Empty:
            pass
        if not self._dead:
            self.root.after(25, self._pump)

    def _write(self, text: str, tag: str = "out", end: str = "\n") -> None:
        """``end="\\r"`` writes a *live* line that the next write replaces (progress bars)."""
        at_bottom = self.text.yview()[1] >= 0.999
        self.text.configure(state="normal")
        if self._live:
            self.text.delete("live", "end-1c")
            self._live = False
        if end == "\r":
            self.text.mark_set("live", "end-1c")
            self.text.mark_gravity("live", "left")
            self.text.insert("end", text, tag)
            self._live = True
        else:
            self.text.insert("end", text + end, tag)
        self.text.configure(state="disabled")
        if at_bottom:
            self.text.see("end")

    def _clear_screen(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self._live = False
        self.text.configure(state="disabled")

    # ══ boot (worker thread) ══════════════════════════════════════════════════

    def _boot(self):
        self.emit("Initializing CIP...", "info")
        threading.Thread(target=self._boot_work, daemon=True).start()

    def _boot_work(self):
        out, err = ConsoleStream.pair(self.emit)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                import CIP as cip                       # heavy: pandas, onnxruntime, biopython
                self.cip = cip
                self.q.put(("call", lambda: self.root.title(f"CIP {cip.VERSION}")))
                self.q.put(("call", lambda: self.status_vars["Version"].set(cip.VERSION)))
                self._boot_steps(cip)
        except ImportError as e:
            self.emit(f"[Error] Missing dependency: {e}", "error")
            self.emit("  Install the requirements with:  pip install -r requirements.txt", "error")
            self.q.put(("call", lambda: self._fatal(f"Missing dependency:\n\n{e}")))
        except Exception as e:
            self.emit(f"[Error] Could not start CIP: {e}", "error")
            self.emit(traceback.format_exc(), "dim")
            self.q.put(("call", lambda: self._fatal(f"Could not start CIP:\n\n{e}")))
        finally:
            out.finish()
            err.finish()

    def _boot_steps(self, cip):
        """Same sequence as the ``__main__`` block of CIP.py (output goes to the console)."""
        print(f"CpG Island Predictor (CIP) {cip.VERSION}")
        self.emit("Copyright: AGPL-3.0-or-later (see LICENSE file)", "dim")
        self.emit("See https://github.com/lorenzoorsini3/CpG-Island-Predictor for source code", "dim")

        metadata = cip._load_metadata()
        n_features = None
        status = {"Architecture": "unknown", "Trained on": "—", "Tested on": "—",
                  "Threshold": "0.5 (default)", "Features": "—"}
        if metadata is not None:
            arch          = metadata.get("architecture_version", "unknown")
            n_features    = metadata.get("n_features")
            train_species = metadata.get("training_species", [])
            test_species  = metadata.get("test_species", [])
            self.threshold = metadata.get("prediction_threshold", 0.5)
            print(f"    Model architecture : {arch}")
            if train_species:
                print(f"    Trained on         : {', '.join(train_species)}")
            if test_species:
                print(f"    Evaluated on       : {', '.join(test_species)}")
            cip.log.info("Architecture: %s, trained on: %s", arch, train_species)
            status.update({"Architecture": arch,
                           "Trained on": ", ".join(train_species) or "—",
                           "Tested on": ", ".join(test_species) or "—",
                           "Threshold": f"{self.threshold:.4f}",
                           "Features": str(n_features) if n_features is not None else "—"})
        else:
            cip.handle_warning("warning", "Metadata not found, proceeding without model info.", cip.log)

        if metadata is not None and cip._check_version_compatibility(metadata):
            self.q.put(("call", lambda: self._fatal(
                "This version of CIP is too old for the loaded model (see the console).")))
            return

        try:
            self.model = cip.rt.InferenceSession(cip._MODEL_FILE)
        except Exception as e:
            print(cip.handle_error(e, cip._MODEL_FILE))
            cip.log.error("Failed to load model '%s': %s", cip._MODEL_FILE, e)
            self.q.put(("call", lambda: self._fatal(f"Could not load the model:\n\n{e}")))
            return

        if n_features is not None and len(cip.FEATURES_ORDER) != n_features:
            cip.handle_warning(
                "warning",
                f"Metadata expects {n_features} features but this version of CIP would build "
                f"{len(cip.FEATURES_ORDER)}. Results may be unreliable.",
                cip.log,
            )

        self.emit("")
        self.emit("Open a FASTA file with the button above, or type its path "
                  "[--bed] [--gff3] in the input line. /help for usage.", "info")
        self.q.put(("call", lambda: self._boot_done(status)))

    def _boot_done(self, status: dict):
        for key, val in status.items():
            self.status_vars[key].set(val)
        self.status_vars["State"].set("ready")
        self._set_busy(False)

    def _fatal(self, message: str):
        self.status_vars["State"].set("not available")
        messagebox.showerror("CIP", message, parent=self.root)
        self._quit(confirm=False)

    # ══ running an analysis ═══════════════════════════════════════════════════

    def _submit(self, event=None):
        if self.busy:
            return "break"
        line = self.entry.get().strip()
        if not line:
            return "break"
        self.entry.delete(0, "end")
        if not self.history or self.history[-1] != line:
            self.history.append(line)
        self.hist_idx = len(self.history)
        self._run_line(line)
        return "break"

    def _run_line(self, line: str):
        """Echo and execute one line typed in the input line (or sent by a button)."""
        if self.busy or self.cip is None:
            return
        self.emit(f"› {line}", "you")

        if line == "/quit":
            self._quit()
            return
        if line == "/help":
            self.emit(HELP_TEXT, "help")
            return
        if line == "/clear":
            self.q.put(("call", self._clear_screen))
            return

        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        if not parts:
            return
        flags   = {p.lstrip("-").lower() for p in parts[1:]}
        formats = flags & self.cip._SUPPORTED_FORMATS
        unknown = flags - self.cip._SUPPORTED_FORMATS
        if unknown:
            msg = f"Unrecognised flags ignored: {', '.join('--' + f for f in sorted(unknown))}"
            self.emit(f"WARNING: {msg}", "warn")
            self.cip.log.warning(msg)
        self._start_run(os.path.expanduser(parts[0]), formats)

    def _open_dialog(self):
        if self.busy or self.cip is None:
            return
        path = filedialog.askopenfilename(title="Open FASTA", initialdir=self.last_dir,
                                          filetypes=FASTA_TYPES, parent=self.root)
        if not path:
            return
        self.last_dir = os.path.dirname(path)
        formats = {name for name, var in (("bed", self.var_bed), ("gff3", self.var_gff3)) if var.get()}
        shown = shlex.quote(path) + "".join(f" --{f}" for f in sorted(formats))
        self.emit(f"› {shown}", "you")
        self._start_run(path, formats)

    def _start_run(self, path: str, formats: set):
        # The switches in the top bar add to any flags typed in the line.
        if self.var_bed.get():
            formats = formats | {"bed"}
        if self.var_gff3.get():
            formats = formats | {"gff3"}
        self._predicting = True
        self._set_busy(True)
        self.status_vars["State"].set("running…")
        t0 = time.monotonic()

        def work():
            out, err = ConsoleStream.pair(self.emit)
            summary = None
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    summary = self.cip.predict_from_fasta(self.model, path, formats, self.threshold)
                except Exception as e:
                    self.cip.log.error("Run failed for '%s': %s\n%s", path, e, traceback.format_exc())
                    print(self.cip.handle_error(e, path))
            out.finish()
            err.finish()
            elapsed = time.monotonic() - t0
            self.q.put(("call", lambda: self._run_done(summary, elapsed)))

        threading.Thread(target=work, daemon=True).start()

    def _run_done(self, summary, elapsed: float):
        self._predicting = False
        if self._closing:
            return
        if summary:
            rv = self.run_vars
            rv["File"].set(os.path.basename(summary["fasta"]))
            rv["Sequences"].set(f"{summary['n_records']} in file, {summary['n_predicted']} analysed")
            rv["CpG islands"].set(str(summary["n_island"]))
            rv["Not islands"].set(str(summary["n_non_island"]))
            rv["Skipped"].set(str(summary["n_skipped"]))
            rv["Time"].set(f"{elapsed:.1f} s")
            total = max(summary["n_predicted"], 1)
            w = self.run_canvas.winfo_reqwidth()
            wy = int(w * summary["n_island"] / total)
            self.run_canvas.coords(self.bar_yes, 0, 0, wy, 14)
            self.run_canvas.coords(self.bar_no, wy, 0, w if summary["n_predicted"] else wy, 14)
        self.status_vars["State"].set("ready")
        self._set_busy(False)

    # ══ small actions ═════════════════════════════════════════════════════════

    def _open_outputs(self):
        if self.cip is None:
            return
        outs = self.cip._OUTS_DIR
        try:
            outs.mkdir(exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(outs)                                  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(outs)])
        except Exception as e:
            self.emit(f"[Error] Could not open {outs}: {e}", "error")

    def _history(self, step: int):
        if not self.history:
            return "break"
        self.hist_idx = max(0, min(len(self.history), self.hist_idx + step))
        self.entry.delete(0, "end")
        if self.hist_idx < len(self.history):
            self.entry.insert(0, self.history[self.hist_idx])
        return "break"

    def _text_key(self, event):
        """Printable keys typed in the console are forwarded to the input line."""
        if event.state & 0x4 or not event.char or not event.char.isprintable():
            return None
        if str(self.entry["state"]) == "normal":
            self.entry.focus_set()
            self.entry.insert("end", event.char)
        return "break"

    def _set_busy(self, busy: bool):
        self.busy = busy
        self.entry.configure(state="disabled" if busy else "normal")
        for w in self.cmd_buttons:
            w.state(["disabled"] if busy else ["!disabled"])
        if not busy:
            self.entry.focus_set()

    def _toggle_fullscreen(self, event=None):
        """F11: real full screen (no title bar) on/off."""
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _zoom(self, step: int):
        self.font_size = max(8, min(24, self.font_size + step))
        self.font.configure(size=self.font_size)
        self.font_bold.configure(size=self.font_size)
        self._configure_help_tag()

    # ══ shutdown ══════════════════════════════════════════════════════════════

    def _quit(self, confirm: bool = True):
        if self._closing:
            return
        if confirm and self._predicting:
            if not messagebox.askyesno("CIP", "An analysis is still running; its output files may be "
                                       "incomplete if you quit now.\n\nQuit anyway?",
                                       default="no", parent=self.root):
                return
        self._closing = True
        self._set_busy(True)
        if self.cip is None:                       # start-up failed before CIP.py was imported
            self._finish_quit()
            return

        def work():
            out, err = ConsoleStream.pair(self.emit)
            self.emit("")
            self.emit("Session ended.", "dim")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.cip.log.info("Session ended.")
                try:
                    self.cip.wait_for_archiver()   # prints a message only if log archiving is pending
                except Exception as e:
                    self.cip.handle_warning("error", f"Compression error: {e}", self.cip.log)
            out.finish()
            err.finish()
            time.sleep(1.0)
            self.q.put(("call", self._finish_quit))

        threading.Thread(target=work, daemon=True).start()

    def _finish_quit(self):
        self._dead = True
        self.root.destroy()

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: self._quit(confirm=False))
        self.root.mainloop()


def main():
    try:
        app = CipApp()
    except tk.TclError as e:
        print(f"Cannot open a window ({e}). Is a graphical desktop session available?", file=sys.stderr)
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
