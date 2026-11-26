"""Trajectory Analysis dialog: pick a trajectory, choose an analysis, plot and export."""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import analysis, formats
from .plot import make_plot_widget
from .wheel_guard import install_wheel_guard
from .worker import JobThread

_log = logging.getLogger(__name__)
TITLE = "Trajectory Analysis"
MAX_TABLE_ROWS = 5000


def _dspin(lo, hi, decimals, step, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    if suffix:
        w.setSuffix(suffix)
    return w


def _ispin(lo, hi):
    w = QSpinBox()
    w.setRange(lo, hi)
    return w


def describe(topology, frames):
    """One-paragraph description of a loaded trajectory."""
    parts = [f"{topology.n_atoms} atoms, {frames.n_frames} frame(s)"]
    box = frames.default_box
    if box is not None:
        import numpy as np

        lengths = np.linalg.norm(box, axis=1)
        parts.append(f"cell {lengths[0]:.3f} x {lengths[1]:.3f} x {lengths[2]:.3f} A")
    else:
        parts.append("no cell")
    if topology.has_residues:
        parts.append(f"{len(set(topology.resnames))} residue name(s)")
    counts = {}
    for s in topology.symbols:
        counts[s] = counts.get(s, 0) + 1
    formula = " ".join(
        f"{s}{n}" for s, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    if len(formula) > 80:
        formula = formula[:77] + "..."
    parts.append(formula)
    if frames.energy(0) is not None:
        parts.append("per-frame energies found")
    if frames.time(0) is not None:
        parts.append("frame times found")
    return "; ".join(parts)



def _scrolled_column(page):
    """The settings column in a vertical scroll area, so it never makes the dialog taller
    than the screen; Run and the progress bar stay below it, always in view."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(page)
    area.setMinimumWidth(
        page.minimumSizeHint().width() + area.verticalScrollBar().sizeHint().width()
    )
    # Scrolling over a field must not change it: fields take the wheel once clicked.
    install_wheel_guard(page)
    return area


def fit_to_screen(dialog, fraction=0.9):
    """Shrink the dialog's first size to the screen it opens on (small laptop screens)."""
    parent = dialog.parentWidget()
    screen = parent.screen() if parent is not None else dialog.screen()
    if screen is None:
        return
    area = screen.availableGeometry()
    dialog.resize(
        min(dialog.width(), int(area.width() * fraction)),
        min(dialog.height(), int(area.height() * fraction)),
    )


class TrajectoryAnalysisDialog(QDialog):
    def __init__(
        self,
        parent=None,
        persistent_settings=None,
        get_molecule=None,
        mark_modified=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.resize(1100, 760)
        fit_to_screen(self)
        self.persistent_settings = (
            persistent_settings if persistent_settings is not None else {}
        )
        self.get_molecule = get_molecule
        self.mark_modified = mark_modified
        self._updating = False
        self.topology = None
        self.frames = None
        self.source_label = ""
        self.result = None
        self.thread = None

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        root.addLayout(top)

        # ---------------------------------------------------------- left column
        column = QVBoxLayout()
        top.addLayout(column, 0)
        settings_page = QWidget()
        left = QVBoxLayout(settings_page)
        left.setContentsMargins(0, 0, 0, 0)

        src = QGroupBox("Trajectory")
        sl = QGridLayout(src)
        self.file_radio = QRadioButton("File:")
        self.mol_radio = QRadioButton("Current molecule (each conformer is a frame)")
        self.file_radio.setChecked(True)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Multi-frame PDB, GRO, XYZ or SDF")
        self.browse_button = QPushButton("Browse...")
        self.load_button = QPushButton("Load")
        sl.addWidget(self.file_radio, 0, 0)
        sl.addWidget(self.path_edit, 0, 1)
        sl.addWidget(self.browse_button, 0, 2)
        sl.addWidget(self.mol_radio, 1, 0, 1, 2)
        sl.addWidget(self.load_button, 1, 2)
        self.info_label = QLabel("Nothing loaded.")
        self.info_label.setWordWrap(True)
        sl.addWidget(self.info_label, 2, 0, 1, 3)
        left.addWidget(src)

        rng = QGroupBox("Frames and time")
        fl = QFormLayout(rng)
        self.first_spin = _ispin(1, 10**9)
        self.last_spin = _ispin(0, 10**9)
        self.last_spin.setSpecialValueText("last")
        self.stride_spin = _ispin(1, 10**6)
        self.dt_spin = _dspin(0.0, 1e9, 4, 1.0, " ps")
        self.dt_spin.setSpecialValueText("from file / frame number")
        fl.addRow("First frame:", self.first_spin)
        fl.addRow("Last frame:", self.last_spin)
        fl.addRow("Stride:", self.stride_spin)
        fl.addRow("Time per file frame:", self.dt_spin)
        left.addWidget(rng)

        cell = QGroupBox("Periodic cell")
        cl = QGridLayout(cell)
        self.cell_combo = QComboBox()
        self.cell_combo.addItem("From the file / molecule (moleditpy_cell)", "file")
        self.cell_combo.addItem("Custom cell", "custom")
        self.cell_combo.addItem("None (no periodicity)", "none")
        cl.addWidget(self.cell_combo, 0, 0, 1, 6)
        self.cell_spins = []
        for k, label in enumerate(("a", "b", "c", "alpha", "beta", "gamma")):
            spin = (
                _dspin(0.0, 1e5, 3, 1.0, " A")
                if k < 3
                else _dspin(1.0, 179.0, 2, 1.0, " deg")
            )
            self.cell_spins.append(spin)
            cl.addWidget(QLabel(label), 1 + k // 3, 2 * (k % 3))
            cl.addWidget(spin, 1 + k // 3, 2 * (k % 3) + 1)
        left.addWidget(cell)

        ana = QGroupBox("Analysis")
        al = QVBoxLayout(ana)
        self.analysis_combo = QComboBox()
        for key, label in analysis.ANALYSES:
            self.analysis_combo.addItem(label, key)
        al.addWidget(self.analysis_combo)
        self.pages = QStackedWidget()
        al.addWidget(self.pages)
        self._build_pages()
        hint = QLabel(
            "Selections: all, O, element O H, resname SOL, name OW, resid 1-10, "
            "index 0-99, water, protein, heavy, hydrogen; combine with and / or / not "
            "and parentheses."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        al.addWidget(hint)
        left.addWidget(ana)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        left.addStretch(1)
        column.addWidget(_scrolled_column(settings_page), 1)
        column.addLayout(run_row)
        column.addWidget(self.progress)

        # --------------------------------------------------------- right column
        splitter = QSplitter(Qt.Orientation.Vertical)
        top.addWidget(splitter, 1)
        self.plot, self.plot_backend = make_plot_widget(self)
        splitter.addWidget(self.plot)
        lower = QWidget()
        ll = QVBoxLayout(lower)
        ll.setContentsMargins(0, 0, 0, 0)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(110)
        self.summary.setFont(QFont("Courier New", 9))
        ll.addWidget(self.summary)
        self.table = QTableWidget(0, 0)
        ll.addWidget(self.table, 1)
        export = QHBoxLayout()
        self.csv_button = QPushButton("Export CSV...")
        self.png_button = QPushButton("Export PNG...")
        self.close_button = QPushButton("Close")
        for b in (self.csv_button, self.png_button):
            b.setEnabled(False)
            export.addWidget(b)
        export.addStretch(1)
        backend = QLabel(
            "Plot: matplotlib"
            if self.plot_backend == "matplotlib"
            else "Plot: built-in"
        )
        backend.setStyleSheet("color: gray;")
        export.addWidget(backend)
        export.addWidget(self.close_button)
        ll.addLayout(export)
        splitter.addWidget(lower)
        splitter.setSizes([480, 280])

        # ---------------------------------------------------------------- wiring
        self.browse_button.clicked.connect(self.browse)
        self.load_button.clicked.connect(self.load_source)
        self.path_edit.returnPressed.connect(self.load_source)
        self.analysis_combo.currentIndexChanged.connect(self.pages.setCurrentIndex)
        self.cell_combo.currentIndexChanged.connect(self._cell_mode_changed)
        self.run_button.clicked.connect(self.run_analysis)
        self.cancel_button.clicked.connect(self.cancel_job)
        self.csv_button.clicked.connect(self.export_csv)
        self.png_button.clicked.connect(self.export_png)
        self.close_button.clicked.connect(self.close)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.run_analysis)
        for w in self._setting_widgets():
            for signal in (
                "valueChanged",
                "textChanged",
                "currentIndexChanged",
                "toggled",
            ):
                sig = getattr(w, signal, None)
                if sig is not None:
                    sig.connect(self._settings_changed)
                    break

        self.apply_settings(self.persistent_settings)

    # ------------------------------------------------------------ parameter pages

    def _build_pages(self):
        # RDF
        page = QWidget()
        f = QFormLayout(page)
        self.rdf_a = QLineEdit()
        self.rdf_b = QLineEdit()
        self.rdf_rmax = _dspin(0.1, 1000.0, 2, 0.5, " A")
        self.rdf_bins = _ispin(1, 100000)
        f.addRow("Selection A:", self.rdf_a)
        f.addRow("Selection B:", self.rdf_b)
        f.addRow("r max:", self.rdf_rmax)
        f.addRow("Bins:", self.rdf_bins)
        self.pages.addWidget(page)
        # MSD
        page = QWidget()
        f = QFormLayout(page)
        self.msd_sel = QLineEdit()
        self.msd_dims = QComboBox()
        for d in ("xyz", "xy", "xz", "yz", "x", "y", "z"):
            self.msd_dims.addItem(d, d)
        self.msd_com = QCheckBox("Remove centre-of-mass drift of the selection")
        self.msd_fit_start = _dspin(0.0, 1.0, 2, 0.05)
        self.msd_fit_end = _dspin(0.0, 1.0, 2, 0.05)
        f.addRow("Selection:", self.msd_sel)
        f.addRow("Dimensions:", self.msd_dims)
        f.addRow(self.msd_com)
        f.addRow("Fit from (fraction of lags):", self.msd_fit_start)
        f.addRow("Fit to (fraction of lags):", self.msd_fit_end)
        self.pages.addWidget(page)
        # RMSD and RMSF share widgets on two pages
        self.rmsd_widgets = {}
        for key in ("rmsd", "rmsf"):
            page = QWidget()
            f = QFormLayout(page)
            sel = QLineEdit()
            fit = QLineEdit()
            fit.setPlaceholderText("same as the selection")
            ref = _ispin(1, 10**9)
            align = QCheckBox("Align each frame on the fit selection (Kabsch)")
            unwrap = QCheckBox("Follow atoms through the cell between frames")
            f.addRow("Selection:", sel)
            f.addRow("Fit selection:", fit)
            f.addRow("Reference frame:", ref)
            f.addRow(align)
            f.addRow(unwrap)
            self.rmsd_widgets[key] = (sel, fit, ref, align, unwrap)
            self.pages.addWidget(page)
        # Rg
        page = QWidget()
        f = QFormLayout(page)
        self.rg_sel = QLineEdit()
        self.rg_mass = QCheckBox("Mass-weighted")
        self.rg_unwrap = QCheckBox("Follow atoms through the cell between frames")
        f.addRow("Selection:", self.rg_sel)
        f.addRow(self.rg_mass)
        f.addRow(self.rg_unwrap)
        self.pages.addWidget(page)
        # density
        page = QWidget()
        f = QFormLayout(page)
        self.dens_sel = QLineEdit()
        self.dens_axis = QComboBox()
        for label, key in (
            ("a (cell) / x", "a"),
            ("b (cell) / y", "b"),
            ("c (cell) / z", "c"),
        ):
            self.dens_axis.addItem(label, key)
        self.dens_bins = _ispin(1, 100000)
        self.dens_kind = QComboBox()
        self.dens_kind.addItem("Mass density", "mass")
        self.dens_kind.addItem("Number density", "number")
        f.addRow("Selection:", self.dens_sel)
        f.addRow("Axis:", self.dens_axis)
        f.addRow("Bins:", self.dens_bins)
        f.addRow("Quantity:", self.dens_kind)
        self.pages.addWidget(page)
        # energy
        page = QWidget()
        f = QFormLayout(page)
        self.energy_relative = QCheckBox("Relative to the lowest frame")
        f.addRow(self.energy_relative)
        note = QLabel(
            "Reads energies written in the file: extended-XYZ energy=..., ORCA "
            "'E -76.3', xtb/CREST comment lines, PDB REMARKs, SDF energy properties."
        )
        note.setWordWrap(True)
        f.addRow(note)
        self.pages.addWidget(page)

    def _setting_widgets(self):
        widgets = [
            self.path_edit,
            self.file_radio,
            self.first_spin,
            self.last_spin,
            self.stride_spin,
            self.dt_spin,
            self.cell_combo,
            *self.cell_spins,
            self.analysis_combo,
            self.rdf_a,
            self.rdf_b,
            self.rdf_rmax,
            self.rdf_bins,
            self.msd_sel,
            self.msd_dims,
            self.msd_com,
            self.msd_fit_start,
            self.msd_fit_end,
            self.rg_sel,
            self.rg_mass,
            self.rg_unwrap,
            self.dens_sel,
            self.dens_axis,
            self.dens_bins,
            self.dens_kind,
            self.energy_relative,
        ]
        for group in self.rmsd_widgets.values():
            widgets.extend(group)
        return widgets

    # ------------------------------------------------------------------ settings

    @staticmethod
    def _set_combo(combo, data):
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)

    def apply_settings(self, settings):
        p = analysis.merged_params(settings or {})
        settings = settings or {}
        self._updating = True
        try:
            self.path_edit.setText(str(settings.get("path", "")))
            if settings.get("source", "file") == "molecule":
                self.mol_radio.setChecked(True)
            else:
                self.file_radio.setChecked(True)
            self.first_spin.setValue(int(p["first"]))
            self.last_spin.setValue(int(p["last"]))
            self.stride_spin.setValue(int(p["stride"]))
            self.dt_spin.setValue(float(p["dt"]))
            self._set_combo(self.cell_combo, p["cell_mode"])
            for spin, value in zip(self.cell_spins, p["custom_cell"]):
                spin.setValue(float(value))
            self._set_combo(self.analysis_combo, settings.get("analysis", "rdf"))
            self.pages.setCurrentIndex(self.analysis_combo.currentIndex())
            self.rdf_a.setText(p["rdf_sel_a"])
            self.rdf_b.setText(p["rdf_sel_b"])
            self.rdf_rmax.setValue(float(p["rdf_rmax"]))
            self.rdf_bins.setValue(int(p["rdf_bins"]))
            self.msd_sel.setText(p["msd_sel"])
            self._set_combo(self.msd_dims, p["msd_dims"])
            self.msd_com.setChecked(bool(p["msd_remove_com"]))
            self.msd_fit_start.setValue(float(p["msd_fit_start"]))
            self.msd_fit_end.setValue(float(p["msd_fit_end"]))
            for sel, fit, ref, align, unwrap in self.rmsd_widgets.values():
                sel.setText(p["rmsd_sel"])
                fit.setText(p["rmsd_fit_sel"])
                ref.setValue(int(p["rmsd_ref"]))
                align.setChecked(bool(p["rmsd_align"]))
                unwrap.setChecked(bool(p["unwrap"]))
            self.rg_sel.setText(p["rg_sel"])
            self.rg_mass.setChecked(bool(p["rg_mass_weighted"]))
            self.rg_unwrap.setChecked(bool(p["unwrap"]))
            self.dens_sel.setText(p["dens_sel"])
            self._set_combo(self.dens_axis, p["dens_axis"])
            self.dens_bins.setValue(int(p["dens_bins"]))
            self._set_combo(self.dens_kind, p["dens_kind"])
            self.energy_relative.setChecked(bool(p["energy_relative"]))
        finally:
            self._updating = False
        self._cell_mode_changed()

    def current_kind(self):
        return self.analysis_combo.currentData() or "rdf"

    def read_settings(self):
        kind = self.current_kind()
        rsel, rfit, rref, ralign, runwrap = self.rmsd_widgets[
            "rmsf" if kind == "rmsf" else "rmsd"
        ]
        unwrap = self.rg_unwrap.isChecked() if kind == "rg" else runwrap.isChecked()
        return {
            "source": "file" if self.file_radio.isChecked() else "molecule",
            "path": self.path_edit.text().strip(),
            "analysis": kind,
            "first": self.first_spin.value(),
            "last": self.last_spin.value(),
            "stride": self.stride_spin.value(),
            "dt": self.dt_spin.value(),
            "cell_mode": self.cell_combo.currentData(),
            "custom_cell": [s.value() for s in self.cell_spins],
            "rdf_sel_a": self.rdf_a.text().strip() or "all",
            "rdf_sel_b": self.rdf_b.text().strip() or "all",
            "rdf_rmax": self.rdf_rmax.value(),
            "rdf_bins": self.rdf_bins.value(),
            "msd_sel": self.msd_sel.text().strip() or "all",
            "msd_dims": self.msd_dims.currentData(),
            "msd_remove_com": self.msd_com.isChecked(),
            "msd_fit_start": self.msd_fit_start.value(),
            "msd_fit_end": self.msd_fit_end.value(),
            "rmsd_sel": rsel.text().strip() or "all",
            "rmsd_fit_sel": rfit.text().strip(),
            "rmsd_ref": rref.value(),
            "rmsd_align": ralign.isChecked(),
            "unwrap": unwrap,
            "rg_sel": self.rg_sel.text().strip() or "all",
            "rg_mass_weighted": self.rg_mass.isChecked(),
            "dens_sel": self.dens_sel.text().strip() or "all",
            "dens_axis": self.dens_axis.currentData(),
            "dens_bins": self.dens_bins.value(),
            "dens_kind": self.dens_kind.currentData(),
            "energy_relative": self.energy_relative.isChecked(),
        }

    def _settings_changed(self, *_args):
        if self._updating:
            return
        self.persistent_settings.update(self.read_settings())
        if self.mark_modified is not None:
            try:
                self.mark_modified()
            except Exception as exc:  # noqa: BLE001 - host API guard
                _log.debug("mark_modified failed: %s", exc)

    def _cell_mode_changed(self, *_args):
        custom = self.cell_combo.currentData() == "custom"
        for spin in self.cell_spins:
            spin.setEnabled(custom)

    # -------------------------------------------------------------------- loading

    def browse(self):
        start = os.path.dirname(self.path_edit.text()) or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open trajectory", start, formats.FILE_FILTER
        )
        if path:
            self.path_edit.setText(path)
            self.file_radio.setChecked(True)
            self.load_source()

    def _close_frames(self):
        if self.frames is not None:
            try:
                self.frames.close()
            except Exception as exc:  # noqa: BLE001 - best effort
                _log.debug("closing the trajectory failed: %s", exc)
        self.frames = None
        self.topology = None

    def load_source(self):
        if self.thread is not None:
            return
        if self.mol_radio.isChecked():
            mol = self.get_molecule() if self.get_molecule is not None else None
            try:
                topology, frames = formats.from_rdkit(mol)
            except (formats.FormatError, AttributeError) as exc:
                QMessageBox.warning(
                    self, TITLE, f"Cannot use the current molecule: {exc}"
                )
                return
            self._loaded((topology, frames, "Current molecule"))
            return
        path = self.path_edit.text().strip()
        if not path:
            self.browse()
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, TITLE, f"File not found:\n{path}")
            return

        def job(_progress, _cancelled):
            topology, frames = formats.open_trajectory(path)
            return topology, frames, os.path.basename(path)

        self.info_label.setText(f"Indexing {os.path.basename(path)}...")
        self._start(job, self._loaded)

    def _loaded(self, payload):
        self._close_frames()
        self.topology, self.frames, self.source_label = payload
        self.info_label.setText(
            f"{self.source_label}: {describe(self.topology, self.frames)}"
        )
        self.last_spin.setMaximum(max(0, self.frames.n_frames))
        self._settings_changed()

    # ------------------------------------------------------------------- running

    def _start(self, job, on_success):
        self.thread = JobThread(job, self)
        self.thread.progress.connect(self._on_progress)
        self.thread.succeeded.connect(on_success)
        self.thread.failed.connect(self._on_failed)
        self.thread.was_cancelled.connect(self._on_cancelled)
        self.thread.finished.connect(self._on_finished)
        self.progress.setRange(0, 0)
        self._set_busy(True)
        self.thread.start()

    def _set_busy(self, busy):
        for w in (self.run_button, self.load_button, self.browse_button):
            w.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _on_progress(self, done, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _on_failed(self, message):
        if self.topology is None:
            self.info_label.setText("Nothing loaded.")
        QMessageBox.warning(self, TITLE, message)

    def _on_cancelled(self):
        self.summary.setPlainText("Cancelled.")

    def _on_finished(self):
        thread = self.thread
        self.thread = None
        if thread is not None:
            thread.deleteLater()
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if self.result is not None else 0)
        self._set_busy(False)

    def cancel_job(self):
        if self.thread is not None:
            self.thread.cancel()

    def run_analysis(self):
        if self.thread is not None:
            return
        if self.frames is None:
            self.load_source()
            if self.frames is None:
                if self.thread is not None:
                    self.summary.setPlainText(
                        "Loading; press Run again when it is done."
                    )
                return
        params = self.read_settings()
        kind = params["analysis"]
        topology, frames = self.topology, self.frames

        def job(progress, cancelled):
            return analysis.run(kind, topology, frames, params, progress, cancelled)

        self.summary.setPlainText("Running...")
        self._start(job, self.show_result)

    def show_result(self, result):
        self.result = result
        self.plot.set_result(result)
        text = [f"{result.title} - {self.source_label}", result.summary] + result.notes
        self.summary.setPlainText("\n".join(t for t in text if t))
        rows = list(result.rows)
        shown = rows[:MAX_TABLE_ROWS]
        self.table.clear()
        self.table.setColumnCount(len(result.columns))
        self.table.setHorizontalHeaderLabels(result.columns)
        self.table.setRowCount(len(shown))
        for r, row in enumerate(shown):
            for c, value in enumerate(row):
                if isinstance(value, str):
                    text_value = value
                else:
                    try:
                        text_value = f"{float(value):.6g}"
                    except (TypeError, ValueError):
                        text_value = str(value)
                self.table.setItem(r, c, QTableWidgetItem(text_value))
        if len(rows) > MAX_TABLE_ROWS:
            self.summary.appendPlainText(
                f"Table shows the first {MAX_TABLE_ROWS} of {len(rows)} rows; "
                "CSV export contains all of them."
            )
        self.csv_button.setEnabled(True)
        self.png_button.setEnabled(True)

    # -------------------------------------------------------------------- export

    def _default_name(self, ext):
        base = os.path.splitext(self.source_label or "trajectory")[0].replace(" ", "_")
        return f"{base}_{self.current_kind()}.{ext}"

    def export_csv(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", self._default_name("csv"), "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(self.result.csv_text())
        except OSError as exc:
            QMessageBox.critical(self, TITLE, f"Could not write the file:\n{exc}")

    def export_png(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", self._default_name("png"), "PNG images (*.png)"
        )
        if not path:
            return
        try:
            self.plot.save_png(path)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QMessageBox.critical(self, TITLE, f"Could not write the image:\n{exc}")

    # ---------------------------------------------------------------- lifecycle

    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.cancel()
            self.thread.wait(5000)
        self._close_frames()
        super().closeEvent(event)
