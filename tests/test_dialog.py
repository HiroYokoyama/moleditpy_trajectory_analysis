"""The dialog and plot widgets with real PySide6 on the offscreen platform."""

import time
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PySide6.QtWidgets")

from trajectory_analysis import plot as plot_mod
from trajectory_analysis.analysis import Result, Series

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _wait(app, dlg, timeout=30.0):
    end = time.time() + timeout
    while dlg.thread is not None and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert dlg.thread is None, "the background job did not finish"


@pytest.fixture
def dialog(qapp, monkeypatch):
    from trajectory_analysis import main_dialog

    warnings = []
    monkeypatch.setattr(
        main_dialog.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2])
    )
    marks = []
    settings = {}
    dlg = main_dialog.TrajectoryAnalysisDialog(
        persistent_settings=settings,
        get_molecule=lambda: None,
        mark_modified=lambda: marks.append(1),
    )
    dlg.warnings = warnings
    dlg.marks = marks
    dlg.settings_store = settings
    yield dlg
    dlg.close()


def test_load_run_every_analysis(qapp, dialog):
    dialog.path_edit.setText(str(EXAMPLES / "water_box.gro"))
    dialog.load_source()
    _wait(qapp, dialog)
    assert dialog.frames is not None
    assert "192 atoms, 40 frame(s)" in dialog.info_label.text()
    dialog.rdf_a.setText("name OW")
    dialog.rdf_b.setText("name OW")
    dialog.rdf_rmax.setValue(6.0)
    dialog.msd_sel.setText("name OW")
    for _sel, *_rest in dialog.rmsd_widgets.values():
        _sel.setText("resid 1-4")
    for k in range(dialog.analysis_combo.count()):
        dialog.analysis_combo.setCurrentIndex(k)
        dialog.result = None
        dialog.run_analysis()
        _wait(qapp, dialog)
        kind = dialog.current_kind()
        if kind == "energy":
            assert dialog.warnings and "no per-frame energies" in dialog.warnings[-1]
            continue
        assert dialog.result is not None, kind
        assert dialog.table.rowCount() > 0
        assert dialog.csv_button.isEnabled()
    assert dialog.settings_store["analysis"] == "energy"
    assert dialog.settings_store["rdf_sel_a"] == "name OW"
    assert dialog.marks


def test_energy_and_exports(qapp, dialog, tmp_path, monkeypatch):
    from trajectory_analysis import main_dialog

    dialog.path_edit.setText(str(EXAMPLES / "ethanol_conf.xyz"))
    dialog.load_source()
    _wait(qapp, dialog)
    dialog.analysis_combo.setCurrentIndex(dialog.analysis_combo.findData("energy"))
    dialog.run_analysis()
    _wait(qapp, dialog)
    assert dialog.result is not None
    assert "10 of 10 frames" in dialog.summary.toPlainText()

    csv_path = tmp_path / "e.csv"
    png_path = tmp_path / "e.png"
    monkeypatch.setattr(
        main_dialog.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(csv_path), ""),
    )
    dialog.export_csv()
    assert (
        "frame,time_(ps,energy" in csv_path.read_text()
        or "frame," in csv_path.read_text()
    )
    monkeypatch.setattr(
        main_dialog.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(png_path), ""),
    )
    dialog.export_png()
    assert png_path.stat().st_size > 1000


def test_settings_round_trip(qapp, dialog):
    s = dialog.read_settings()
    s.update({"analysis": "density", "dens_axis": "b", "rdf_bins": 77, "dt": 2.5})
    dialog.apply_settings(s)
    out = dialog.read_settings()
    assert out["analysis"] == "density"
    assert out["dens_axis"] == "b"
    assert out["rdf_bins"] == 77
    assert out["dt"] == 2.5
    assert dialog.pages.currentIndex() == dialog.analysis_combo.currentIndex()


def test_bad_selection_is_reported(qapp, dialog):
    dialog.path_edit.setText(str(EXAMPLES / "water_box.gro"))
    dialog.load_source()
    _wait(qapp, dialog)
    dialog.rdf_a.setText("element Fe")
    dialog.analysis_combo.setCurrentIndex(0)
    dialog.run_analysis()
    _wait(qapp, dialog)
    assert dialog.result is None
    assert "matches no atoms" in dialog.warnings[-1]


def test_missing_molecule_is_reported(qapp, dialog):
    dialog.mol_radio.setChecked(True)
    dialog.load_source()
    assert dialog.frames is None
    assert "current molecule" in dialog.warnings[-1]


def _result():
    return Result(
        "Test",
        "x (A)",
        "y",
        [Series("a", [0, 1, 2], [0, 1, 4]), Series("b", [0, 2], [1, 3], "dash")],
        ["x", "y"],
        [[0, 0]],
    )


def test_builtin_plot_png(qapp, tmp_path):
    widget = plot_mod.PainterPlot()
    widget.set_result(_result())
    path = tmp_path / "p.png"
    widget.save_png(str(path))
    assert path.stat().st_size > 1000
    widget.set_result(None)
    widget.save_png(str(path))


def test_matplotlib_plot_png(qapp, tmp_path):
    if not plot_mod.matplotlib_available():
        pytest.skip("matplotlib is not installed")
    widget = plot_mod.MatplotlibPlot()
    widget.set_result(_result())
    path = tmp_path / "m.png"
    widget.save_png(str(path))
    assert path.stat().st_size > 1000


def test_nice_ticks():
    assert plot_mod.nice_ticks(0, 10) == [0, 2, 4, 6, 8, 10]
    ticks = plot_mod.nice_ticks(-0.013, 0.021)
    assert ticks[0] <= -0.013 and ticks[-1] >= 0.021
    assert plot_mod.nice_ticks(5, 5)
    assert plot_mod.format_tick(0) == "0"


def test_settings_column_scrolls_and_ignores_the_wheel(qapp, dialog):
    """The settings column scrolls (Run stays reachable on a laptop screen) and an
    unfocused field does not change when the wheel scrolls over it."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QScrollArea

    assert dialog.minimumSizeHint().height() < 600
    (area,) = [a for a in dialog.findChildren(QScrollArea) if a.widget() is not None]
    assert not area.widget().isAncestorOf(dialog.run_button)
    spin = dialog.stride_spin
    assert area.widget().isAncestorOf(spin)
    assert spin.focusPolicy() == Qt.FocusPolicy.StrongFocus
    before = spin.value()
    event = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(spin, event)
    assert spin.value() == before
