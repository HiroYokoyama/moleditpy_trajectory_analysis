"""Trajectory Analysis plugin for MoleditPy.

RDF, MSD and diffusion coefficients, RMSD/RMSF, radius of gyration, density profiles
and per-frame energies for multi-frame PDB, GRO, XYZ and SDF files or the conformers
of the current molecule.  Frames are read lazily from memory-mapped files and the
analyses run in a background thread with progress and cancel.
"""

import logging

_log = logging.getLogger(__name__)

PLUGIN_NAME = "Trajectory Analysis"
PLUGIN_VERSION = "0.1.1"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Analyse MD trajectories (multi-frame PDB/GRO/XYZ, SDF conformers): RDF with "
    "minimum image, MSD and diffusion, RMSD/RMSF with alignment, radius of "
    "gyration, density profiles and per-frame energies, with CSV/PNG export."
)
PLUGIN_CATEGORY = "Analysis"
PLUGIN_TAGS = ["Analysis"]
PLUGIN_DEPENDENCIES = []
PLUGIN_OPTIONAL_DEPENDENCIES = ["matplotlib"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=5.0.0, <6.0.0"
PLUGIN_LICENSE = "LGPL-3.0-or-later"

WINDOW_ID = "trajectory_analysis_dialog"
MENU_LABEL = "Trajectory Analysis..."

_context = None
_dialog_opened = False


def get_default_settings():
    from .analysis import DEFAULTS

    settings = dict(DEFAULTS)
    settings.update({"source": "file", "path": "", "analysis": "rdf"})
    return settings


current_settings = {}


def _open_dialog(mw=None):
    """Open (or raise) the dialog.  Not named ``run`` so the host adds no extra entry."""
    global _dialog_opened

    if _context is not None:
        mw = _context.get_main_window()
        existing = _context.get_window(WINDOW_ID)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return existing

    from .main_dialog import TrajectoryAnalysisDialog

    def _get_molecule():
        try:
            if _context is not None:
                return _context.current_molecule
        except Exception as exc:  # noqa: BLE001 - host API guard
            _log.warning("%s: could not read the molecule: %s", PLUGIN_NAME, exc)
        return getattr(mw, "current_mol", None)

    def _mark_modified():
        if _context is not None:
            try:
                _context.mark_project_modified()
            except Exception as exc:  # noqa: BLE001 - host API guard
                _log.debug("mark_project_modified failed: %s", exc)

    if not current_settings:
        current_settings.update(get_default_settings())
    _dialog_opened = True
    dlg = TrajectoryAnalysisDialog(
        parent=mw,
        persistent_settings=current_settings,
        get_molecule=_get_molecule,
        mark_modified=_mark_modified,
    )
    if _context is not None:
        _context.register_window(WINDOW_ID, dlg)
    dlg.show()
    return dlg


def initialize(context):
    """Entry point for the plugin API; called once when MoleditPy starts."""
    global _context
    _context = context

    context.add_analysis_tool(MENU_LABEL, lambda *_args: _open_dialog())

    def save_state():
        # Nothing is written into projects in which the dialog was never used.
        if not _dialog_opened:
            return {}
        return {"settings": dict(current_settings)}

    def load_state(data):
        global _dialog_opened
        if not isinstance(data, dict):
            return
        saved = data.get("settings")
        if not isinstance(saved, dict):
            return
        current_settings.clear()
        current_settings.update(get_default_settings())
        current_settings.update(saved)
        _dialog_opened = True
        dlg = context.get_window(WINDOW_ID)
        if dlg is not None:
            try:
                dlg.apply_settings(current_settings)
            except Exception as exc:  # noqa: BLE001 - host API guard
                _log.warning("%s: could not apply loaded state: %s", PLUGIN_NAME, exc)

    def handle_reset():
        global _dialog_opened
        dlg = context.get_window(WINDOW_ID)
        if dlg is not None and dlg.isVisible():
            # An open dialog keeps its settings; the user may still be working.
            return
        current_settings.clear()
        _dialog_opened = False

    context.register_save_handler(save_state)
    context.register_load_handler(load_state)
    context.register_document_reset_handler(handle_reset)
