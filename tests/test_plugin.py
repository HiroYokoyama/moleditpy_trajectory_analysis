"""initialize() and the session handlers against a fake PluginContext."""

import pytest

pytest.importorskip("numpy")

import trajectory_analysis as plugin


class FakeWindow:
    def __init__(self):
        self.visible = True
        self.applied = None

    def isVisible(self):
        return self.visible

    def apply_settings(self, settings):
        self.applied = dict(settings)


class FakeContext:
    def __init__(self):
        self.tools = []
        self.save = self.load = self.reset = None
        self.windows = {}

    def add_analysis_tool(self, label, callback):
        self.tools.append((label, callback))

    def register_save_handler(self, cb):
        self.save = cb

    def register_load_handler(self, cb):
        self.load = cb

    def register_document_reset_handler(self, cb):
        self.reset = cb

    def get_window(self, window_id):
        return self.windows.get(window_id)


@pytest.fixture
def ctx():
    plugin.current_settings.clear()
    plugin._dialog_opened = False
    context = FakeContext()
    plugin.initialize(context)
    yield context
    plugin._context = None
    plugin.current_settings.clear()
    plugin._dialog_opened = False


def test_registers_analysis_tool(ctx):
    assert [label for label, _ in ctx.tools] == ["Trajectory Analysis..."]


def test_save_is_empty_until_used(ctx):
    assert ctx.save() == {}


def test_load_then_save_round_trip(ctx):
    window = FakeWindow()
    ctx.windows[plugin.WINDOW_ID] = window
    ctx.load({"settings": {"analysis": "msd", "dt": 2.0}})
    assert window.applied["analysis"] == "msd"
    saved = ctx.save()["settings"]
    assert saved["analysis"] == "msd"
    assert saved["dt"] == 2.0
    assert saved["rdf_bins"] == plugin.get_default_settings()["rdf_bins"]


def test_load_ignores_garbage(ctx):
    ctx.load(None)
    ctx.load({"settings": "x"})
    assert ctx.save() == {}


def test_reset_keeps_open_dialog_and_clears_closed(ctx):
    ctx.load({"settings": {"analysis": "rg"}})
    window = FakeWindow()
    ctx.windows[plugin.WINDOW_ID] = window
    ctx.reset()
    assert ctx.save()["settings"]["analysis"] == "rg"
    window.visible = False
    ctx.reset()
    assert ctx.save() == {}
