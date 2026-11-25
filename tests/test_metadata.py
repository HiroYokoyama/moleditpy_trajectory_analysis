"""Registry metadata, read statically so it runs without any host dependency."""

import ast
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "trajectory_analysis" / "__init__.py"


def _constants():
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith("PLUGIN_"):
                out[target.id] = ast.literal_eval(node.value)
    return out


def test_required_metadata():
    c = _constants()
    assert c["PLUGIN_NAME"] == "Trajectory Analysis"
    assert c["PLUGIN_VERSION"] == "0.1.0"
    assert c["PLUGIN_AUTHOR"] == "HiroYokoyama"
    assert c["PLUGIN_CATEGORY"] == "Analysis"
    assert c["PLUGIN_SUPPORTED_MOLEDITPY_VERSION"] == ">=5.0.0, <6.0.0"
    assert c["PLUGIN_LICENSE"] == "LGPL-3.0-or-later"
    assert c["PLUGIN_DEPENDENCIES"] == []
    assert "matplotlib" in c["PLUGIN_OPTIONAL_DEPENDENCIES"]
    assert c["PLUGIN_DESCRIPTION"]


def test_package_name_matches_plugin_name():
    c = _constants()
    assert c["PLUGIN_NAME"].lower().replace(" ", "_") == INIT.parent.name


def test_no_run_function():
    # The host adds a Plugins-menu entry for any module-level run(); the plugin
    # registers its Analysis-menu entry itself.
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "run" not in names
    assert "initialize" in names
