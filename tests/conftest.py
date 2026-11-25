import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MOLEDITPY_HEADLESS", "1")
# The plugin runs on PySide6; keep pytest-qt and matplotlib on the same binding.
os.environ["PYTEST_QT_API"] = "pyside6"
os.environ["QT_API"] = "pyside6"


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
