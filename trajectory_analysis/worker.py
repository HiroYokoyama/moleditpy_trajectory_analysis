"""Background thread that runs one job (opening a file or an analysis)."""

import traceback

from PySide6.QtCore import QThread, Signal

from .analysis import Cancelled


class JobThread(QThread):
    """Run ``job(progress, cancelled)`` off the GUI thread.

    Emits ``progress(done, total)`` while running, then exactly one of
    ``succeeded(result)``, ``failed(message)`` or ``was_cancelled()``.
    """

    progress = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)
    was_cancelled = Signal()

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._job = job
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def is_cancelled(self):
        return self._cancel

    def _report(self, done, total):
        self.progress.emit(int(done), int(total))

    def run(self):
        try:
            result = self._job(self._report, self.is_cancelled)
        except Cancelled:
            self.was_cancelled.emit()
            return
        except (ValueError, OSError, MemoryError) as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)
            return
        except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
            self.failed.emit(
                f"{exc.__class__.__name__}: {exc}\n\n{traceback.format_exc(limit=6)}"
            )
            return
        if self._cancel:
            self.was_cancelled.emit()
        else:
            self.succeeded.emit(result)
