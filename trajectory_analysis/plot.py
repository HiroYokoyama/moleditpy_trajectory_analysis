"""Plot widget: matplotlib when it is installed, otherwise a small QPainter plot.

matplotlib is an optional dependency; the fallback draws axes, ticks, series and a
legend so every analysis is usable without it.  Both paths export PNG.
"""

import logging
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QVBoxLayout, QWidget

_log = logging.getLogger(__name__)
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")


def matplotlib_available():
    """True when matplotlib is installed and its Qt backend uses PySide6.

    matplotlib uses the Qt binding that is already imported (PySide6 inside
    MoleditPy); if another binding won, its canvas cannot go into this dialog and
    the built-in plot is used instead.
    """
    try:
        from matplotlib.backends import qt_compat
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401
        from matplotlib.figure import Figure  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means no matplotlib
        return False
    return str(getattr(qt_compat, "QT_API", "")).lower() == "pyside6"


def nice_ticks(lo, hi, target=6):
    """Round tick values covering [lo, hi]."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0, 1.0]
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    raw = (hi - lo) / max(1, target)
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for factor in (1, 2, 2.5, 5, 10):
        step = factor * mag
        if raw <= step:
            break
    first = math.floor(lo / step + 1e-9)
    last = math.ceil(hi / step - 1e-9)
    ticks = [round(k * step, 12) for k in range(first, last + 1)][:50]
    return ticks


def format_tick(value):
    if value == 0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.6g}"


def _finite_pairs(series):
    pts = []
    for x, y in zip(list(series.x), list(series.y)):
        x, y = float(x), float(y)
        if math.isfinite(x) and math.isfinite(y):
            pts.append((x, y))
    return pts


class PainterPlot(QWidget):
    """Dependency-free line plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.setMinimumSize(320, 220)

    def set_result(self, result):
        self.result = result
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        try:
            self.draw(painter, self.width(), self.height())
        finally:
            painter.end()

    def draw(self, painter, width, height):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(QRectF(0, 0, width, height), QColor("white"))
        painter.setPen(QColor("black"))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        if self.result is None or not self.result.series:
            painter.drawText(
                QRectF(0, 0, width, height),
                Qt.AlignmentFlag.AlignCenter,
                "Run an analysis to see a plot.",
            )
            return
        series = [(s, _finite_pairs(s)) for s in self.result.series]
        pts = [p for _, ps in series for p in ps]
        if not pts:
            painter.drawText(
                QRectF(0, 0, width, height), Qt.AlignmentFlag.AlignCenter, "No data"
            )
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xt = nice_ticks(min(xs), max(xs))
        yt = nice_ticks(min(ys), max(ys))
        x0, x1 = xt[0], xt[-1]
        y0, y1 = yt[0], yt[-1]
        left, right, top, bottom = 70.0, 15.0, 30.0, 45.0
        pw = max(10.0, width - left - right)
        ph = max(10.0, height - top - bottom)

        def to_px(x, y):
            return QPointF(
                left + (x - x0) / (x1 - x0 or 1.0) * pw,
                top + ph - (y - y0) / (y1 - y0 or 1.0) * ph,
            )

        painter.drawText(
            QRectF(0, 4, width, 20), Qt.AlignmentFlag.AlignHCenter, self.result.title
        )
        grid = QPen(QColor("#dddddd"))
        axis_pen = QPen(QColor("black"))
        for t in xt:
            p = to_px(t, y0)
            painter.setPen(grid)
            painter.drawLine(QPointF(p.x(), top), QPointF(p.x(), top + ph))
            painter.setPen(axis_pen)
            painter.drawText(
                QRectF(p.x() - 40, top + ph + 4, 80, 16),
                Qt.AlignmentFlag.AlignHCenter,
                format_tick(t),
            )
        for t in yt:
            p = to_px(x0, t)
            painter.setPen(grid)
            painter.drawLine(QPointF(left, p.y()), QPointF(left + pw, p.y()))
            painter.setPen(axis_pen)
            painter.drawText(
                QRectF(0, p.y() - 8, left - 6, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                format_tick(t),
            )
        painter.setPen(axis_pen)
        painter.drawRect(QRectF(left, top, pw, ph))
        painter.drawText(
            QRectF(left, height - 20, pw, 18),
            Qt.AlignmentFlag.AlignHCenter,
            self.result.xlabel,
        )
        painter.save()
        painter.translate(12, top + ph / 2)
        painter.rotate(-90)
        painter.drawText(
            QRectF(-ph / 2, -8, ph, 18),
            Qt.AlignmentFlag.AlignHCenter,
            self.result.ylabel,
        )
        painter.restore()

        legend_y = top + 6
        for k, (s, ps) in enumerate(series):
            color = QColor(COLORS[k % len(COLORS)])
            pen = QPen(color, 1.6)
            if s.style == "dash":
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            if s.style == "points" or len(ps) == 1:
                painter.setBrush(color)
                for x, y in ps:
                    painter.drawEllipse(to_px(x, y), 2.5, 2.5)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                path = QPainterPath(to_px(*ps[0]))
                for x, y in ps[1:]:
                    path.lineTo(to_px(x, y))
                painter.drawPath(path)
            if len(series) > 1:
                painter.drawLine(
                    QPointF(left + pw - 150, legend_y + 6),
                    QPointF(left + pw - 125, legend_y + 6),
                )
                painter.setPen(axis_pen)
                painter.drawText(QPointF(left + pw - 120, legend_y + 10), s.label)
                legend_y += 16

    def save_png(self, path, width=1000, height=650):
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(QColor("white"))
        painter = QPainter(image)
        try:
            self.draw(painter, width, height)
        finally:
            painter.end()
        if not image.save(path, "PNG"):
            raise OSError(f"could not write {path}")


class MatplotlibPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(6, 4), dpi=100, layout="tight")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        try:
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

            layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        except Exception as exc:  # noqa: BLE001 - toolbar is cosmetic
            _log.debug("no matplotlib toolbar: %s", exc)
        self.result = None
        self.set_result(None)

    def set_result(self, result):
        self.result = result
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if result is None or not result.series:
            ax.text(
                0.5, 0.5, "Run an analysis to see a plot.", ha="center", va="center"
            )
            ax.set_axis_off()
        else:
            for k, s in enumerate(result.series):
                color = COLORS[k % len(COLORS)]
                if s.style == "points":
                    ax.plot(s.x, s.y, "o", ms=3, color=color, label=s.label)
                elif s.style == "dash":
                    ax.plot(s.x, s.y, "--", color=color, label=s.label)
                else:
                    ax.plot(s.x, s.y, "-", lw=1.4, color=color, label=s.label)
            ax.set_title(result.title)
            ax.set_xlabel(result.xlabel.replace("A^2", "Å²").replace("(A)", "(Å)"))
            ax.set_ylabel(
                result.ylabel.replace("A^2", "Å²")
                .replace("(A)", "(Å)")
                .replace("A^3", "Å³")
                .replace("cm^3", "cm³")
            )
            ax.grid(True, alpha=0.3)
            if len(result.series) > 1:
                ax.legend()
        self.canvas.draw_idle()

    def save_png(self, path):
        self.figure.savefig(path, dpi=150)


def make_plot_widget(parent=None, prefer_matplotlib=True):
    """(widget, backend_name)."""
    if prefer_matplotlib and matplotlib_available():
        try:
            return MatplotlibPlot(parent), "matplotlib"
        except Exception as exc:  # noqa: BLE001 - broken matplotlib install
            _log.warning("matplotlib plot failed, using the built-in plot: %s", exc)
    return PainterPlot(parent), "builtin"
