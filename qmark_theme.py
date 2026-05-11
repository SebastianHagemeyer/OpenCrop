"""Match the qmark dashboard's chrome when launched as its subprocess.

Reads two env vars passed by qmark's `_launch_sibling_tool`:
  QMARK_THEME           "light" or "dark" (default "light")
  QMARK_FONT_SCALE_PCT  integer 50-300 (default 100)

Falls back to the light palette when launched standalone, so the tool
still has consistent chrome on its own. Self-contained — does not import
anything from the qmark-app repo so this file can ship in OpenName /
OpenCrop independently.
"""
from __future__ import annotations

import ctypes
import os
import sys

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


PALETTES = {
    "light": {
        "bg":         "#f0f0f0",
        "fg":         "#000000",
        "panel_bg":   "#ffffff",
        "panel_fg":   "#000000",
        "subtle_fg":  "#444444",
        "select_bg":  "#0078d7",
        "select_fg":  "#ffffff",
        "border":     "#bbbbbb",
        "accent":     "#1565c0",
    },
    "dark": {
        "bg":         "#2b2b2b",
        "fg":         "#e6e6e6",
        "panel_bg":   "#1e1e1e",
        "panel_fg":   "#e6e6e6",
        "subtle_fg":  "#a0a0a0",
        "select_bg":  "#1565c0",
        "select_fg":  "#ffffff",
        "border":     "#3c3c3c",
        "accent":     "#64b5f6",
    },
}


def _build_palette(p: dict) -> QPalette:
    qp = QPalette()
    bg = QColor(p["bg"])
    fg = QColor(p["fg"])
    panel_bg = QColor(p["panel_bg"])
    panel_fg = QColor(p["panel_fg"])
    subtle = QColor(p["subtle_fg"])
    qp.setColor(QPalette.Window, bg)
    qp.setColor(QPalette.WindowText, fg)
    qp.setColor(QPalette.Base, panel_bg)
    qp.setColor(QPalette.AlternateBase, bg)
    qp.setColor(QPalette.Text, panel_fg)
    qp.setColor(QPalette.Button, bg)
    qp.setColor(QPalette.ButtonText, fg)
    qp.setColor(QPalette.Highlight, QColor(p["select_bg"]))
    qp.setColor(QPalette.HighlightedText, QColor(p["select_fg"]))
    qp.setColor(QPalette.PlaceholderText, subtle)
    qp.setColor(QPalette.ToolTipBase, panel_bg)
    qp.setColor(QPalette.ToolTipText, panel_fg)
    qp.setColor(QPalette.Light, bg.lighter(120))
    qp.setColor(QPalette.Midlight, bg.lighter(110))
    qp.setColor(QPalette.Dark, bg.darker(140))
    qp.setColor(QPalette.Mid, bg.darker(120))
    qp.setColor(QPalette.Shadow, QColor("#000000"))
    qp.setColor(QPalette.BrightText, QColor("#ffffff"))
    qp.setColor(QPalette.Link, QColor(p["accent"]))
    qp.setColor(QPalette.LinkVisited, QColor(p["accent"]))
    for role in (QPalette.WindowText, QPalette.Text,
                 QPalette.ButtonText, QPalette.HighlightedText):
        qp.setColor(QPalette.Disabled, role, subtle)
    return qp


def _build_stylesheet(p: dict) -> str:
    return f"""
    QToolTip {{
        color: {p['panel_fg']};
        background-color: {p['panel_bg']};
        border: 1px solid {p['border']};
    }}
    QGroupBox {{
        border: 1px solid {p['border']};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 4px;
        color: {p['fg']};
    }}
    QMenuBar {{ background-color: {p['bg']}; color: {p['fg']}; }}
    QMenuBar::item:selected {{
        background-color: {p['select_bg']};
        color: {p['select_fg']};
    }}
    QMenu {{
        background-color: {p['bg']};
        color: {p['fg']};
        border: 1px solid {p['border']};
    }}
    QMenu::item:selected {{
        background-color: {p['select_bg']};
        color: {p['select_fg']};
    }}
    QStatusBar {{ background-color: {p['bg']}; color: {p['fg']}; }}
    QHeaderView::section {{
        background-color: {p['bg']};
        color: {p['fg']};
        border: 1px solid {p['border']};
        padding: 2px 6px;
    }}
    QListWidget, QTreeWidget, QPlainTextEdit, QTextEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {p['panel_bg']};
        color: {p['panel_fg']};
        border: 1px solid {p['border']};
        selection-background-color: {p['select_bg']};
        selection-color: {p['select_fg']};
    }}
    QComboBox {{
        background-color: {p['panel_bg']};
        color: {p['panel_fg']};
        border: 1px solid {p['border']};
        padding: 2px 6px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p['panel_bg']};
        color: {p['panel_fg']};
        selection-background-color: {p['select_bg']};
        selection-color: {p['select_fg']};
    }}
    QPushButton {{
        background-color: {p['bg']};
        color: {p['fg']};
        border: 1px solid {p['border']};
        padding: 4px 10px;
        border-radius: 3px;
    }}
    QPushButton:hover {{
        background-color: {p['select_bg']};
        color: {p['select_fg']};
    }}
    QPushButton:disabled {{ color: {p['subtle_fg']}; }}
    QTabWidget::pane {{ border: 1px solid {p['border']}; }}
    QTabBar::tab {{
        background: {p['bg']};
        color: {p['fg']};
        padding: 4px 12px;
        border: 1px solid {p['border']};
    }}
    QTabBar::tab:selected {{
        background: {p['select_bg']};
        color: {p['select_fg']};
    }}
    QScrollBar:vertical, QScrollBar:horizontal {{
        background: {p['panel_bg']};
        border: 0;
    }}
    QScrollBar::handle {{ background: {p['border']}; border-radius: 3px; }}
    QCheckBox, QRadioButton {{ color: {p['fg']}; spacing: 6px; }}
    """


def _set_window_dark_titlebar(widget, dark: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        hwnd = int(widget.winId())
    except Exception:
        return
    if not hwnd:
        return
    try:
        dwmapi = ctypes.windll.dwmapi
    except (AttributeError, OSError):
        return
    value = ctypes.c_int(1 if dark else 0)
    for attr in (20, 19):
        try:
            if dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)
            ) == 0:
                return
        except Exception:
            pass


class _DarkTitlebarFilter(QObject):
    """Re-applies the dark titlebar to every top-level window the first
    time it is shown, so dialogs created after startup match."""

    def __init__(self, dark: bool) -> None:
        super().__init__()
        self._dark = dark

    def eventFilter(self, obj, ev):
        if (ev.type() == QEvent.Show
                and obj.isWidgetType()
                and obj.isWindow()):
            _set_window_dark_titlebar(obj, self._dark)
        return False


def apply_qmark_theme(app: QApplication) -> str:
    """Apply the theme handed off by qmark (or light as a standalone default).

    Returns the active theme name ("light" or "dark").
    """
    name = (os.environ.get("QMARK_THEME") or "light").strip().lower()
    if name not in PALETTES:
        name = "light"
    p = PALETTES[name]

    try:
        scale_pct = int(os.environ.get("QMARK_FONT_SCALE_PCT") or 100)
    except ValueError:
        scale_pct = 100
    scale_pct = max(50, min(300, scale_pct))

    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    app.setPalette(_build_palette(p))
    app.setStyleSheet(_build_stylesheet(p))

    f = app.font()
    f.setPointSize(max(1, int(round(9 * scale_pct / 100.0))))
    app.setFont(f)

    dark = (name == "dark")
    if not hasattr(app, "_qmark_dark_filter"):
        app._qmark_dark_filter = _DarkTitlebarFilter(dark)
        app.installEventFilter(app._qmark_dark_filter)
    else:
        app._qmark_dark_filter._dark = dark
    for w in app.topLevelWidgets():
        _set_window_dark_titlebar(w, dark)

    return name
