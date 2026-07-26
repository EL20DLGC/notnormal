# Copyright (C) 2025-2026 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Entry point for NotNormal GUI.
"""

from notnormal.gui import not_normal_gui as nng
from multiprocessing import freeze_support
from sys import platform


def _set_dpi_aware():
    if platform != 'win32':
        return

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            from ctypes import windll
            windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main() -> None:
    freeze_support()
    _set_dpi_aware()
    app = nng.NotNormalGUI()
    app.mainloop()
    app.quit()


if __name__ == "__main__":
    main()