# Copyright (C) 2025 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Entry point for NotNormal GUI.
"""

from notnormal.gui import not_normal_gui as nng
from multiprocessing import freeze_support


def main() -> None:
    freeze_support()
    app = nng.NotNormalGUI()
    app.mainloop()
    app.quit()


if __name__ == "__main__":
    main()