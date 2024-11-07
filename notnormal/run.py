"""
Used to run the NotNormal GUI as the GUI is now compiled.
"""


from notnormal import not_normal_gui as nng
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    app = nng.NotNormalGUI()
    app.mainloop()
    app.quit()
