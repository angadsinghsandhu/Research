import dearpygui.dearpygui as dpg

class WindowManager:
    def __init__(self):
        self.windows = {}
        self.current_window = None

    def add_window(self, name, window):
        self.windows[name] = window

    def show_window(self, name):
        if self.current_window:
            self.windows[self.current_window].hide()
        self.current_window = name
        self.windows[name].show()
