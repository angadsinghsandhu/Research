import dearpygui.dearpygui as dpg

class MainWindow:
    def __init__(self, window_manager):
        self.window_manager = window_manager
        with dpg.window(label="Main Window", tag="main_window", width=-1, height=-1):
            dpg.add_text("This is the Main Window")
            dpg.add_button(label="Open Secondary Window", callback=self.open_secondary)

    def open_secondary(self):
        self.window_manager.show_window("secondary")

    def show(self):
        dpg.show_item("main_window")

    def hide(self):
        dpg.hide_item("main_window")
