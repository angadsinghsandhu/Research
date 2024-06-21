import dearpygui.dearpygui as dpg

class SecondaryWindow:
    def __init__(self, window_manager):
        self.window_manager = window_manager
        with dpg.window(label="Secondary Window", tag="secondary_window", width=-1, height=-1):
            dpg.add_text("This is the Secondary Window")
            dpg.add_button(label="Open Tertiary Window", callback=self.open_tertiary)
            dpg.hide_item("secondary_window")  # Hide window initially

    def open_tertiary(self):
        self.window_manager.show_window("tertiary")

    def show(self):
        dpg.show_item("secondary_window")

    def hide(self):
        dpg.hide_item("secondary_window")
