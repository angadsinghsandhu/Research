import dearpygui.dearpygui as dpg

class TertiaryWindow:
    def __init__(self, window_manager):
        self.window_manager = window_manager
        with dpg.window(label="Tertiary Window", tag="tertiary_window", width=-1, height=-1):
            dpg.add_text("This is the Tertiary Window")
            dpg.add_button(label="Back to Main Window", callback=self.back_to_main)
            dpg.hide_item("tertiary_window")  # Hide window initially

    def back_to_main(self):
        self.window_manager.show_window("main")

    def show(self):
        dpg.show_item("tertiary_window")

    def hide(self):
        dpg.hide_item("tertiary_window")
