# state_manager.py
"""
Manages the state transitions between different windows.
"""

import dearpygui.dearpygui as dpg
from states import AppState

class StateManager:
    def __init__(self):
        self.current_state = None

    def transition_to(self, new_state):
        if self.current_state is not None:
            self.hide_current_state()
        self.current_state = new_state
        self.show_current_state()

    def hide_current_state(self):
        if self.current_state == AppState.MAIN:
            dpg.hide_item("main_window")
        elif self.current_state == AppState.SETTINGS:
            dpg.hide_item("settings_window")
        elif self.current_state == AppState.HELP:
            dpg.hide_item("help_window")
        elif self.current_state == AppState.ABOUT:
            dpg.hide_item("about_window")
        elif self.current_state == AppState.WEBCAM:
            dpg.hide_item("webcam_window")

    def show_current_state(self):
        if self.current_state == AppState.MAIN:
            dpg.show_item("main_window")
        elif self.current_state == AppState.SETTINGS:
            dpg.show_item("settings_window")
        elif self.current_state == AppState.HELP:
            dpg.show_item("help_window")
        elif self.current_state == AppState.ABOUT:
            dpg.show_item("about_window")
        elif self.current_state == AppState.WEBCAM:
            dpg.show_item("webcam_window")
            # Initialize video only when transitioning to the webcam state
            from modules.video_utils import initialize_video
            initialize_video()
