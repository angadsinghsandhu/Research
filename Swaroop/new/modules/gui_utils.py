# modules/gui_utils.py
"""
Utility functions for creating the GUI using Dear PyGui.
"""

import dearpygui.dearpygui as dpg
from state_manager import StateManager
from states import AppState

state_manager = StateManager()

def create_main_window():
    """
    Create the main window with the video stream.
    """
    with dpg.window(label="Main Window", tag="main_window", show=False):
        dpg.add_text("Webcam Stream")
        dpg.add_image("texture_tag")
        dpg.add_button(label="Settings", callback=lambda: state_manager.transition_to(AppState.SETTINGS))
        dpg.add_button(label="Help", callback=lambda: state_manager.transition_to(AppState.HELP))

def create_settings_window():
    """
    Create a settings window.
    """
    with dpg.window(label="Settings", tag="settings_window", show=False):
        dpg.add_text("Settings")
        dpg.add_checkbox(label="Option 1", tag="option1")
        dpg.add_checkbox(label="Option 2", tag="option2")
        dpg.add_button(label="Main", callback=lambda: state_manager.transition_to(AppState.MAIN))
        dpg.add_button(label="Start Webcam", callback=lambda: state_manager.transition_to(AppState.WEBCAM))

def create_help_window():
    """
    Create a help window.
    """
    with dpg.window(label="Help", tag="help_window", show=False):
        dpg.add_text("Help")
        dpg.add_button(label="About", callback=lambda: state_manager.transition_to(AppState.ABOUT))
        dpg.add_button(label="Main", callback=lambda: state_manager.transition_to(AppState.MAIN))

def create_about_window():
    """
    Create an about window.
    """
    with dpg.window(label="About", tag="about_window", show=False):
        dpg.add_text("This is the about window.")
        dpg.add_button(label="Help", callback=lambda: state_manager.transition_to(AppState.HELP))
