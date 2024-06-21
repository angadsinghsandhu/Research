# main.py
"""
Main script to initialize and run the Dear PyGui application with webcam video streaming.
"""

import dearpygui.dearpygui as dpg
from modules import video_utils, gui_utils
from state_manager import StateManager
from states import AppState

def main():
    # Initialize Dear PyGui
    dpg.create_context()
    dpg.create_viewport(title='Custom Title', width=600, height=800)
    dpg.setup_dearpygui()

    # Initialize video and get the first frame
    video_utils.initialize_video()

    # Create GUI windows
    gui_utils.create_settings_window()
    gui_utils.create_main_window()
    gui_utils.create_help_window()
    gui_utils.create_about_window()

    # Set the initial state to settings
    state_manager = StateManager()
    state_manager.transition_to(AppState.SETTINGS)

    # Show Dear PyGui metrics and viewport
    dpg.show_metrics()
    dpg.show_viewport()

    # Main loop to update texture with webcam frames
    while dpg.is_dearpygui_running():
        video_utils.update_frame()
        dpg.render_dearpygui_frame()

    # Cleanup
    video_utils.cleanup_video()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
