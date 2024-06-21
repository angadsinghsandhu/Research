import dearpygui.dearpygui as dpg
from window_manager import WindowManager
from windows.main_window import MainWindow
from windows.secondary_window import SecondaryWindow
from windows.tertiary_window import TertiaryWindow
from style import apply_styles, load_fonts

def main():
    dpg.create_context()

    # Load fonts and apply styles
    load_fonts()
    apply_styles()

    window_manager = WindowManager()
    
    # Instantiate windows
    main_window = MainWindow(window_manager)
    secondary_window = SecondaryWindow(window_manager)
    tertiary_window = TertiaryWindow(window_manager)

    # Register windows with the manager
    window_manager.add_window("main", main_window)
    window_manager.add_window("secondary", secondary_window)
    window_manager.add_window("tertiary", tertiary_window)

    # Show the main window initially
    window_manager.show_window("main")

    dpg.create_viewport(title='Multi-Window App', width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Ensure the event loop runs
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        
    dpg.destroy_context()

if __name__ == "__main__":
    main()
