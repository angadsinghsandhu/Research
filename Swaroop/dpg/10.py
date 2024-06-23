import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Introduction App', width=600, height=300)

dpg.configure_app(docking=True, docking_space=True)

def print_callback(sender, app_data, user_data):
    print(f"Sender '{sender}' returned: {app_data}")
    if user_data: print(f"User Data: {user_data}")

with dpg.window(tag="win1", label="Window 1") as primary_window:
    dpg.add_text("Hello, world!")

with dpg.window(tag="win2", label="Window 2") as primary_window:
    dpg.add_text("Hello, There!")

dpg.setup_dearpygui()
dpg.show_viewport()
# dpg.set_primary_window("win1", True)
dpg.start_dearpygui()
dpg.destroy_context()