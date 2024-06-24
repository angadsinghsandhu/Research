import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Introduction App', width=600, height=300)

def print_callback(sender, app_data, user_data):
    print(f"Sender '{sender}' returned: {app_data}")
    if user_data: print(f"User Data: {user_data}")

with dpg.window(tag="win1", label="Window 1") as primary_window:
    dpg.add_simple_plot(tag="Simple Plot", default_value=[0.3, 0.9, 2.5, 8.9], histogram=True)

    dpg.add_text(tag="str", default_value="Hello, world!")

    with dpg.tooltip(tag="tip1", parent="str"):
        dpg.add_text("This is a tooltip")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("win1", True)
dpg.start_dearpygui()
dpg.destroy_context()