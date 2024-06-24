import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Introduction App', width=600, height=300)

def print_callback(sender, app_data, user_data):
    print(f"Sender '{sender}' returned: {app_data}")
    if user_data: print(f"User Data: {user_data}")

with dpg.window(tag="win1", label="Window 1") as primary_window:
    dpg.add_button(tag="btn1", label="Button 1", callback=print_callback)
    with dpg.popup(tag="popup1", parent="btn1", mousebutton=dpg.mvMouseButton_Left, modal=True):
        dpg.add_text("This is a popup")
        dpg.add_button(tag="btn2", label="Close ", callback=lambda: dpg.configure_item("popup1", show=False))

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("win1", True)
dpg.start_dearpygui()
dpg.destroy_context()