import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Introduction App', width=600, height=300)

def print_callback(sender, app_data, user_data):
    print(f"Sender '{sender}' returned: {app_data}")
    if user_data: print(f"User Data: {user_data}")

with dpg.window(tag="win1", label="Window 1") as primary_window:
    
    with dpg.tab_bar(tag="tab_bar1", callback=print_callback):
        with dpg.tab(tag="tab1", label="Tab 1", closable=False):
            dpg.add_button(tag="btn1", label="Button 1")
        with dpg.tab(tag="tab2", label="Tab 2", closable=True):
            dpg.add_button(tag="btn2", label="Button 2")
        dpg.add_tab_button(tag="tab_btn1", label="Tab Button 1", callback=print_callback)


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("win1", True)
dpg.start_dearpygui()
dpg.destroy_context()