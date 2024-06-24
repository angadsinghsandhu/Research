import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="Parent Stack System Test App", width=800, height=800)

def print_callback(sender, app_data):
    # get the value from the registry
    print(dpg.get_value("com"))

with dpg.value_registry():
    dpg.add_value_registry(label="registry", tag="abc")
    dpg.add_bool_value(label="bool1", tag="xyz", default_value=True, parent="abc")
    dpg.add_int_value(label="int1", tag="com", default_value=0)

# add window
with dpg.window(tag="win1", label="Window 1"):
    # add checkbox
    dpg.add_checkbox(tag="c1", label="Checkbox 1")
    
    # add input float
    dpg.add_input_float(tag="f1", label="Input Float 1")

    # # input int and int3 slider with same tag
    dpg.add_input_int(tag="it1", label="Input Int 1", source="com", callback=print_callback)
    dpg.add_drag_int(tag="di3", label="Slider Int 1", source="com", callback=print_callback)

# dpg developer tools
# dpg.show_documentation()
# dpg.show_style_editor()
dpg.show_debug()
# dpg.show_about()
dpg.show_metrics()
# dpg.show_font_manager()
dpg.show_item_registry()

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()