import dearpygui.dearpygui as dpg

# # add_additional_font("fonts/OpenSans-VariableFont.ttf", 20)

dpg.create_context()
dpg.create_viewport(title='Introduction App', width=600, height=300)

with dpg.window(tag="Tutorial"):
    dpg.add_button(label="Hello World!")

with dpg.window(tag="Window 2"):
    dpg.add_button(label="Press me!")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Tutorial", True)
dpg.start_dearpygui()
dpg.destroy_context()