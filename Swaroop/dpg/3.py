import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Debug Test App', width=600, height=300)

dpg.show_debug()

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()