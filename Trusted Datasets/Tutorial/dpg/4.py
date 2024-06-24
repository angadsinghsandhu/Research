import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="Parent Stack System Test App", width=800, height=800)

# windows and stacks
with dpg.window(tag="win1", label="Window 1"):
    dpg.add_button(tag="btn1", label="Button 1")
    with dpg.tab_bar(tag="tab_bar1"):
        with dpg.tab(tag="tab1", label="Tab 1"):
            dpg.add_checkbox(tag="chk1", label="Checkbox 1")
        with dpg.tab(tag="tab2", label="Tab 2"):
            dpg.add_checkbox(tag="chk2", label="Checkbox 2")
    dpg.add_radio_button(tag="rad1", items=["Radio 1", "Radio 2", "Radio 3"])

# add buttons to the parent stack system by using the parent and before parameter
dpg.add_button(tag="btn2", label="Button 2", before="tab_bar1")
dpg.add_button(tag="btn3", label="Button 3", parent="win1")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()