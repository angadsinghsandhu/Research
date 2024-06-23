import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Introduction App', width=600, height=300)

def call(sender, app_data, user_data):
    print(f"Sender '{sender}' returned: {app_data}")
    if user_data: print(f"User Data: {user_data}")

def align(sender, app_data, user_data):
    for i in dpg.get_all_items():
        if (dpg.get_item_type(i) == "mvAppItemType::mvText"):
            length_of_item = (dpg.get_item_rect_size(i))[0]
            height_of_item = (dpg.get_item_rect_size(i))[1]
            height_of_item_from_top = dpg.get_item_pos(i)[1]
            # get the y coordinate of the child from the top of the page
            length_of_parent = dpg.get_item_rect_size(dpg.get_item_parent(i))[0]
            height_of_parent = dpg.get_item_rect_size(dpg.get_item_parent(i))[1]

            print(f"height_of_item: {height_of_item}")
            print(f"height_of_parent: {height_of_parent}")
            print(f"height_of_item_from_top: {height_of_item_from_top}")
            print("=====================================")

            # dpg.set_item_pos(i, pos=(((length_of_parent - length_of_item) / 2), height_of_item_from_top))

# update progress bar
def update_progress_bar(sender, app_data, user_data):
    dpg.configure_item("progress1", default_value=app_data/100.0)


with dpg.window(tag="Tutorial") as tutorial_window:
    dpg.add_button(label="Press me!", tag="btn1", callback= lambda: dpg.set_primary_window("Win2", True))
    dpg.add_checkbox(label="Check me!", tag="chk1", callback=call)
    dpg.add_text(label="Label 1", tag="lbl1", default_value="Value")
    # dpg.add_spacer(width=100, height=100)
    dpg.add_text(label="Label 2", tag="lbl2", default_value="Label", color=(255, 0, 0, 255))

    dpg.add_input_int(label="Input Int", tag="int1", callback=call)
    dpg.add_drag_int(label="Drag Int", tag="int2", callback=call)
    dpg.add_slider_int(label="Slider Int", tag="int3", callback=update_progress_bar)

    # add raido button with combo and listbox
    dpg.add_radio_button(tag="radio1", items=["Item a", "Item b"], callback=call)
    dpg.add_combo(tag="combo1", items=["Item a", "Item b"], callback=call)
    dpg.add_listbox(tag="listbox1", items=["Item a", "Item b"], callback=call)

    # progress bar
    dpg.add_progress_bar(tag="progress1", default_value=0.5, overlay="Progress")

with dpg.window(tag="Win2") as window2:
    dpg.add_text(label="Label 3", tag="lbl3", default_value="Hello There!")
    dpg.add_button(label="Press me!", tag="btn2", callback= lambda: dpg.set_primary_window("Tutorial", True))

with dpg.item_handler_registry(label="handler", tag="align_handler") as align_handler:
    dpg.add_item_resize_handler(callback=align)
dpg.bind_item_handler_registry("Tutorial", "align_handler")


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Tutorial", True)
dpg.start_dearpygui()
dpg.destroy_context()