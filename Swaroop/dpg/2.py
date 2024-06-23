import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Callbacks Test App', width=600, height=300)

# create a callback function
def call(sender, app_data, user_data):
    print(f"Sender: {sender}")
    print(f"App Data: {app_data}")
    if user_data:
        print(f"User Data: {user_data}")

    # change callback of a widget
    if sender != "slide": dpg.set_item_callback("slide", callback=call2)

def call2(sender, app_data):
    print(f"Sender 2: {sender}")
    print(f"App Data 2: {app_data}")

with dpg.window(tag="Tutorial"):
    # set callback and user data
    dpg.add_button(tag="btn1", label="Hello World!", callback=call, user_data="yo")
    dpg.add_slider_float(tag="slide", label="Slider", callback=call)

dpg.setup_dearpygui()
dpg.show_viewport()
# dpg.set_primary_window("Tutorial", True)
dpg.start_dearpygui()
dpg.destroy_context()
