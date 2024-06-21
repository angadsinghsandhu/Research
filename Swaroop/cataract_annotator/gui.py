import dearpygui.dearpygui as dpg
import sounddevice as sd
from cataract_annotator.video_processing import setup_video_player, video_loop, video_loop2, setup_camera
from cataract_annotator.audio_processing import audio_callback
from cataract_annotator.utils import save_annotation_files

# Global variables
user_name = ""
user_level = ""
samplerate = 44100
channels = 2

def start_callback(sender, app_data):
    global user_name, user_level
    user_name = dpg.get_value("Name")
    user_level = dpg.get_value("Level")
    dpg.configure_item("start_window", show=False)
    dpg.show_item("main_window")
    # dpg.configure_item("file_dialog_id", show=True)

    # dpg.configure_item("setup_camera", show=True)
    setup_camera()

def file_callback(sender, app_data):
    video_path = app_data['file_path_name']
    dpg.configure_item("file_dialog_id", show=False)
    setup_video_player(video_path)

def start_gui():
    dpg.create_context()
    dpg.create_viewport(title='Cataract-1K Video Annotation Tool', width=800, height=600)
    dpg.setup_dearpygui()
    print("..")

    # Create starting page
    with dpg.window(label="Start", tag="start_window"):
        dpg.add_text("Enter your details to start annotating the video:")
        dpg.add_input_text(label="Name", tag="Name")
        dpg.add_combo(label="Level", items=["Beginner", "Intermediate", "Advanced"], tag="Level")
        dpg.add_button(label="Start", callback=start_callback)

    # Create main window and set it to be hidden initially
    with dpg.window(label="Main Window", tag="main_window", show=False):
        dpg.add_text("Main annotation window")
        # Add additional UI elements here as needed

    # Create file dialog
    with dpg.file_dialog(directory_selector=False, show=False, callback=file_callback, tag="file_dialog_id"):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".mp4", color=(150, 255, 150, 255))

    dpg.show_viewport()
    dpg.show_item("start_window")

    # Enter the main loop
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback):
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            video_loop2()

    dpg.destroy_context()
    save_annotation_files(user_name, user_level)

    # with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback):
    #     while dpg.is_dearpygui_running():
    #         # print("??")
    #         video_loop2()
    #         # video_loop()

    # dpg.start_dearpygui()
    # dpg.destroy_context()

    # save_annotation_files(user_name, user_level)

    # # # Create file dialog
    # # with dpg.file_dialog(directory_selector=False, show=False, callback=file_callback, id="file_dialog_id"):
    # #     dpg.add_file_extension(".*")
    # #     dpg.add_file_extension(".mp4", color=(150, 255, 150, 255))

    # # print("++")

    

    # # print("++")

    # # with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback):
    # #     while dpg.is_dearpygui_running():
    # #         print("??")
    # #         video_loop()
    # #         print("??")

    # # print("--")

    # # dpg.start_dearpygui()
    # # dpg.destroy_context()

    # # print("--")

    # # save_annotation_files(user_name, user_level)

    # # print("--")
