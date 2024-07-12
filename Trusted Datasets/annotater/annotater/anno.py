import threading, os
import customtkinter as ctk
from annotater.player import VideoPlayer
from annotater.setup import file_setup, change_directory
from tkinter import messagebox

# Global Variables
in_path, out_path, file_name = None, None, None
files = []

def annotate(app):
    global in_path, out_path, files

    for file_name in files:
        print(f"Annotating {file_name}")
        done_event = threading.Event()
        VideoPlayer(app, in_path, file_name, out_path, done_event=done_event)
        done_event.wait()

    # close the application
    messagebox.showinfo("Info", "All files have been annotated.")

def run(app):
    global in_path, out_path, files

    # Setup Files
    in_path, out_path, files = file_setup()

    change_dir_button = ctk.CTkButton(app, text="Change Directory", command=change_directory)
    change_dir_button.pack(pady=40)

    # Open Video Player Button
    video_button = ctk.CTkButton(app, text="Begin Annotating", command=lambda: annotate(app))
    video_button.pack(pady=20)

# Functions
def create_annotater(app):
    app.iconbitmap("./imgs/tool.ico")

    # add default styling options
    ctk.set_default_color_theme("dark-blue")  # Set the default color theme
    # ctk.set_font("Courier", 12)  # Set the default font and size
    app.option_add("*Font", "Courier 12")  # Set the default font and size

    # Set geometry to the full screen
    screen_width = 400
    screen_height = 200
    offset = 1/4
    geo = f"{screen_width}x{screen_height}+{offset*screen_width}+{offset*screen_height}"
    # geo = f"{screen_width}x{screen_height}-9-9"
    print(f"Geometry: {geo}")
    app.geometry(geo)

    # update window position
    app.update()

    # Run the main application loop
    run(app)

    # Start the main application loop
    app.mainloop()