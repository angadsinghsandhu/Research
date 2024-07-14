# General Imports
import threading
import customtkinter as ctk
from tkinter import messagebox

# Local Imports
from annotater.player import VideoPlayer
from annotater.setup import file_setup, change_directory
from config import config

def annotate(app, current_file_label):
    file_name = config.fetch_top_file
    if not file_name:
        current_file_label.configure(text="All files have been annotated.")
    else:
        done_event = threading.Event()
        player = VideoPlayer(app, file_name, done_event=done_event)
        done_event.wait()
        _ = config.refetch_files()
        current_file_label.configure(text=f"Current File to be Annotated: {config.fetch_top_file}")

def refresh(current_file_label):
    _ = config.refetch_files()
    current_file_label.configure(text=f"Current File to be Annotated: {config.fetch_top_file}")

def watch():
    # TODO: add logic to watch video
    pass

# Functions
def create_annotater(app):
    app.iconbitmap("./imgs/tool.ico")

    # add default styling options
    ctk.set_default_color_theme("dark-blue")  # Set the default color theme
    # ctk.set_font("Courier", 12)  # Set the default font and size
    app.option_add("*Font", "Courier 12")  # Set the default font and size

    # Set geometry to the full screen
    screen_width = 400
    screen_height = 350
    offset = 1/4
    geo = f"{screen_width}x{screen_height}+{offset*screen_width}+{offset*screen_height}"
    # geo = f"{screen_width}x{screen_height}-9-9"
    print(f"Geometry: {geo}")
    app.geometry(geo)

    # Setup Files
    file_setup()
    
    # Add a label to the window
    change_dir_button = ctk.CTkButton(app, text="Change Directory", command=change_directory)
    change_dir_button.pack(pady=20)

    # Add label for the current file name
    current_file_label = ctk.CTkLabel(app, text=f"Current File to be Annotated: {config.fetch_top_file}")
    current_file_label.pack(pady=20)

    # Open Video Player Button
    video_button = ctk.CTkButton(app, text="Begin Annotating", command=lambda: annotate(app, current_file_label))
    video_button.pack(pady=20)

    # Button to refresh files
    refresh_button = ctk.CTkButton(app, text="Refresh Files", command=lambda: refresh(current_file_label))
    refresh_button.pack(pady=20)

    watch_button = ctk.CTkButton(app, text="Watch Annotated Video", command=lambda: watch())
    watch_button.pack(pady=20)

    # Start the main application loop
    app.mainloop()