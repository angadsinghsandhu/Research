# General Imports
import threading, logging
import customtkinter as ctk
from tkinter import filedialog

# Local Imports
from annotater.setup import align_window
from annotater.player import VideoPlayer, AnnotatedPlayer
from config import config

# Set up logging
logger = logging.getLogger('app')

def change(app, current_file_label):
    try:
        config.change_directory()
        current_file_label.configure(text=f"Current File to be Annotated: {config.fetch_top_file}")
        logger.info("Directory changed")
    except Exception as e:
        logger.exception(f"Error changing directory: {e}")

def annotate(app, current_file_label):
    file_name = config.fetch_top_file
    if not file_name:
        current_file_label.configure(text="All files have been annotated.")
        logger.debug("All files have been annotated.")
    else:
        logger.info(f"Starting annotation for file: {file_name}")
        done_event = threading.Event()

        try:
            player = VideoPlayer(app, file_name, done_event=done_event)
            done_event.wait()
            _ = config.refetch_files()
            current_file_label.configure(text=f"Current File to be Annotated: {config.fetch_top_file}")
            logger.info(f"Annotation completed for file: {file_name}")
        except Exception as e:
            logger.exception(f"Error during annotation for file: {file_name}: {e}")

def refresh(current_file_label):
    try:
        _ = config.refetch_files()
        current_file_label.configure(text=f"Current File to be Annotated: {config.fetch_top_file}")
        logger.info("File list refreshed")
    except Exception as e:
        logger.exception(f"Error refreshing file list: {e}")

def watch():
    try:
        # get which file to watch
        watch_file = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")], title="Select a file to watch", initialdir=config.out_path)
        if watch_file:
            meta_file = watch_file.replace("_annotated.mp4", "_annotated.json")
            logger.info(f"Started watching file: {watch_file}")
            annoPlayer = AnnotatedPlayer(watch_file, meta_file)
        else:
            logger.warning("No file selected for watching")
    except Exception as e:
        logger.exception(f"Error during watching: {e}")

# Functions
def create_annotater(app):
    logger.info("Creating annotater")
    try:
        app.iconbitmap("./imgs/tool.ico")

        # add default styling options
        ctk.set_default_color_theme("dark-blue")  # Set the default color theme
        app.option_add("*Font", "Courier 12")  # Set the default font and size
        logger.debug("Default styling options set")

        # Set geometry 
        (mid_x, mid_y), (window_width, window_height), (screen_width, screen_height) = align_window(app, 400, 350)

        app.resizable(False, False)

        # Setup Files
        config.file_setup()
        logger.info("File setup completed")
        
        # Add a label to the window
        change_dir_button = ctk.CTkButton(app, text="Change Directory", command=lambda: change(app, current_file_label))
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

        logger.info("Annotater UI setup completed")

        # Start the main application loop
        app.mainloop()
        logger.info("Main application loop started")
    except Exception as e:
        logger.exception(f"An error occurred during annotater creation: {e}")