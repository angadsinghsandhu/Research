# General Imports
import os, logging, ctypes
from customtkinter import filedialog
from tkinter import messagebox

# Set up logging
logger = logging.getLogger('app')

def align_window(app, window_width=1/2, window_height=1/2, horizontal="center", vertical="center", offset=-9, top_bar=29):
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()

    if window_width and window_width <= 1: window_width *= screen_width
    if window_height and window_height <= 1: window_height *= screen_height

    app.geometry(f"{window_width}x{window_height}")

    while app.winfo_width() == 200 or app.winfo_height() == 200:
        app.update()
        app.after(1000)

    window_width, window_height = app.winfo_width(), app.winfo_height()

    # find distace of window from right
    if horizontal == "center": position_right = (screen_width / 2) - (window_width / 2)
    elif horizontal == "left": position_right = 0
    elif horizontal == "right": position_right = screen_width - window_width
    else: position_right = (screen_width / 2) - (window_width / 2)

    # find distance of window from top
    if vertical == "center": position_top = (screen_height / 2) - (window_height / 2)
    elif vertical == "top": position_top = 0
    elif vertical == "bottom": position_top = screen_height - window_height
    else: position_top = (screen_height / 2) - (window_height / 2)

    # calculate the mid point of the window
    mid_x, mid_y = int(position_right+offset), int(position_top-top_bar+offset)

    app.geometry(f"+{mid_x}+{mid_y}")
    app.update()

    print(f"==> Screen Width: {screen_width}, Screen Height: {screen_height}, Window Width: {window_width}, Window Height: {window_height}, position_right: {position_right}, position_top: {position_top}")

    return (mid_x, mid_y), (int(window_width), int(window_height)), (int(screen_width), int(screen_height))
