import cv2

import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, messagebox, StringVar

from PIL import Image

from player import open_player

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

import subprocess, threading, time

from data import _data

def run(app):
    # global save_annotations_button
    # TODO : Startup Screen

    # Open Video Player Button
    video_button = ctk.CTkButton(app, text="Open Video", command=open_player)
    video_button.pack(pady=20)

# run main loop
if __name__ == "__main__":
    # Set the theme (optional)
    ctk.set_appearance_mode("Dark")  # Can be "Dark" or "Light"

    # Create the main application window
    app = ctk.CTk()

    app.title("Annotater") # Set the title of the window

    # add default styling options
    ctk.set_default_color_theme("dark-blue")  # Set the default color theme
    # ctk.set_icon("path/to/icon.png")  # Set the icon of the window
    # ctk.set_font("Arial", 12)  # Set the default font and size
    # ctk.set_bg("black")  # Set the default background color
    # ctk.set_fg("white")  # Set the default foreground color
    # TODO : set geometry to the full screen
    app.geometry("600x350")  # Set the size of the window

    run(app)

    # Start the main application loop
    app.mainloop()