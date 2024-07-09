# imports
import customtkinter as ctk
from player import VideoPlayer
from customtkinter import filedialog
from tkinter import messagebox
import os

# Global Variables
in_path, out_path, file_name = None, None, None
files = []
cwd_label, in_label, out_label, file_label = None, None, None, None

# Functions
def file_setup():
    global in_path, out_path, files
    global cwd_label, in_label, out_label, file_label

    # get current working directory
    cwd = os.getcwd()
    print(f"Current Working Directory: {cwd}")

    cwd_label.configure(text=f"Current Working Directory: {os.getcwd()}") # update current working directory
    
    # Select Input Video Directory
    if not os.path.exists(f"{cwd}/data"): in_path = filedialog.askdirectory(title="Select Input Directory", initialdir=cwd)
    else: in_path = f"{cwd}/data"

    in_label.configure(text=f"Input Directory: {in_path}") # update input directory

    # Select Output Directory
    if not os.path.exists(f"{cwd}/out"): out_path = filedialog.askdirectory(title="Select Output Directory", initialdir=cwd)
    else: out_path = f"{cwd}/out"

    out_label.configure(text=f"Output Directory: {out_path}") # update output directory

    # get list of names all mp4 files in self.file_path
    in_files = [f for f in os.listdir(in_path) if f.endswith(".mp4")]

    if len(in_files) == 0: 
        messagebox.showerror("Error", "No MP4 files found in the input directory.")
        return
    
    # remove all files from in_files that are already in out_files
    files = [f for f in in_files if f not in os.listdir(out_path) if f.endswith(".mp4")]

    if len(files) == 0:
        messagebox.showinfo("Info", "All files have been annotated.")
        return
    else:
        file_label.configure(text=f"File Name: {files[0]}") # update file name

def annotate():
    global in_path, out_path, files
    global cwd_label, in_label, out_label, file_label

    for file_name in files:
        print(f"Annotating {file_name}")
        file_label.configure(text=f"File Name: {file_name}") # update file name

        player = VideoPlayer(in_path, file_name, out_path)

def run(app):
    global in_path, out_path, files, file_name
    global cwd_label, in_label, out_label, file_label
    # TODO : Startup Screen

    # Labels
    cwd_label = ctk.CTkLabel(app, text=f"Current Working Directory: {os.getcwd()}")     # show currect working directory
    cwd_label.pack(pady=20)
    in_label = ctk.CTkLabel(app, text=f"Input Directory: {in_path}")                    # show input directory
    out_label = ctk.CTkLabel(app, text=f"Output Directory: {out_path}")
    file_label = ctk.CTkLabel(app, text=f"File Name: {file_name}")

    # Setup Files
    file_setup()

    # Change Directory Button
    def change_directory():
        new_directory = filedialog.askdirectory(title="Select New Directory")
        if new_directory:
            os.chdir(new_directory)
            print(f"Current working directory changed to: {new_directory}")
            file_setup()

    change_dir_button = ctk.CTkButton(app, text="Change Directory", command=change_directory)
    change_dir_button.pack(pady=40)

    # begin annotating
    annotate()

    # Open Video Player Button
    video_button = ctk.CTkButton(app, text="Begin Annotating", command=annotate)
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