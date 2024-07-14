# General Imports
import os
from customtkinter import filedialog
from tkinter import messagebox

# Local Imports
from config import config

def file_setup():
    # get current working directory
    cwd = os.getcwd()
    
    # Select Input Video Directory
    if not os.path.exists(f"{cwd}\data"): in_path = filedialog.askdirectory(title="Select Input Directory", initialdir=cwd)
    else: in_path = f"{cwd}\data"

    # Select Output Directory
    if not os.path.exists(f"{cwd}\out"): out_path = filedialog.askdirectory(title="Select Output Directory", initialdir=cwd)
    else: out_path = f"{cwd}\out"

    # get list of names all mp4 files in self.file_path
    in_files = [f for f in os.listdir(in_path) if f.endswith(".mp4")]

    if len(in_files) == 0: 
        messagebox.showerror("Error", "No MP4 files found in the input directory.")
    
    files = config.refetch_files(in_path, out_path)

def change_directory():
    new_directory = filedialog.askdirectory(title="Select New Directory")
    if new_directory:
        os.chdir(new_directory)
        print(f"Current working directory changed to: {new_directory}")
        in_path, out_path, files = file_setup()
        return in_path, out_path, files, new_directory