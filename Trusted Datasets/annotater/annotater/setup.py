import os
from customtkinter import filedialog
from tkinter import messagebox
from config import config

def refetch_files():
    # TODO : attach to button in annotater
    in_path, out_path = config.in_path, config.out_path
    files = [f for f in os.listdir(in_path) if f.endswith(".mp4")]
    files = [f for f in files if f not in os.listdir(out_path) if f.endswith(".mp4")]
    config.update(in_path, out_path, files)
    return files

def file_setup():
    global in_path, out_path, files
    global cwd_label, in_label, out_label, file_label

    # get current working directory
    cwd = os.getcwd()
    print(f"Current Working Directory: {cwd}")
    
    # Select Input Video Directory
    if not os.path.exists(f"{cwd}/data"): in_path = filedialog.askdirectory(title="Select Input Directory", initialdir=cwd)
    else: in_path = f"{cwd}/data"

    # Select Output Directory
    if not os.path.exists(f"{cwd}/out"): out_path = filedialog.askdirectory(title="Select Output Directory", initialdir=cwd)
    else: out_path = f"{cwd}/out"

    # get list of names all mp4 files in self.file_path
    in_files = [f for f in os.listdir(in_path) if f.endswith(".mp4")]

    if len(in_files) == 0: 
        messagebox.showerror("Error", "No MP4 files found in the input directory.")
        return None, None, None
    
    # remove all files from in_files that are already in out_files
    files = [f for f in in_files if f not in os.listdir(out_path) if f.endswith(".mp4")]

    if len(files) == 0:
        messagebox.showinfo("Info", "All files have been annotated.")
        return None, None, None
    
    config.update(in_path, out_path, files)
    return in_path, out_path, files

def change_directory():
    new_directory = filedialog.askdirectory(title="Select New Directory")
    if new_directory:
        os.chdir(new_directory)
        print(f"Current working directory changed to: {new_directory}")
        in_path, out_path, files = file_setup()
        return in_path, out_path, files, new_directory