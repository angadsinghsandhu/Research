# imports
import customtkinter as ctk
from player import VideoPlayer
from customtkinter import filedialog
from tkinter import messagebox
from PIL import Image
import os

# TODO: Add draw annotations

# TODO: test Surface audio signal

# TODO: test annotations recorded 240p and recoreded in 1080p, scaled down to 240p

# Global Variables
in_path, out_path, file_name = None, None, None
files = []
cwd_label, in_label, out_label, file_label = None, None, None, None

# Functions
def show_splash(root):
    # FIXME : understand why image is not showing on splash screen
    splash = ctk.CTkToplevel(root)
    splash.title("Loading...")

    # Center the splash screen
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    window_width = int(screen_width//3)
    window_height = int(screen_height//3)
    position_top = (screen_height // 2) - (window_height // 2)
    position_right = (screen_width // 2) - (window_width // 2)
    splash.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    # Make the splash screen topmost
    splash.attributes("-topmost", True)

    # Load and resize the image
    image_path = "./imgs/jhu.png"
    max_image_width = window_width - 40  # Max width for the image with padding
    max_image_height = window_height // 2  # Max height for the image

    if os.path.exists(image_path):
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Scale down the image if it is too big
        if img_width > max_image_width or img_height > max_image_height:
            scaling_factor = min(max_image_width / img_width, max_image_height / img_height)
            new_width = int(img_width * scaling_factor)
            new_height = int(img_height * scaling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        img_ctk = ctk.CTkImage(img)
        # Create and place the image label
        image_label = ctk.CTkLabel(splash, image=img_ctk)
        image_label.pack(pady=20)
    else:
        print(f"Image not found: {image_path}")

    # Create and place the text label
    label = ctk.CTkLabel(splash, text="Welcome to the Annotater Application", font=("Arial", 16))
    label.pack(pady=10)

    countdown_label = ctk.CTkLabel(splash, text="Closing in 3 seconds", font=("Courier", 12))
    countdown_label.pack(pady=10)

    def update_countdown(count):
        if count > 0:
            countdown_label.configure(text=f"Closing in {count} seconds")
            splash.after(1000, update_countdown, count-1)
        else:
            splash.destroy()
            run(root)

    window_position_label = ctk.CTkLabel(splash, text=f"Window Position: {splash.winfo_x()}, {splash.winfo_y()}")
    window_position_label.pack(pady=2)

    # update window position label on window move
    def update_window_position(event):
        window_position_label.configure(text=f"Window Position: {splash.winfo_x()}, {splash.winfo_y()}")

    splash.bind("<Configure>", update_window_position)


    splash.update()
    update_countdown(3)

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

def change_directory():
    new_directory = filedialog.askdirectory(title="Select New Directory")
    if new_directory:
        os.chdir(new_directory)
        print(f"Current working directory changed to: {new_directory}")
        file_setup()

def annotate(app):
    global in_path, out_path, files
    global cwd_label, in_label, out_label, file_label

    for file_name in files:
        print(f"Annotating {file_name}")
        file_label.configure(text=f"File Name: {file_name}") # update file name

        VideoPlayer(app, in_path, file_name, out_path)

    # # close the application
    # messagebox.showinfo("Info", "All files have been annotated.")

def run(app):
    global in_path, out_path, files, file_name
    global cwd_label, in_label, out_label, file_label
    # TODO : Startup Screen

    # Labels
    cwd_label = ctk.CTkLabel(app, text=f"Current Working Directory: {os.getcwd()}")    # show currect working directory
    in_label = ctk.CTkLabel(app, text=f"Input Directory: {in_path}")                    # show input directory
    out_label = ctk.CTkLabel(app, text=f"Output Directory: {out_path}")
    file_label = ctk.CTkLabel(app, text=f"File Name: {file_name}")
    cwd_label.pack(pady=20), in_label.pack(pady=2), out_label.pack(pady=2), file_label.pack(pady=2)

    # add and update (screen width, screen length, window height, window width, window position x and y) location tags
    screen_size_label = ctk.CTkLabel(app, text=f"Screen Width: {app.winfo_screenwidth()}, Screen Height: {app.winfo_screenheight()}")
    window_size_label = ctk.CTkLabel(app, text=f"Window Width: {app.winfo_width()}, Window Height: {app.winfo_height()}")
    window_position_label = ctk.CTkLabel(app, text=f"Window Position: {app.winfo_x()}, {app.winfo_y()}")

    # pack the labels
    screen_size_label.pack(pady=2), window_size_label.pack(pady=2), window_position_label.pack(pady=2)

    # update window position label on window move
    def update_window_position(event):
        window_position_label.configure(text=f"Window Position: {app.winfo_x()}, {app.winfo_y()}")

    # Handle window move event
    app.bind("<Configure>", update_window_position)

    # Setup Files
    file_setup()

    change_dir_button = ctk.CTkButton(app, text="Change Directory", command=change_directory)
    change_dir_button.pack(pady=40)

    # Open Video Player Button
    video_button = ctk.CTkButton(app, text="Begin Annotating", command=lambda: annotate(app))
    video_button.pack(pady=20)

# run main loop
if __name__ == "__main__":

    # Set the theme (optional)
    ctk.set_appearance_mode("Dark")  # Can be "Dark" or "Light"

    # Create the main application window
    app = ctk.CTk()
    app.title("Annotater") # Set the title of the window
    app.protocol("WM_DELETE_WINDOW", lambda: app.destroy())

    # Show splash screen
    show_splash(app)

    app.iconbitmap("./imgs/tool.ico")

    # add default styling options
    ctk.set_default_color_theme("dark-blue")  # Set the default color theme
    # ctk.set_icon("")  # Set the icon of the window
    # ctk.set_font("Arial", 12)  # Set the default font and size
    # ctk.set_bg("black")  # Set the default background color
    # ctk.set_fg("white")  # Set the default foreground color

    # Set geometry to the full screen
    screen_width = int(app.winfo_screenwidth()/2)
    screen_height = int(app.winfo_screenheight()/2)
    offset = 1/4
    # geo = f"{screen_width}x{screen_height}+{offset*screen_width}+{offset*screen_height}"
    geo = f"{screen_width}x{screen_height}-9-9"
    print(geo)
    app.geometry(geo)

    # Start the main application loop
    app.mainloop()