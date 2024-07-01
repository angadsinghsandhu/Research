import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, messagebox, StringVar
import cv2
from PIL import Image

# Global variables to store user input
user_name = ""
user_level = ""

def validate_inputs(name, level):
    return bool(name) and level != "Select Level"

def save_info(name, level):
    global user_name, user_level
    user_name = name
    user_level = level
    messagebox.showinfo("Info", f"Name: {user_name}, Level: {user_level}")

def open_video():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        play_video(file_path)

def play_video(file_path):
    cap = cv2.VideoCapture(file_path)
    
    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            ctk_img = CTkImage(light_image=img, size=(frame.shape[1], frame.shape[0]))  # Use CTkImage
            video_label.configure(image=ctk_img)
            video_label.image = ctk_img  # Keep a reference to the image to avoid garbage collection
            video_label.after(10, update_frame)
        else:
            cap.release()

    video_window = ctk.CTkToplevel()
    video_window.title("Video Player")
    video_label = ctk.CTkLabel(video_window)
    video_label.pack()
    update_frame()

def run(app):
    # TODO : Startup Screen

    # TODO : Ask to select Video (save video name for reference later)
    # Create a frame for user inputs
    input_frame = ctk.CTkFrame(app)
    input_frame.pack(pady=20)

    # Name Label and Text Field
    name_label = ctk.CTkLabel(input_frame, text="Name:")
    name_label.grid(row=0, column=0, padx=10, pady=10)
    name_entry = ctk.CTkEntry(input_frame, width=200)
    name_entry.grid(row=0, column=1, padx=10, pady=10)

    # Level Label and Dropdown
    level_label = ctk.CTkLabel(input_frame, text="Level:")
    level_label.grid(row=1, column=0, padx=10, pady=10)
    level_var = StringVar(value="Select Level")
    level_dropdown = ctk.CTkOptionMenu(input_frame, variable=level_var, values=["Beginner", "Intermediate", "Advanced"])
    level_dropdown.grid(row=1, column=1, padx=10, pady=10)

    # Save Button
    save_button = ctk.CTkButton(input_frame, text="Save", state="disabled", command=lambda: save_info(name_entry.get(), level_var.get()))
    save_button.grid(row=2, columnspan=2, pady=20)

    # Video Button
    video_button = ctk.CTkButton(app, text="Open Video", command=open_video)
    video_button.pack(pady=20)

    def check_validity(*args):
        if validate_inputs(name_entry.get(), level_var.get()):
            save_button.configure(state="normal")
        else:
            save_button.configure(state="disabled")

    # Bind validation check to name entry and level dropdown changes
    name_entry.bind("<KeyRelease>", check_validity)
    level_var.trace("w", check_validity)

    # TODO : open video in a window (add play/pause button and seeker, record audio and annotations)

    # TODO : Save annotations and audio with the orignal video using ffmpg (ask for save location)

# run main loop
if __name__ == "__main__":
    # Set the theme (optional)
    ctk.set_appearance_mode("Dark")  # Can be "Dark" or "Light"

    # Create the main application window
    app = ctk.CTk()

    app.title("CustomTkinter Example") # Set the title of the window

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