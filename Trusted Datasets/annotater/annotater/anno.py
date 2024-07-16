# General Imports
import threading, os, json, cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from imutils.video import FileVideoStream

# Local Imports
from annotater.setup import file_setup, change_directory
from annotater.player import VideoPlayer
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
    # get which file to watch
    watch_file = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")], title="Select a file to watch", initialdir=config.out_path)
    meta_file = watch_file.replace("_annotated.mp4", "_annotated.json")

    if not os.path.exists(watch_file):
        messagebox.showerror("Error", "File not found.")
        return

    if not os.path.exists(meta_file):
        messagebox.showerror("Error", "File metadata not found.")
        return
    
    # json file
    # example: { "223": ["start", [339, 39]], "234": ["move", [182, 201]], "235": ["move", [180, 202]], "245": ["end", [180, 202]], "metadata": { "video_name": "zhusky.mp4", "frame_rate": 30, "frame_count": 10225, "frame_width": 640, "frame_height": 360, "sample_rate": 44100, "channels": 2 }}
    meta = json.load(open(meta_file, "r"))

    # open video player and display annotations
    vs = FileVideoStream('result.mp4').start()
    # time.sleep(0.2)

    print("Playing video...")

    while vs.more():
        frame = vs.read()
        if frame is None:
            break

        frame_number = int(vs.stream.get(cv2.CAP_PROP_POS_FRAMES))
        if str(frame_number) in meta:
            action, point = meta[str(frame_number)]
            if action == "start":
                cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
            elif action == "move":
                cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
            elif action == "end":
                cv2.circle(frame, tuple(point), 5, (255, 0, 0), -1)

        cv2.imshow("Annotated Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()

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