import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import threading
from time import sleep

# Global variables
drawing = False
last_point = None
frames = []
play = False
cap = None
fps = None
frame_size = None
frame_delay = None
last_frame = None

def draw_annotation(sender, app_data):
    global last_point, drawing, last_frame, play
    if not play:
        x, y = dpg.get_mouse_pos()
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            if not drawing:
                drawing = True
                last_point = (x, y)
            else:
                cv2.line(last_frame, last_point, (x, y), (0, 255, 0), 1)
                frames.append(last_frame.copy())
                last_point = (x, y)
                dpg.set_value("video_texture", cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        else:
            if drawing:
                drawing = False
                cv2.line(last_frame, last_point, (x, y), (0, 255, 0), 1)
                frames.append(last_frame.copy())
                dpg.set_value("video_texture", cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))

def play_video():
    global play
    play = True

def pause_video():
    global play
    play = False

def video_loop2():
    global cap, play, last_frame
    if cap and cap.isOpened():
        if play:
            ret, frame = cap.read()
            if ret:
                last_frame = frame.copy()
                dpg.set_value("camera_texture", cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))

def setup_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Unable to open the webcam")
    
    # Create texture registry at a higher scope
    with dpg.texture_registry(label="texture_registry") as texture_registry:
        dpg.add_dynamic_texture(width=640, height=480, default_value=np.zeros((480, 640, 3), dtype=np.uint8), tag="camera_texture")

    with dpg.window(label="Camera Feed", width=640, height=520, pos=(50, 50)):
        dpg.add_image("camera_texture")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play", callback=play_video)
            dpg.add_button(label="Pause", callback=pause_video)
    
    # # Start a DearPyGui timer to update the video feed
    # dpg.set_frame_callback(video_loop2)

    # Start a thread to update the video feed
    def video_thread():
        while True:
            video_loop2()
            sleep(1/30)  # Update the video feed at ~30 FPS
    
    threading.Thread(target=video_thread, daemon=True).start()

def video_loop():
    global last_frame, play, cap, frame_delay
    if cap and cap.isOpened():
        if play:
            ret, frame = cap.read()
            if ret:
                last_frame = frame.copy()
                frames.append(last_frame)
                dpg.set_value("video_texture", cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        dpg.render_dearpygui_frame()

def setup_video_player(video_path):
    global cap, fps, frame_size, frame_delay, play
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    frame_delay = int((1/fps)*1000)
    play = False

    with dpg.texture_registry(label="texture_registry"):
        dpg.add_dynamic_texture(width=frame_size[0], height=frame_size[1], default_value=cv2.cvtColor(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8), cv2.COLOR_BGR2RGB), tag="video_texture")

    with dpg.window(label="Video Player", width=frame_size[0], height=frame_size[1] + 60):
        dpg.add_image("video_texture")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play", callback=play_video)
            dpg.add_button(label="Pause", callback=pause_video)

    # Start a thread to update the video feed
    def video_thread():
        while True:
            video_loop()
            sleep(1/30)  # Update the video feed at ~30 FPS
    
    threading.Thread(target=video_thread, daemon=True).start()
