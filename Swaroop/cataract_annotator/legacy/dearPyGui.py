import cv2
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
import subprocess
import dearpygui.dearpygui as dpg

# # pip installs
# pip install opencv-python
# pip install numpy
# pip install scipy
# pip install sounddevice
# pip install dearpygui

# Initialize the drawing flag and last point
drawing = False
last_point = None
frames = []
audio_data = []
samplerate = 44100
channels = 2

# Define the callback function for drawing annotations
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

# Function to handle audio recording
def audio_callback(indata, frames, time, status):
    audio_data.append(indata.copy())

# Function to start video playback
def play_video():
    global play
    play = True

# Function to pause video playback
def pause_video():
    global play
    play = False

# Function to handle the main loop for video processing and annotation
def video_loop():
    global last_frame, play, cap, frame_delay, window_name
    if cap.isOpened():
        if play:
            ret, frame = cap.read()
            if ret:
                last_frame = frame.copy()
                frames.append(last_frame)
                dpg.set_value("video_texture", cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        dpg.render_dearpygui_frame()

# Setup Dear PyGui
dpg.create_context()

# Setup the GUI for file selection
def file_callback(sender, app_data):
    global video_path
    video_path = app_data['file_path_name']
    dpg.configure_item("file_dialog_id", show=False)
    setup_video_player()

def setup_video_player():
    global cap, fps, frame_size, frame_delay, play
    # Video capture and window setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    frame_delay = int((1/fps)*1000)
    play = False

    # Create a texture for displaying the video
    dpg.add_texture_registry(label="texture_registry")
    dpg.add_dynamic_texture(width=frame_size[0], height=frame_size[1], default_value=cv2.cvtColor(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8), cv2.COLOR_BGR2RGB), tag="video_texture", parent="texture_registry")

    with dpg.window(label="Video Player", width=frame_size[0], height=frame_size[1] + 60):
        dpg.add_image("video_texture")
        dpg.add_button(label="Play", callback=play_video)
        dpg.add_same_line()
        dpg.add_button(label="Pause", callback=pause_video)

    # Main loop for Dear PyGui and video processing
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback):
        while dpg.is_dearpygui_running():
            video_loop()

# Create file dialog
with dpg.file_dialog(directory_selector=False, show=True, callback=file_callback, id="file_dialog_id"):
    dpg.add_file_extension(".*")
    dpg.add_file_extension(".mp4", color=(150, 255, 150, 255))

# Start Dear PyGui
dpg.create_viewport(title='Cataract-1K Video Annotation Tool', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

# Save the annotated video along with recorded audio
output_video_filename = dpg.get_value("Output Video") or "output_video"
output_audio_filename = dpg.get_value("Output Audio") or "output_audio"

output_video_path = f"{output_video_filename}.mp4"
output_audio_path = f"{output_audio_filename}.wav"

# Save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
for frame in frames:
    out.write(frame)
out.release()

# Save audio
audio_array = np.concatenate(audio_data, axis=0)
write(output_audio_path, samplerate, audio_array)
audio_delay = -1

# Combine audio and video using FFmpeg
command = [
    'ffmpeg',
    '-i', output_video_path,
    '-itsoffset', str(audio_delay),
    '-i', output_audio_path,
    '-map', '0:v',
    '-map', '1:a',
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-strict', 'experimental',
    '-shortest',
    output_video_path.replace('.mp4', '_annotated.mp4')
]
subprocess.run(command)
